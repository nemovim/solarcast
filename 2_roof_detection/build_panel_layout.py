"""옥상 설치가능 마스크 안에 태양광 모듈 배열(직사각형)을 실제로 깔아 본다.

인계 마스크(building_masks/{id}/available.png)는 모양이 들쭉날쭉해서 그 면적 그대로
설치할 수는 없다. 그래서 마스크 안에 모듈을 격자로 놓아 보고, 완전히 들어가는
모듈만 남겨 줄(row) 단위 직사각형으로 묶는다. 지도에는 이 직사각형을 그리고,
설치 용량은 모듈 수 × 모듈 용량으로 센다.

  - 방향: 옥상 마스크의 최소 외접 사각형 축을 따른다. 두 축 중 동서에 가까운 축을
          줄 방향으로 삼아 모듈이 남쪽을 보게 한다.
  - 배치: 모듈 MODULE_W(줄 방향) × MODULE_D(깊이), 줄 간격 ROW_GAP(그림자·통로).
          조각마다 격자 위상을 몇 가지 밀어 보고 모듈이 가장 많이 들어가는 것을 고른다.
  - 묶기: 한 줄에서 이어진 모듈을 하나의 직사각형으로 합치고, MIN_RUN 개 미만은 버린다.

    python shadow-map-satellite/scripts/build_panel_layout.py

출력: data/rooftop_area/web/panel_layout.geojson  (인계 폴더 sam_vith_v1 은 건드리지 않는다)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "rooftop_area" / "sam_vith_v1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "2_roof" / "panel_layout.geojson"
ZOOM = 19
TILE_PX = 256
EARTH_CIRCUM = 40075016.686

MODULE_W = 1.0  # m — 줄 방향 모듈 폭
MODULE_D = 2.0  # m — 줄에 수직인 모듈 깊이 (2㎡ ≈ 400W 급)
ROW_GAP = 1.0  # m — 앞줄 그림자·유지보수 통로
MIN_RUN = 3  # 이보다 짧은 줄 조각(모듈 수)은 배열로 치지 않는다
FILL = 0.98  # 모듈 칸의 이 비율 이상이 설치가능이어야 들어간 것으로 본다
PHASES = 4  # 격자 위상을 축마다 이만큼 나눠 시도한다


# solar-panel/src/export_geojson.py 와 같은 공식. 그 모듈은 contourpy 를 불러오므로
# 좌표 변환 두 개만 옮겨 둔다.
def pixel_to_lonlat(fx: float, fy: float, zoom: int) -> tuple[float, float]:
    scale = float(TILE_PX * 2**zoom)
    longitude = fx / scale * 360.0 - 180.0
    latitude = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * fy / scale))))
    return longitude, latitude


def ground_m_per_px(latitude: float, zoom: int) -> float:
    return EARTH_CIRCUM * math.cos(math.radians(latitude)) / (TILE_PX * 2**zoom)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--precision", type=int, default=7, help="경위도 소수 자릿수")
    return ap.parse_args()


def row_angle(roof: np.ndarray, available: np.ndarray) -> float:
    """줄 방향 각도(도, 영상 x 축 기준). 옥상 마스크가 비면 설치가능 마스크로 잡는다."""

    source = roof if roof.any() else available
    points = cv2.findNonZero(source.astype(np.uint8))
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    # minAreaRect 의 angle 은 w 변의 방향이다. 두 축 중 영상 x(동서)에 가까운 쪽을 고른다.
    axis = angle if w >= h else angle + 90.0
    axis = ((axis + 90.0) % 180.0) - 90.0  # -90 ~ 90
    other = axis - 90.0 if axis > 0 else axis + 90.0
    return axis if abs(axis) <= abs(other) else other


def fit_rows(mask: np.ndarray, cell_w: float, cell_d: float, pitch: float) -> list[tuple[float, float, int]]:
    """회전된 마스크에 격자를 깔아 (x0, y0, 모듈 수) 줄 조각 목록을 낸다."""

    integral = cv2.integral(mask.astype(np.uint8))
    h, w = mask.shape

    def inside(x0: float, y0: float) -> bool:
        xa, ya = int(math.ceil(x0)), int(math.ceil(y0))
        xb, yb = int(math.floor(x0 + cell_w)), int(math.floor(y0 + cell_d))
        if xa < 0 or ya < 0 or xb > w or yb > h or xb <= xa or yb <= ya:
            return False
        total = integral[yb, xb] - integral[ya, xb] - integral[yb, xa] + integral[ya, xa]
        return total >= FILL * (xb - xa) * (yb - ya)

    best: list[tuple[float, float, int]] = []
    best_count = 0
    for py in range(PHASES):
        for px in range(PHASES):
            runs: list[tuple[float, float, int]] = []
            count = 0
            y0 = py * pitch / PHASES
            while y0 + cell_d <= h:
                x0 = px * cell_w / PHASES
                run_start, run_len = 0.0, 0
                while x0 + cell_w <= w + cell_w:  # 마지막 칸 뒤에서 열린 줄을 닫는다
                    if x0 + cell_w <= w and inside(x0, y0):
                        if run_len == 0:
                            run_start = x0
                        run_len += 1
                    else:
                        if run_len >= MIN_RUN:
                            runs.append((run_start, y0, run_len))
                            count += run_len
                        run_len = 0
                    x0 += cell_w
                y0 += pitch
            if count > best_count:
                best, best_count = runs, count
    return best


def main() -> None:
    args = parse_args()
    with (args.input / "buildings.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    features: list[dict[str, object]] = []
    total_modules = 0
    for row in rows:
        mask_path = args.input / row["available_mask_path"] if row["available_mask_path"] else None
        if not mask_path or not mask_path.exists():
            continue
        available = np.asarray(Image.open(mask_path).convert("L")) > 127
        if not available.any():
            continue
        roof_path = args.input / row["roof_mask_path"]
        roof = np.asarray(Image.open(roof_path).convert("L")) > 127 if roof_path.exists() else available

        m_per_px = ground_m_per_px(float(row["center_latitude"]), ZOOM)
        cell_w, cell_d = MODULE_W / m_per_px, MODULE_D / m_per_px
        pitch = (MODULE_D + ROW_GAP) / m_per_px

        # 줄 방향이 x 축이 되도록 돌린다. 잘리지 않게 캔버스를 대각선 길이로 키운다.
        angle = row_angle(roof, available)
        h, w = available.shape
        side = int(math.ceil(math.hypot(w, h))) + 2
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot[0, 2] += side / 2 - w / 2
        rot[1, 2] += side / 2 - h / 2
        inverse = cv2.invertAffineTransform(rot)
        turned = cv2.warpAffine(available.astype(np.uint8), rot, (side, side), flags=cv2.INTER_NEAREST)

        # 덩어리마다 위상을 따로 맞춘다 — 한 격자로 전부 덮으면 작은 덩어리가 손해를 본다
        count, labels = cv2.connectedComponents(turned, connectivity=8)
        x0g, y0g = int(row["px_x0"]), int(row["px_y0"])
        modules_here = 0
        for index in range(1, count):
            ys, xs = np.nonzero(labels == index)
            top, left = ys.min(), xs.min()
            window = labels[top:ys.max() + 1, left:xs.max() + 1] == index
            for rx, ry, n in fit_rows(window, cell_w, cell_d, pitch):
                corners = np.array([
                    [left + rx, top + ry],
                    [left + rx + n * cell_w, top + ry],
                    [left + rx + n * cell_w, top + ry + cell_d],
                    [left + rx, top + ry + cell_d],
                ])
                crop = corners @ inverse[:, :2].T + inverse[:, 2]
                ring = [
                    [round(v, args.precision) for v in pixel_to_lonlat(x0g + cx, y0g + cy, ZOOM)]
                    for cx, cy in crop
                ]
                # 영상 y 는 아래로 자라므로 픽셀에서 시계방향이면 경위도에서 반시계(RFC 7946 외곽)
                ring.append(ring[0])
                features.append({
                    "type": "Feature",
                    "properties": {
                        "building_id": row["building_id"],
                        "modules": n,
                        "area_m2": round(n * MODULE_W * MODULE_D, 1),
                        "valid": int(row["valid"]),
                    },
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                })
                modules_here += n
        total_modules += modules_here

    valid_modules = sum(f["properties"]["modules"] for f in features if f["properties"]["valid"])
    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "source": args.input.name,
            "module_m": [MODULE_W, MODULE_D],
            "row_gap_m": ROW_GAP,
            "min_run": MIN_RUN,
            "arrays": len(features),
            "modules": total_modules,
            "valid_modules": valid_modules,
        },
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collection, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"output={args.output}")
    print(f"arrays={len(features)} modules={total_modules} valid_modules={valid_modules}")
    print(f"module_area_m2={total_modules * MODULE_W * MODULE_D:.0f} "
          f"valid_module_area_m2={valid_modules * MODULE_W * MODULE_D:.0f}")
    print(f"bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
