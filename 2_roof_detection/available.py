"""설치 가능 영역(available area) 산출 파이프라인.

    A_avail = A_roof - A_shadow/불균질 - A_edge_buffer
              -> 연결성분 분리 -> 최소 면적 미만 제거

학습 모델을 쓰지 않는다. 라벨이 없어도 돌아가고, 모든 숫자의 출처를
설명할 수 있다는 것이 이 방식의 요점이다.

기존 패널 면적(A_panel)은 1번 파트(탐지) 산출물을 받아서 빼야 한다.
여기서는 참고용 힌트(panel_hint)만 색 규칙으로 계산해 오버레이에 표시한다.
힌트는 검증용이지 최종 수치가 아니다.

사용법
    python src/available.py --limit 40 --overlay      # 시범 실행 + 검증 이미지
    python src/available.py --min-roof 500            # 전체 대상 배치
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_aerial_imagery"))

import config
import tiles as T  # 1_aerial_imagery/tiles.py 단일 소스 참조


# --- 튜닝 파라미터 -------------------------------------------------------
# 전부 근거를 말할 수 있는 값이어야 한다. 임의값은 주석으로 이유를 남긴다.

EDGE_BUFFER_M = 1.0     # 옥상 가장자리 이격 (추락방지 + 유지보수 통로)
MIN_COMPONENT_M2 = 10.0  # 이보다 작은 자투리는 시공 단위가 안 나온다
PACKING_EFF = 0.70      # 순면적 -> 실설치 면적. 패널 이격/통로/미세 장애물 흡수
FLAT_STD_MAX = 12.0     # 국소 표준편차 임계. 이보다 크면 구조물/엣지로 본다
FLAT_WIN = 5            # 국소 통계 윈도 (5px ~ 1.2m)
SHADOW_GRAY_MAX = 60    # 이보다 어두우면 그림자로 본다
PANEL_GRAY_MAX = 105    # 패널 힌트: 어둡고
PANEL_BLUE_MIN = 8      # 패널 힌트: B가 R보다 이만큼 크다


@dataclass
class Params:
    edge_buffer_m: float = EDGE_BUFFER_M
    min_component_m2: float = MIN_COMPONENT_M2
    packing_eff: float = PACKING_EFF
    flat_std_max: float = FLAT_STD_MAX
    shadow_gray_max: int = SHADOW_GRAY_MAX


# --- 타일 모자이크에서 잘라내기 -----------------------------------------


class TileMosaic:
    """raw_tiles 를 전역 픽셀 좌표계로 다루는 얇은 래퍼.

    전역 픽셀 좌표: px = tile_x_float * 256, py = tile_y_float * 256
    (z19 기준). 타일은 필요할 때만 읽고 캐시한다.
    """

    def __init__(self, tile_dir: Path, z: int = config.ZOOM, tile_px: int = config.TILE_PX):
        self.dir = tile_dir
        self.z = z
        self.tp = tile_px
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _tile(self, x: int, y: int) -> np.ndarray:
        key = (x, y)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        path = self.dir / f"{x}_{y}.jpeg"
        if path.exists():
            arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        else:
            arr = np.zeros((self.tp, self.tp, 3), dtype=np.uint8)
        if len(self._cache) > 256:          # 메모리 상한. LRU 까지는 필요 없다
            self._cache.clear()
        self._cache[key] = arr
        return arr

    def crop(self, px0: int, py0: int, px1: int, py1: int) -> np.ndarray:
        """전역 픽셀 좌표 [px0,px1) x [py0,py1) 구간을 이어붙여 반환."""
        h, w = py1 - py0, px1 - px0
        out = np.zeros((h, w, 3), dtype=np.uint8)
        tx0, tx1 = px0 // self.tp, (px1 - 1) // self.tp
        ty0, ty1 = py0 // self.tp, (py1 - 1) // self.tp
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = self._tile(tx, ty)
                gx0, gy0 = tx * self.tp, ty * self.tp
                sx0, sy0 = max(px0, gx0), max(py0, gy0)
                sx1, sy1 = min(px1, gx0 + self.tp), min(py1, gy0 + self.tp)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                out[sy0 - py0:sy1 - py0, sx0 - px0:sx1 - px0] = \
                    tile[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        return out


def lonlat_to_global_px(lon: float, lat: float, z: int, tile_px: int) -> tuple[float, float]:
    tx, ty = T.deg2tile_float(lat, lon, z)
    return tx * tile_px, ty * tile_px


# --- 건물 벡터 ----------------------------------------------------------


def load_buildings(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    return gj["features"]


def polygon_rings(geom: dict) -> list[list[list[float]]]:
    """MultiPolygon / Polygon 을 외곽링 리스트로 평탄화 (구멍은 무시)."""
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]


# --- 핵심 계산 ----------------------------------------------------------


def local_std(gray: np.ndarray, win: int) -> np.ndarray:
    """윈도 내 표준편차. 균질한 평면은 낮고, 구조물 경계는 높다."""
    g = gray.astype(np.float32)
    mean = ndimage.uniform_filter(g, win)
    mean_sq = ndimage.uniform_filter(g * g, win)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var)


def disk(radius: int) -> np.ndarray:
    if radius < 1:
        return np.ones((1, 1), bool)
    r = int(radius)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def analyze(rgb: np.ndarray, roof: np.ndarray, gsd: float, p: Params) -> dict:
    """지붕 크롭 하나에 대해 설치 가능 면적을 계산한다."""
    px_m2 = gsd * gsd
    gray = rgb.astype(np.float32).mean(axis=2)

    # 1) 가장자리 이격 버퍼
    er = max(int(round(p.edge_buffer_m / gsd)), 1)
    inner = ndimage.binary_erosion(roof, structure=disk(er))

    # 2) 배제: 그림자 + 불균질(구조물/엣지)
    shadow = gray < p.shadow_gray_max
    rough = local_std(gray, FLAT_WIN) > p.flat_std_max
    usable = inner & ~shadow & ~rough

    # 잔털 제거: 열림 연산으로 1~2px 노이즈 정리
    usable = ndimage.binary_opening(usable, structure=disk(1))

    # 3) 연결성분 -> 최소 면적 필터
    lab, n = ndimage.label(usable)
    keep = np.zeros_like(usable)
    comps: list[float] = []
    if n:
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for i, sz in enumerate(sizes, start=1):
            a = float(sz) * px_m2
            if a >= p.min_component_m2:
                keep |= (lab == i)
                comps.append(a)
    comps.sort(reverse=True)

    # 참고용 패널 힌트 (최종 수치 아님. 1번 파트 산출물로 대체될 자리)
    r, g_, b = rgb[..., 0].astype(np.int16), rgb[..., 1].astype(np.int16), rgb[..., 2].astype(np.int16)
    panel_hint = roof & (gray < PANEL_GRAY_MAX) & ((b - r) > PANEL_BLUE_MIN)

    avail = float(keep.sum()) * px_m2
    return {
        "roof_area_m2": float(roof.sum()) * px_m2,
        "inner_area_m2": float(inner.sum()) * px_m2,
        "shadow_area_m2": float((roof & shadow).sum()) * px_m2,
        "rough_area_m2": float((roof & rough).sum()) * px_m2,
        "available_area_m2": avail,
        "install_area_m2": avail * p.packing_eff,
        "largest_component_m2": comps[0] if comps else 0.0,
        "n_components": len(comps),
        "panel_hint_m2": float(panel_hint.sum()) * px_m2,
        "_masks": {"inner": inner, "usable": keep, "shadow": roof & shadow,
                   "rough": roof & rough, "panel": panel_hint},
    }


def grade(rec: dict) -> str:
    """등급. 총면적보다 '가장 큰 한 덩어리'가 사업성을 좌우한다."""
    roof = rec["roof_area_m2"]
    if roof <= 0:
        return "X"
    ratio = rec["available_area_m2"] / roof
    largest = rec["largest_component_m2"]
    if rec["panel_hint_m2"] / roof > 0.40:
        return "X"          # 이미 상당 부분 활용 중으로 의심
    if largest >= 100 and ratio >= 0.35:
        return "A"
    if largest >= 30 and ratio >= 0.15:
        return "B"
    return "C"


# --- 검증용 오버레이 ----------------------------------------------------

OVERLAY_COLORS = {
    "usable": (60, 220, 90),
    "shadow": (40, 90, 220),
    "rough": (240, 160, 40),
    "panel": (230, 40, 210),
}


def make_overlay(rgb: np.ndarray, masks: dict, roof: np.ndarray) -> Image.Image:
    """왼쪽 원본 / 오른쪽 오버레이. 사람이 눈으로 검증하기 위한 것."""
    base = rgb.copy()
    over = rgb.astype(np.float32).copy()
    for key in ("rough", "shadow", "panel", "usable"):
        m = masks[key]
        col = np.array(OVERLAY_COLORS[key], dtype=np.float32)
        over[m] = over[m] * 0.45 + col * 0.55
    over = over.astype(np.uint8)

    # 지붕 외곽선
    edge = roof ^ ndimage.binary_erosion(roof, structure=disk(1))
    over[edge] = (255, 255, 255)
    base[edge] = (255, 255, 255)

    h, w = rgb.shape[:2]
    canvas = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
    canvas.paste(Image.fromarray(base), (0, 0))
    canvas.paste(Image.fromarray(over), (w + 8, 0))
    return canvas


# --- 실행 ---------------------------------------------------------------


def shoelace_m2(ring: list[list[float]], lat0: float) -> float:
    mx = 111320.0 * math.cos(math.radians(lat0))
    my = 110574.0
    s = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        s += (x1 * mx) * (y2 * my) - (x2 * mx) * (y1 * my)
    return abs(s) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=config.DEFAULT_REGION)
    ap.add_argument("--min-roof", type=float, default=500.0, help="대상 최소 건물 면적 m2")
    ap.add_argument("--limit", type=int, default=0, help="상위 N동만 (0=전체)")
    ap.add_argument("--overlay", action="store_true", help="검증 이미지 저장")
    ap.add_argument("--pad", type=int, default=24, help="크롭 여유 px")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    region = config.get_region(args.region)
    z, tp = config.ZOOM, config.TILE_PX
    lat_mid = (region.n + region.s) / 2
    gsd = T.ground_resolution_m_per_px(lat_mid, z, tp)
    p = Params()

    feats = load_buildings(region.buildings)
    cand = []
    for ft in feats:
        rings = polygon_rings(ft["geometry"])
        a = sum(shoelace_m2(r, lat_mid) for r in rings)
        if a >= args.min_roof:
            cand.append((a, ft, rings))
    cand.sort(key=lambda t: -t[0])
    if args.limit:
        cand = cand[:args.limit]

    print(f"gsd {gsd:.4f} m/px | 대상 {len(cand)}동 (>= {args.min_roof:.0f} m2)")
    print(f"버퍼 {p.edge_buffer_m}m / 최소성분 {p.min_component_m2}m2 / "
          f"std<{p.flat_std_max} / packing {p.packing_eff}")

    mosaic = TileMosaic(region.raw_tiles, z, tp)
    out_dir = Path(args.out) if args.out else region.dir / "available"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / "overlay"
    if args.overlay:
        ov_dir.mkdir(exist_ok=True)

    rows = []
    for i, (approx_a, ft, rings) in enumerate(cand):
        pid = ft["id"].split(".")[-1]
        pxs, pys = [], []
        for ring in rings:
            for lon, lat in ring:
                x, y = lonlat_to_global_px(lon, lat, z, tp)
                pxs.append(x)
                pys.append(y)
        px0, px1 = int(min(pxs)) - args.pad, int(max(pxs)) + args.pad + 1
        py0, py1 = int(min(pys)) - args.pad, int(max(pys)) + args.pad + 1
        rgb = mosaic.crop(px0, py0, px1, py1)

        mask_img = Image.new("L", (px1 - px0, py1 - py0), 0)
        drw = ImageDraw.Draw(mask_img)
        for ring in rings:
            pts = []
            for lon, lat in ring:
                x, y = lonlat_to_global_px(lon, lat, z, tp)
                pts.append((x - px0, y - py0))
            drw.polygon(pts, fill=255)
        roof = np.asarray(mask_img) > 0
        if roof.sum() < 50:
            continue

        rec = analyze(rgb, roof, gsd, p)
        masks = rec.pop("_masks")
        rec["grade"] = grade(rec)
        props = ft["properties"]
        rows.append({
            "building_id": pid,
            "buld_nm": props.get("buld_nm", ""),
            "rd_nm": props.get("rd_nm", ""),
            "gro_flo_co": props.get("gro_flo_co", ""),
            **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in rec.items()},
        })

        if args.overlay:
            make_overlay(rgb, masks, roof).save(
                ov_dir / f"{i:03d}_{pid}_{rec['grade']}.png")

    csv_path = out_dir / "available.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    print(f"\n{len(rows)}동 처리 -> {csv_path}")
    if args.overlay:
        print(f"검증 이미지 -> {ov_dir}")
    if rows:
        import collections
        gc = collections.Counter(r["grade"] for r in rows)
        print("등급:", dict(sorted(gc.items())))
        tot_roof = sum(r["roof_area_m2"] for r in rows)
        tot_inst = sum(r["install_area_m2"] for r in rows)
        print(f"옥상 합계 {tot_roof:,.0f} m2 -> 설치가능 추정 {tot_inst:,.0f} m2 "
              f"({tot_inst / tot_roof * 100:.1f}%)")


if __name__ == "__main__":
    main()
