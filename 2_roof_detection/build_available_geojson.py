"""옥상 설치가능 영역 마스크(rooftop_area)를 지도용 WGS84 GeoJSON 폴리곤으로 바꾼다.

인계 데이터는 건물마다 크롭한 1비트 마스크(building_masks/{id}/available.png)와
그 크롭의 z19 전역 픽셀 좌상단(buildings.csv 의 px_x0, px_y0)만 준다.
지도에 그리고 그림자와 겹쳐 보려면 폴리곤이 필요하므로, 패널 GeoJSON 을 만드는
solar-panel/src/export_geojson.py 의 윤곽 추출·좌표 변환을 그대로 빌려 쓴다.
10 m² 미만 조각은 인계 단계에서 이미 버렸으므로 여기서는 면적으로 거르지 않는다.

    python shadow-map-satellite/scripts/build_available_geojson.py

출력: data/rooftop_area/web/available.geojson  (인계 폴더 sam_vith_v1 은 건드리지 않는다)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "3_panel_detection"))

from src.export_geojson import (  # noqa: E402
    close_ring,
    component_rings,
    ground_m_per_px,
    pixel_to_lonlat,
    signed_area,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "rooftop_area" / "sam_vith_v1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "2_roof" / "available_roof.geojson"
ZOOM = 19


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--simplify", type=float, default=1.0, help="Douglas-Peucker 허용오차(px), 1px ≈ 0.24m")
    ap.add_argument("--precision", type=int, default=7, help="경위도 소수 자릿수")
    return ap.parse_args()


def to_lonlat(ring: np.ndarray, x0: float, y0: float, precision: int) -> list[list[float]]:
    points = [
        (round(lon, precision), round(lat, precision))
        for lon, lat in (pixel_to_lonlat(x0 + px, y0 + py, ZOOM) for px, py in ring)
    ]
    return [list(p) for p in close_ring(points)]


def main() -> None:
    args = parse_args()
    with (args.input / "buildings.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    features: list[dict[str, object]] = []
    missing = 0
    for row in rows:
        mask_path = args.input / row["available_mask_path"]
        if not row["available_mask_path"] or not mask_path.exists():
            missing += 1
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        if not mask.any():
            continue

        x0, y0 = int(row["px_x0"]), int(row["px_y0"])
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        windows = ndimage.find_objects(labels)
        m_per_px = ground_m_per_px(float(row["center_latitude"]), ZOOM)

        for index, (rs, cs) in enumerate(windows, start=1):
            component = labels[rs, cs] == index
            shapes = component_rings(component, args.simplify)
            if not shapes:
                continue
            polygons = []
            for rings in shapes:
                coordinates = []
                for position, ring in enumerate(rings):
                    points = to_lonlat(ring, x0 + cs.start, y0 + rs.start, args.precision)
                    # RFC 7946: 외곽 반시계, 구멍 시계 — 최종 경위도에서 맞춘다
                    if (signed_area(points) > 0) != (position == 0):
                        points.reverse()
                    coordinates.append(points)
                polygons.append(coordinates)
            geometry = (
                {"type": "Polygon", "coordinates": polygons[0]}
                if len(polygons) == 1
                else {"type": "MultiPolygon", "coordinates": polygons}
            )
            features.append({
                "type": "Feature",
                "properties": {
                    "building_id": row["building_id"],
                    "area_m2": round(float(component.sum()) * m_per_px**2, 1),
                    "valid": int(row["valid"]),
                },
                "geometry": geometry,
            })

    total = sum(f["properties"]["area_m2"] for f in features)
    valid_total = sum(f["properties"]["area_m2"] for f in features if f["properties"]["valid"])
    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "source": args.input.name,
            "buildings": len(rows),
            "features": len(features),
            "area_m2": round(total, 1),
            "valid_area_m2": round(valid_total, 1),
        },
        "features": features,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collection, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"output={args.output}")
    print(f"buildings={len(rows)} missing_masks={missing} features={len(features)}")
    print(f"area_m2={total:.1f} valid_area_m2={valid_total:.1f}")
    print(f"bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
