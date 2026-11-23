"""Merge NLR panel masks across tile boundaries and export one WGS84 GeoJSON.

Detections are produced per 256px z19 tile, so a panel array straddling a tile
boundary arrives as several clipped instances (1112 of 2257 touched an edge in the
original nlr_conf_0p25 run) with no de-duplication anywhere in the pipeline. This script stitches the per-tile union
masks into one mosaic raster, labels 8-connected components across the seams, and
polygonises each component, so one physical array becomes one feature.

Memory: the mosaic is 15872x14080 for the bitgaram run, so the bool raster plus the
int32 label array peak at roughly 1.1 GB and the whole run takes about 7 seconds.
That is fine for a one-off local export. For a much larger area, switch to per-tile
labelling plus union-find across the seams instead of one global raster.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "solar_panel" / "final_neck_conf_0p20"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "3_panels" / "panels_final.geojson"

TILE_PX = 256
EARTH_CIRCUM = 40075016.685578488  # equatorial circumference (m)


def point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    if start == end:
        return math.dist(point, start)
    dx, dy = end[0] - start[0], end[1] - start[1]
    return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / math.hypot(dx, dy)


def simplify_line(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Small dependency-free Ramer-Douglas-Peucker simplifier."""
    if len(points) <= 2 or epsilon <= 0:
        return points
    distances = [point_line_distance(point, points[0], points[-1]) for point in points[1:-1]]
    if not distances or max(distances) <= epsilon:
        return [points[0], points[-1]]
    pivot = distances.index(max(distances)) + 1
    return simplify_line(points[: pivot + 1], epsilon)[:-1] + simplify_line(points[pivot:], epsilon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.6,
        help="Douglas-Peucker tolerance in mosaic pixels (0.6px is about 0.15m)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=2.0,
        help="drop components smaller than this many square metres",
    )
    parser.add_argument("--precision", type=int, default=7, help="decimal places kept on lon/lat")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one of the result CSVs. They carry a BOM and JSON inside a field."""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def pixel_to_lonlat(fx: float, fy: float, zoom: int) -> tuple[float, float]:
    """Mosaic pixel -> WGS84, the standard slippy-map formula (cf. src/where.py)."""

    scale = float(TILE_PX * 2**zoom)
    longitude = fx / scale * 360.0 - 180.0
    latitude = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * fy / scale))))
    return longitude, latitude


def ground_m_per_px(latitude: float, zoom: int) -> float:
    return EARTH_CIRCUM * math.cos(math.radians(latitude)) / (TILE_PX * 2**zoom)


def stitch_masks(root: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, int, int]:
    """OR every tile's union mask into one mosaic-sized boolean raster."""

    x0 = min(int(row["tile_x"]) for row in rows)
    y0 = min(int(row["tile_y"]) for row in rows)
    x1 = max(int(row["tile_x"]) for row in rows)
    y1 = max(int(row["tile_y"]) for row in rows)
    mosaic = np.zeros(((y1 - y0 + 1) * TILE_PX, (x1 - x0 + 1) * TILE_PX), dtype=bool)
    for row in rows:
        tile = np.asarray(Image.open(root / row["mask_path"]).convert("L")) > 127
        top = (int(row["tile_y"]) - y0) * TILE_PX
        left = (int(row["tile_x"]) - x0) * TILE_PX
        mosaic[top : top + TILE_PX, left : left + TILE_PX] |= tile
    return mosaic, x0, y0


def signed_area(ring: list[tuple[float, float]]) -> float:
    total = 0.0
    for (ax, ay), (bx, by) in zip(ring, ring[1:]):
        total += ax * by - bx * ay
    return total / 2.0


def close_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return points if points[0] == points[-1] else points + [points[0]]


def component_rings(window: np.ndarray, epsilon: float) -> list[list[np.ndarray]]:
    """Polygonise one component's sub-window into [outer, *holes] rings, in window px.

    A 1px zero pad is added so contours of foreground touching the window edge close
    properly; the pad offset is removed again before returning.
    """

    padded = np.zeros((window.shape[0] + 2, window.shape[1] + 2), dtype=float)
    padded[1:-1, 1:-1] = window
    generator = contourpy.contour_generator(z=padded, fill_type=contourpy.FillType.OuterCode)
    polygons, codes = generator.filled(0.5, 2.0)

    shapes: list[list[np.ndarray]] = []
    for points, code in zip(polygons, codes):
        starts = list(np.flatnonzero(code == 1)) + [len(points)]
        rings = []
        for begin, end in zip(starts, starts[1:]):
            ring = simplify_line([(float(x), float(y)) for x, y in points[begin:end]], epsilon)
            if len(ring) >= 4:
                rings.append(np.asarray(ring, dtype=float) - 1.0)
        if rings:
            shapes.append(rings)
    return shapes


def attribute_components(
    root: Path, instances: list[dict[str, str]], labels: np.ndarray, x0: int, y0: int
) -> tuple[dict[int, dict[str, object]], int]:
    """Map every instance onto the components its mask covers.

    An instance mask can consist of several disjoint blobs, so it may land on more
    than one component. Crediting only the largest overlap leaves 193 of 1711
    components with no attributes at all, so every overlapped component is credited
    with the pixels it actually received.
    """

    pixels_by_label: dict[int, Counter] = defaultdict(Counter)
    tiles: dict[int, set[str]] = defaultdict(set)
    instance_count: dict[int, int] = defaultdict(int)
    confidence_max: dict[int, float] = {}
    confidence_sum: dict[int, float] = defaultdict(float)
    orphans = 0

    for row in instances:
        mask = np.asarray(Image.open(root / row["mask_path"]).convert("L")) > 127
        top = (int(row["tile_y"]) - y0) * TILE_PX
        left = (int(row["tile_x"]) - x0) * TILE_PX
        covered = labels[top : top + TILE_PX, left : left + TILE_PX][mask]
        covered = covered[covered > 0]
        if covered.size == 0:
            orphans += 1
            continue
        confidence = float(row["confidence"])
        for component, count in Counter(covered.tolist()).items():
            pixels_by_label[component][row["label"]] += count
            tiles[component].add(row["tile_id"])
            instance_count[component] += 1
            confidence_sum[component] += confidence * count
            confidence_max[component] = max(confidence_max.get(component, 0.0), confidence)

    attributes: dict[int, dict[str, object]] = {}
    for component, counter in pixels_by_label.items():
        total = sum(counter.values())
        attributes[component] = {
            "label": counter.most_common(1)[0][0],
            "label_mix": {name: round(count / total, 3) for name, count in counter.most_common()},
            "conf_max": round(confidence_max[component], 4),
            "conf_mean": round(confidence_sum[component] / total, 4),
            "instances": instance_count[component],
            "tiles": sorted(tiles[component]),
        }
    return attributes, orphans


def main() -> None:
    args = parse_args()
    root = args.input
    if not (root / "summary.csv").is_file():
        raise SystemExit(f"Missing inference summary in {root}. Supply --input pointing to a completed inference run.")

    summary = [row for row in read_csv(root / "summary.csv") if int(row["instances"]) > 0]
    if not summary:
        raise SystemExit(f"no tile in {root}/summary.csv has any instance")
    zooms = {int(row["zoom"]) for row in summary}
    if len(zooms) != 1:
        raise SystemExit(f"expected a single zoom level, got {sorted(zooms)}")
    zoom = zooms.pop()

    mosaic, x0, y0 = stitch_masks(root, summary)
    union_pixels = int(mosaic.sum())
    labels, count = ndimage.label(mosaic, structure=np.ones((3, 3), dtype=bool))
    windows = ndimage.find_objects(labels)
    sizes = ndimage.sum(mosaic, labels, np.arange(1, count + 1))

    instances = read_csv(root / "instances.csv")
    attributes, orphans = attribute_components(root, instances, labels, x0, y0)

    features: list[dict[str, object]] = []
    dropped_small = 0
    for index, window in enumerate(windows, start=1):
        rows, cols = window
        # Component centre in mosaic pixels, used for the local ground resolution.
        centre_lat = pixel_to_lonlat(
            (x0 * TILE_PX) + (cols.start + cols.stop) / 2.0,
            (y0 * TILE_PX) + (rows.start + rows.stop) / 2.0,
            zoom,
        )[1]
        metres_per_pixel = ground_m_per_px(centre_lat, zoom)
        area = float(sizes[index - 1]) * metres_per_pixel**2
        if area < args.min_area:
            dropped_small += 1
            continue

        shapes = component_rings(labels[window] == index, args.simplify)
        if not shapes:
            dropped_small += 1
            continue

        polygons = []
        for rings in shapes:
            coordinates = []
            for position, ring in enumerate(rings):
                points = [
                    (
                        round(lon, args.precision),
                        round(lat, args.precision),
                    )
                    for lon, lat in (
                        pixel_to_lonlat(
                            (x0 * TILE_PX) + cols.start + px,
                            (y0 * TILE_PX) + rows.start + py,
                            zoom,
                        )
                        for px, py in ring
                    )
                ]
                points = close_ring(points)
                # RFC 7946: exterior ring counter-clockwise, holes clockwise. The
                # contour comes out in image space (y down), so orientation flips
                # once y becomes latitude -- settle it on the final coordinates.
                outer_wanted = position == 0
                if (signed_area(points) > 0) != outer_wanted:
                    points.reverse()
                coordinates.append([list(point) for point in points])
            polygons.append(coordinates)

        properties: dict[str, object] = {
            "id": "",
            "area_m2": round(area, 2),
            "pixels": int(sizes[index - 1]),
        }
        properties.update(
            attributes.get(
                index,
                {
                    "label": "unknown",
                    "label_mix": {},
                    "conf_max": 0.0,
                    "conf_mean": 0.0,
                    "instances": 0,
                    "tiles": [],
                },
            )
        )
        geometry = (
            {"type": "Polygon", "coordinates": polygons[0]}
            if len(polygons) == 1
            else {"type": "MultiPolygon", "coordinates": polygons}
        )
        features.append({"type": "Feature", "properties": properties, "geometry": geometry})

    features.sort(key=lambda feature: -float(feature["properties"]["area_m2"]))
    for number, feature in enumerate(features, start=1):
        feature["properties"]["id"] = f"PV_{number:04d}"

    total_area = sum(float(feature["properties"]["area_m2"]) for feature in features)
    unlabelled = sum(1 for feature in features if feature["properties"]["instances"] == 0)
    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "source": str(root.name),
            "zoom": zoom,
            "tiles_with_instances": len(summary),
            "source_instances": len(instances),
            "components": count,
            "features": len(features),
            "simplify_px": args.simplify,
            "min_area_m2": args.min_area,
            "total_area_m2": round(total_area, 1),
        },
        "features": features,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collection, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    vertices = sum(
        len(ring)
        for feature in features
        for polygon in (
            [feature["geometry"]["coordinates"]]
            if feature["geometry"]["type"] == "Polygon"
            else feature["geometry"]["coordinates"]
        )
        for ring in polygon
    )
    print(f"output={args.output}")
    print(f"tiles={len(summary)} instances={len(instances)} components={count}")
    print(f"features={len(features)} dropped_small={dropped_small} vertices={vertices}")
    print(f"area_m2={total_area:.1f} union_px={union_pixels}")
    print(f"orphan_instances={orphans} unlabelled_features={unlabelled}")
    print(f"bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
