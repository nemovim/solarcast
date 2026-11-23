"""Web Mercator 타일 <-> 경위도 변환 유틸.

VWorld WMTS는 GoogleMapsCompatible(EPSG:3857) 타일 스킴을 쓰므로
표준 slippy-map 공식이 그대로 적용된다.

주의: VWorld의 REST 경로 순서는 {z}/{y}/{x} 로 행(row)이 먼저다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterator, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Web Mercator 상수
EARTH_CIRCUM = 40075016.685578488  # 적도 둘레 (m)
ORIGIN_SHIFT = EARTH_CIRCUM / 2.0  # 20037508.34


class TileRange(NamedTuple):
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def nx(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def ny(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def count(self) -> int:
        return self.nx * self.ny

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for y in range(self.y0, self.y1 + 1):
            for x in range(self.x0, self.x1 + 1):
                yield x, y


def deg2tile_float(lat: float, lon: float, z: int) -> tuple[float, float]:
    """경위도 -> 실수 타일 좌표."""
    n = 2.0**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    """경위도 -> 정수 타일 인덱스."""
    x, y = deg2tile_float(lat, lon, z)
    return math.floor(x), math.floor(y)


def tile2deg(x: float, y: float, z: int) -> tuple[float, float]:
    """타일 좌표 -> 해당 타일 좌상단의 (위도, 경도)."""
    n = 2.0**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def tile_range_for_bbox(w: float, s: float, e: float, n: float, z: int) -> TileRange:
    """경위도 bbox를 덮는 타일 인덱스 범위(양 끝 포함)."""
    x0, _ = deg2tile(s, w, z)
    x1, _ = deg2tile(s, e, z)
    _, y0 = deg2tile(n, w, z)
    _, y1 = deg2tile(s, w, z)
    return TileRange(x0, x1, y0, y1)


def region_tile_range(region, z: int = config.ZOOM) -> TileRange:
    """지역(config.Region) bbox를 덮는 타일 범위."""
    return tile_range_for_bbox(region.w, region.s, region.e, region.n, z)


# --- Web Mercator 미터 좌표 (GeoTIFF 작성용) -----------------------------


def lonlat_to_meters(lon: float, lat: float) -> tuple[float, float]:
    """EPSG:4326 -> EPSG:3857."""
    mx = lon * ORIGIN_SHIFT / 180.0
    my = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * ORIGIN_SHIFT / 180.0
    return mx, my


def tile_nw_meters(x: int, y: int, z: int) -> tuple[float, float]:
    """타일 좌상단의 Web Mercator 미터 좌표."""
    lat, lon = tile2deg(x, y, z)
    return lonlat_to_meters(lon, lat)


def resolution_m_per_px(z: int, tile_px: int = config.TILE_PX) -> float:
    """해당 줌의 Web Mercator 픽셀 해상도 (적도 기준, m/px)."""
    return EARTH_CIRCUM / (tile_px * 2.0**z)


def ground_resolution_m_per_px(lat: float, z: int, tile_px: int = config.TILE_PX) -> float:
    """위도를 반영한 실제 지상 해상도 (m/px)."""
    return resolution_m_per_px(z, tile_px) * math.cos(math.radians(lat))


if __name__ == "__main__":
    for name, region in config.REGIONS.items():
        tr = region_tile_range(region)
        lat_mid = (region.n + region.s) / 2
        res = ground_resolution_m_per_px(lat_mid, config.ZOOM)
        px = config.TILE_PX
        print(f"=== {name} ({region.role}) ===")
        print(f"  {region.desc}")
        print(f"  bbox        : lon {region.w}..{region.e}  lat {region.s}..{region.n}")
        print(f"  tile x      : {tr.x0}..{tr.x1}  ({tr.nx})")
        print(f"  tile y      : {tr.y0}..{tr.y1}  ({tr.ny})")
        print(f"  total tiles : {tr.count}  (~{tr.count * 13 / 1024:.0f} MB)")
        print(f"  ground res  : {res:.4f} m/px")
        print(f"  mosaic px   : {tr.nx * px} x {tr.ny * px}")
        print(
            f"  extent      : {tr.nx * px * res / 1000:.2f} x "
            f"{tr.ny * px * res / 1000:.2f} km"
        )
