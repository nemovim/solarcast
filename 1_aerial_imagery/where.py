"""수집한 영상이 실제로 어디인지 확인하는 도구.

칩 ID, 타일 좌표, 모자이크 픽셀, 경위도 사이를 양방향으로 변환하고
지도 서비스 링크를 뽑아 육안 대조할 수 있게 한다.

사용 예
  python src/where.py --summary
  python src/where.py --chip c037_024
  python src/where.py --tile 446788 207627
  python src/where.py --pixel 7296 6016
  python src/where.py --lonlat 126.7853 35.0235
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

_spec = importlib.util.spec_from_file_location("tiles", Path(__file__).parent / "tiles.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def map_links(lon: float, lat: float) -> list[str]:
    """육안 대조용 지도 링크. 항공사진으로 봐야 비교가 된다."""
    return [
        f"VWorld  https://map.vworld.kr/map/maps.do#{lon},{lat},18,base,satellite",
        f"카카오맵 https://map.kakao.com/link/map/{lat},{lon}",
        f"구글맵  https://www.google.com/maps/@{lat},{lon},19z/data=!3m1!1e3",
        f"OSM     https://www.openstreetmap.org/#map=19/{lat}/{lon}",
    ]


def describe(lon: float, lat: float) -> None:
    print(f"  경위도 : {lon:.6f}, {lat:.6f}")
    print(f"  (위도,경도 순) : {lat:.6f}, {lon:.6f}")
    for ln in map_links(lon, lat):
        print(f"  {ln}")


def region_of(lon: float, lat: float):
    for r in config.REGIONS.values():
        if r.w <= lon <= r.e and r.s <= lat <= r.n:
            return r
    return None


def summary() -> None:
    print("=== 수집 지역 ===\n")
    for name, r in config.REGIONS.items():
        tr = T.region_tile_range(r, config.ZOOM)
        px = config.TILE_PX
        res = T.ground_resolution_m_per_px((r.n + r.s) / 2, config.ZOOM)
        clon, clat = (r.w + r.e) / 2, (r.s + r.n) / 2
        print(f"[{name}] role={r.role}")
        print(f"  {r.desc}")
        print(f"  bbox   : 서 {r.w}  남 {r.s}  동 {r.e}  북 {r.n}")
        print(f"  네 모서리 (경도, 위도):")
        print(f"    좌상 {r.w:.5f}, {r.n:.5f}    우상 {r.e:.5f}, {r.n:.5f}")
        print(f"    좌하 {r.w:.5f}, {r.s:.5f}    우하 {r.e:.5f}, {r.s:.5f}")
        print(f"  중심   : {clon:.6f}, {clat:.6f}")
        print(f"  타일   : x {tr.x0}..{tr.x1} ({tr.nx}) / y {tr.y0}..{tr.y1} ({tr.ny}) = {tr.count}장")
        print(f"  모자이크: {tr.nx*px} x {tr.ny*px} px, {res:.4f} m/px, "
              f"{tr.nx*px*res/1000:.2f} x {tr.ny*px*res/1000:.2f} km")
        print(f"  중심 확인용 링크:")
        for ln in map_links(clon, clat):
            print(f"    {ln}")
        print()


def from_chip(chip_id: str, region_name: str | None) -> None:
    names = [region_name] if region_name else list(config.REGIONS)
    for nm in names:
        r = config.get_region(nm)
        if not r.chips_meta.exists():
            continue
        with r.chips_meta.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["chip_id"] == chip_id:
                    lon, lat = float(row["lon"]), float(row["lat"])
                    size = config.CHIP_SIZE
                    res = T.ground_resolution_m_per_px(lat, config.ZOOM)
                    print(f"=== {chip_id} ({r.name}) ===")
                    print(f"  모자이크 픽셀 : col {row['col']}, row {row['row']} (좌상단)")
                    print(f"  칩 크기       : {size}x{size} px = 지상 {size*res:.0f} x {size*res:.0f} m")
                    print(f"  split         : {row['split']}   rank {row['rank']}   priority {row['priority']}")
                    print(f"  선명도 lap_min: {row['lap_min']}")
                    print("  [칩 중심]")
                    describe(lon, lat)
                    img = r.chips_img / f"{chip_id}.png"
                    print(f"  이미지: {img}  ({'있음' if img.exists() else '없음'})")
                    return
    print(f"{chip_id} 를 찾지 못했습니다.")


def from_tile(x: int, y: int) -> None:
    z = config.ZOOM
    lat_nw, lon_nw = T.tile2deg(x, y, z)
    lat_se, lon_se = T.tile2deg(x + 1, y + 1, z)
    print(f"=== 타일 z{z}/{x}/{y} ===")
    print(f"  범위 : 경도 {lon_nw:.6f}..{lon_se:.6f}  위도 {lat_se:.6f}..{lat_nw:.6f}")
    print("  [타일 중심]")
    describe((lon_nw + lon_se) / 2, (lat_nw + lat_se) / 2)
    r = region_of((lon_nw + lon_se) / 2, (lat_nw + lat_se) / 2)
    if r:
        p = r.raw_tiles / f"{x}_{y}.jpeg"
        print(f"  지역 : {r.name}")
        print(f"  파일 : {p}  ({'있음' if p.exists() else '없음'})")
    print(f"  원본 URL : https://api.vworld.kr/req/wmts/1.0.0/<KEY>/Satellite/{z}/{y}/{x}.jpeg")


def from_pixel(col: int, row: int, region_name: str) -> None:
    r = config.get_region(region_name)
    tr = T.region_tile_range(r, config.ZOOM)
    px = config.TILE_PX
    tx = tr.x0 + col // px
    ty = tr.y0 + row // px
    # 픽셀 중심의 경위도
    n = 2.0**config.ZOOM
    fx = (tr.x0 * px + col + 0.5) / px
    fy = (tr.y0 * px + row + 0.5) / px
    lon = fx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * fy / n))))
    print(f"=== {r.name} 모자이크 픽셀 (col {col}, row {row}) ===")
    print(f"  포함 타일 : z{config.ZOOM}/{tx}/{ty}")
    describe(lon, lat)


def from_lonlat(lon: float, lat: float) -> None:
    print(f"=== 경위도 {lon}, {lat} ===")
    r = region_of(lon, lat)
    if r is None:
        print("  수집 범위 밖입니다.")
        for nm, rg in config.REGIONS.items():
            print(f"    {nm}: 경도 {rg.w}..{rg.e}  위도 {rg.s}..{rg.n}")
        describe(lon, lat)
        return

    z = config.ZOOM
    tx, ty = T.deg2tile(lat, lon, z)
    tr = T.region_tile_range(r, z)
    px = config.TILE_PX
    fx, fy = T.deg2tile_float(lat, lon, z)
    col = int(round((fx - tr.x0) * px))
    row = int(round((fy - tr.y0) * px))

    print(f"  지역     : {r.name} ({r.role})")
    print(f"  타일     : z{z}/{tx}/{ty}")
    tp = r.raw_tiles / f"{tx}_{ty}.jpeg"
    print(f"  타일 파일: {tp}  ({'있음' if tp.exists() else '없음'})")
    print(f"  모자이크 : col {col}, row {row}")

    # 이 지점을 포함하는 칩들
    if r.chips_meta.exists():
        size = config.CHIP_SIZE
        hits = []
        with r.chips_meta.open(encoding="utf-8") as fh:
            for m in csv.DictReader(fh):
                c0, r0 = int(m["col"]), int(m["row"])
                if c0 <= col < c0 + size and r0 <= row < r0 + size:
                    hits.append(m)
        if hits:
            print(f"  포함 칩  : {len(hits)}개 (칩이 50% 겹쳐 여러 개가 정상)")
            for m in sorted(hits, key=lambda m: int(m["rank"]))[:4]:
                print(f"    {m['chip_id']}  split={m['split']}  rank={m['rank']}  "
                      f"priority={m['priority']}")
    describe(lon, lat)


def main() -> int:
    ap = argparse.ArgumentParser(description="좌표 확인 도구")
    ap.add_argument("--summary", action="store_true", help="두 지역의 범위 요약")
    ap.add_argument("--chip", help="칩 ID (예: c037_024)")
    ap.add_argument("--tile", nargs=2, type=int, metavar=("X", "Y"))
    ap.add_argument("--pixel", nargs=2, type=int, metavar=("COL", "ROW"))
    ap.add_argument("--lonlat", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--region", default=None, choices=list(config.REGIONS))
    args = ap.parse_args()

    if args.summary:
        summary()
    elif args.chip:
        from_chip(args.chip, args.region)
    elif args.tile:
        from_tile(*args.tile)
    elif args.pixel:
        from_pixel(*args.pixel, args.region or config.DEFAULT_REGION)
    elif args.lonlat:
        from_lonlat(*args.lonlat)
    else:
        ap.print_help()
        print("\n예시:")
        print("  python src/where.py --summary")
        print("  python src/where.py --chip c037_024")
        print("  python src/where.py --lonlat 126.7853 35.0235")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
