"""옥상 설치가능 영역 인계 데이터 생성.

팀원 패널 결과(data/solar_panel/nlr_conf_0p25/)와 같은 규약으로 내보낸다.
타일 키 `{x}_{y}`, 줌 19, 타일 중심 경위도, 256x256 1비트 마스크 PNG.
수합하는 쪽에서 타일 키로 패널 결과와 바로 겹칠 수 있고, 건물 단위 표로도 쓸 수 있다.

    data/rooftop_area/<name>/
    ├── summary.csv                 타일 단위 (앞 10개 컬럼은 패널 summary.csv 와 동일)
    ├── buildings.csv               건물 단위 속성 + 면적 + 신뢰도
    ├── buildings.geojson           지도용 (건물 바닥 도면 + 같은 속성)
    ├── masks/roof/{tile}.png       옥상 마스크
    ├── masks/available/{tile}.png  설치가능 마스크
    └── building_masks/{id}/roof.png, available.png   건물 크롭 (bbox 는 buildings.csv)

면적 산정 (건물 하나)
    옥상   = SAM 분할 -> footprint 교집합 -> 시드 연결성분 -> 중정 제외   (roof.py)
    패널   = 팀원 rooftop-* 인스턴스 중 confidence >= PANEL_CONF 인 화소
    가능   = (옥상 - 패널) -> 가장자리 1m 버퍼 -> 그림자/불균질 제외
             -> 10 m2 미만 조각 폐기                                        (available.py)
    설치   = 가능 x 0.70 (패널 이격·통로·소형 장애물 흡수)

사용법
    python src/export.py --limit 20          # 시험
    python src/export.py                     # 전체
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_aerial_imagery"))

import config
import tiles as T
from available import (Params, TileMosaic, analyze, load_buildings,
                       lonlat_to_global_px, polygon_rings)
from roof import (CKPT, PAD_PX, conf_grade, load_sharpness, negative_points,
                  pick_mask, refine_mask, seed_points)

PANEL_DIR = "solar_panel/nlr_conf_0p25"
PANEL_CONF = 0.5                     # 설치가능 산정에 쓰는 패널 신뢰도 컷
PANEL_CONF_LEVELS = (0.3, 0.5, 0.7)  # 수합 쪽에서 다시 자를 수 있게 면적을 같이 낸다
LEAN_RISK_HEIT = 40.0                # 이 높이 이상은 사선 촬영으로 옥상이 밀려 보일 수 있다


def fnum(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


class PanelLayer:
    """팀원 인스턴스 마스크를 전역 픽셀 좌표로 읽어 '화소별 최대 confidence' 로 합친다.

    옥상(rooftop-*)만 쓴다. carport/ground 는 옥상 면적에서 뺄 대상이 아니다.
    """

    def __init__(self, root: Path, tile_px: int):
        self.root = root
        self.tp = tile_px
        self.by_tile: dict[tuple[int, int], list[tuple[float, str]]] = {}
        with open(root / "instances.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not r["label"].startswith("rooftop"):
                    continue
                key = (int(r["tile_x"]), int(r["tile_y"]))
                self.by_tile.setdefault(key, []).append(
                    (float(r["confidence"]), r["mask_path"]))
        self._cache: dict[str, np.ndarray] = {}

    def _mask(self, rel: str) -> np.ndarray:
        m = self._cache.get(rel)
        if m is None:
            m = np.asarray(Image.open(self.root / rel)) > 0
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[rel] = m
        return m

    def conf_raster(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        out = np.zeros((y1 - y0, x1 - x0), np.float32)
        tp = self.tp
        for ty in range(y0 // tp, (y1 - 1) // tp + 1):
            for tx in range(x0 // tp, (x1 - 1) // tp + 1):
                for conf, rel in self.by_tile.get((tx, ty), []):
                    mk = self._mask(rel)
                    gx0, gy0 = tx * tp, ty * tp
                    sx0, sy0 = max(x0, gx0), max(y0, gy0)
                    sx1, sy1 = min(x1, gx0 + tp), min(y1, gy0 + tp)
                    if sx1 <= sx0 or sy1 <= sy0:
                        continue
                    part = mk[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
                    view = out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                    np.maximum(view, np.where(part, conf, 0.0), out=view)
        return out


class TileCanvas:
    """건물 크롭 마스크를 타일 단위 마스크로 모은다 (OR)."""

    def __init__(self, tile_px: int):
        self.tp = tile_px
        self.roof: dict[tuple[int, int], np.ndarray] = {}
        self.avail: dict[tuple[int, int], np.ndarray] = {}
        self.members: dict[tuple[int, int], set[str]] = {}

    def paint(self, bid: str, x0: int, y0: int, roof: np.ndarray, avail: np.ndarray):
        tp = self.tp
        H, W = roof.shape
        x1, y1 = x0 + W, y0 + H
        for ty in range(y0 // tp, (y1 - 1) // tp + 1):
            for tx in range(x0 // tp, (x1 - 1) // tp + 1):
                gx0, gy0 = tx * tp, ty * tp
                sx0, sy0 = max(x0, gx0), max(y0, gy0)
                sx1, sy1 = min(x1, gx0 + tp), min(y1, gy0 + tp)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                rp = roof[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                if not rp.any():
                    continue
                key = (tx, ty)
                R = self.roof.setdefault(key, np.zeros((tp, tp), bool))
                A = self.avail.setdefault(key, np.zeros((tp, tp), bool))
                R[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0] |= rp
                A[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0] |= \
                    avail[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
                self.members.setdefault(key, set()).add(bid)


def save_bits(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(bool)).save(path)   # mode "1", 패널 결과와 동일


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=config.DEFAULT_REGION)
    ap.add_argument("--name", default="sam_vith_v1", help="출력 폴더 이름")
    ap.add_argument("--model", default="vit_h", choices=list(CKPT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    region = config.get_region(args.region)
    z, tp = config.ZOOM, config.TILE_PX
    lat_mid = (region.n + region.s) / 2
    gsd = T.ground_resolution_m_per_px(lat_mid, z, tp)
    px_m2 = gsd * gsd

    out = config.DATA / "rooftop_area" / args.name
    out.mkdir(parents=True, exist_ok=True)

    with open(region.dir / "ledger" / "ledger.csv", encoding="utf-8-sig") as f:
        led = {r["building_id"]: r for r in csv.DictReader(f)}
    feats = {f["id"].split(".")[-1]: f for f in load_buildings(region.buildings)}
    ids = sorted((b for b in led if b in feats), key=lambda b: -fnum(led[b], "poly_m2"))
    if args.limit:
        ids = ids[:args.limit]

    panels = PanelLayer(config.DATA / PANEL_DIR, tp)
    sharp = load_sharpness(region)
    print(f"대상 {len(ids)}동 | 옥상 패널 인스턴스가 있는 타일 {len(panels.by_tile)}개 "
          f"| 패널 컷 conf>={PANEL_CONF}", flush=True)

    import torch
    from segment_anything import SamPredictor, sam_model_registry
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model](checkpoint=CKPT[args.model]).to(dev)
    predictor = SamPredictor(sam)
    mosaic = TileMosaic(region.raw_tiles, z, tp)
    canvas = TileCanvas(tp)
    params = Params()

    rows, geo, t0 = [], [], time.time()
    for i, bid in enumerate(ids):
        r = led[bid]
        rings = polygon_rings(feats[bid]["geometry"])
        pts = [lonlat_to_global_px(lo, la, z, tp) for ring in rings for lo, la in ring]
        x0 = int(min(p[0] for p in pts)) - PAD_PX
        y0 = int(min(p[1] for p in pts)) - PAD_PX
        x1 = int(max(p[0] for p in pts)) + PAD_PX + 1
        y1 = int(max(p[1] for p in pts)) + PAD_PX + 1

        base = {
            "building_id": bid, "bd_mgt_sn": r.get("bd_mgt_sn", ""),
            "bld_nm": r.get("bld_nm", ""), "new_plat_plc": r.get("new_plat_plc", ""),
            "plat_plc": r.get("plat_plc", ""),
            "main_purps": r.get("main_purps", ""), "heit": fnum(r, "heit"),
            "arch_area": fnum(r, "arch_matched") or fnum(r, "arch_max"),
            "verdict": r.get("verdict", ""), "match_kind": r.get("match_kind", ""),
            "footprint_m2": fnum(r, "poly_m2"),
            "px_x0": x0, "px_y0": y0, "px_x1": x1, "px_y1": y1,
        }

        fi = Image.new("L", (x1 - x0, y1 - y0), 0)
        fd = ImageDraw.Draw(fi)
        for ring in rings:
            fd.polygon([(lonlat_to_global_px(lo, la, z, tp)[0] - x0,
                         lonlat_to_global_px(lo, la, z, tp)[1] - y0)
                        for lo, la in ring], fill=255)
        fp = np.asarray(fi) > 0
        if fp.sum() < 100 or (x1 - x0) * (y1 - y0) > 5_000_000:
            rows.append({**base, "status": "skip_size"})
            continue

        rgb = mosaic.crop(x0, y0, x1, y1)
        seeds = seed_points(fp, 5)
        neg = negative_points(fp, 8)
        p = np.vstack([seeds, neg]) if len(neg) else seeds
        lab = np.concatenate([np.ones(len(seeds)), np.zeros(len(neg))])
        ys, xs = np.nonzero(fp)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], float)
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(point_coords=p.astype(float),
                                             point_labels=lab, box=box,
                                             multimask_output=True)
        m, _, _ = pick_mask(masks, scores, rgb.shape[0] * rgb.shape[1], seeds)
        roof, court_m2 = refine_mask(m, fp, rgb, seeds, gsd)

        on_roof = panels.conf_raster(x0, y0, x1, y1) * roof
        panel = on_roof >= PANEL_CONF
        rec = analyze(rgb, roof & ~panel, gsd, params)
        avail = rec["_masks"]["usable"]

        tile_keys = [(tx, ty) for ty in range(y0 // tp, (y1 - 1) // tp + 1)
                     for tx in range(x0 // tp, (x1 - 1) // tp + 1)]
        laps = [sharp[k] for k in tile_keys if k in sharp]
        lap = float(np.mean(laps)) if laps else 0.0

        cy, cx = ndimage.center_of_mass(roof if roof.any() else fp)
        clat, clon = T.tile2deg((x0 + cx) / tp, (y0 + cy) / tp, z)
        covered = [f"{tx}_{ty}" for tx, ty in tile_keys
                   if fp[max(0, ty * tp - y0):max(0, ty * tp - y0 + tp),
                         max(0, tx * tp - x0):max(0, tx * tp - x0 + tp)].any()]

        rdir = f"building_masks/{bid}"
        save_bits(roof, out / rdir / "roof.png")
        save_bits(avail, out / rdir / "available.png")
        canvas.paint(bid, x0, y0, roof, avail)

        row = {
            **base,
            "center_latitude": round(clat, 8), "center_longitude": round(clon, 8),
            "tile_ids": ";".join(covered),
            "roof_m2": round(float(roof.sum()) * px_m2, 1),
            "court_m2": round(court_m2, 1),
            **{f"panel_m2_c{int(round(t * 10)):02d}":
               round(float((on_roof >= t).sum()) * px_m2, 1) for t in PANEL_CONF_LEVELS},
            "panel_conf_max": round(float(on_roof.max()), 3) if on_roof.any() else "",
            "available_m2": round(rec["available_area_m2"], 1),
            "install_m2": round(rec["install_area_m2"], 1),
            "largest_m2": round(rec["largest_component_m2"], 1),
            "n_components": rec["n_components"],
            "image_sharpness": round(lap, 1),
            "image_conf": conf_grade(lap),
            "lean_risk": int(fnum(r, "heit") >= LEAN_RISK_HEIT),
            # 부지형(폴리곤이 건물이 아니라 부지)·대장없음(검증 불가)은 면적을 믿을 수 없다.
            # 폴리곤과소는 한 지번 여러 동 중 작은 동이라 정상으로 본다.
            "valid": int(r.get("verdict", "") in ("정상", "폴리곤과소")),
            "roof_mask_path": f"{rdir}/roof.png",
            "available_mask_path": f"{rdir}/available.png",
            "status": "ok",
        }
        rows.append(row)
        geo.append({"type": "Feature", "geometry": feats[bid]["geometry"],
                    "properties": {k: v for k, v in row.items()
                                   if not k.startswith("px_")}})

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ids)}  ({(time.time() - t0) / (i + 1):.2f}s/동)",
                  flush=True)

    # --- 건물 단위 --------------------------------------------------------
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "buildings.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    (out / "buildings.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": geo}, ensure_ascii=False),
        encoding="utf-8")

    # --- 타일 단위 (전 타일, 패널 summary.csv 와 같은 키/좌표 컬럼) -------
    tiles_all = sorted((int(q.stem.split("_")[0]), int(q.stem.split("_")[1]))
                       for q in region.raw_tiles.glob("*.jpeg"))
    empty = np.zeros((tp, tp), bool)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["key", "source_filename", "tile_id", "tile_x", "tile_y", "zoom",
                     "center_latitude", "center_longitude", "width", "height",
                     "buildings", "building_ids", "roof_pixels", "available_pixels",
                     "roof_fraction", "available_fraction",
                     "roof_mask_path", "available_mask_path"])
        for tx, ty in tiles_all:
            key = f"{tx}_{ty}"
            R = canvas.roof.get((tx, ty), empty)
            A = canvas.avail.get((tx, ty), empty)
            save_bits(R, out / "masks" / "roof" / f"{key}.png")
            save_bits(A, out / "masks" / "available" / f"{key}.png")
            lat, lon = T.tile2deg(tx + 0.5, ty + 0.5, z)
            mem = sorted(canvas.members.get((tx, ty), set()))
            wr.writerow([key, f"{key}.jpeg", key, tx, ty, z,
                         f"{lat:.8f}", f"{lon:.8f}", tp, tp,
                         len(mem), ";".join(mem), int(R.sum()), int(A.sum()),
                         round(float(R.mean()), 4), round(float(A.mean()), 4),
                         f"masks/roof/{key}.png", f"masks/available/{key}.png"])

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(ok)}/{len(rows)}동 완료, {time.time() - t0:.0f}s -> {out}")
    print(f"옥상 {sum(r['roof_m2'] for r in ok):,.0f} m²  "
          f"패널(c05) {sum(r['panel_m2_c05'] for r in ok):,.0f} m²  "
          f"설치가능 {sum(r['available_m2'] for r in ok):,.0f} m²  "
          f"설치추정 {sum(r['install_m2'] for r in ok):,.0f} m²")


if __name__ == "__main__":
    main()
