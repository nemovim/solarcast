"""SAM 으로 옥상 경계를 영상에서 직접 뽑는다.

FINDINGS.md 1번 참고: buildings.geojson 은 49.3% 가 단순 사각형이라
옥상 마스크로 쓸 수 없다. footprint 는 **위치 힌트**로만 쓰고,
실제 경계는 영상에서 SAM zero-shot 으로 뽑는다.

프롬프트 전략
    footprint 내부를 침식해 얻은 점들을 positive 포인트로 준다.
    footprint 가 부지 경계인 경우(골프연습장 등) 내부 점이 건물 밖에
    찍히므로, 결과 마스크가 footprint 를 거의 다 덮으면 실패로 본다.

사용법
    python src/roof.py --limit 12 --overlay        # 프로토타입 확인
    python src/roof.py --floors 1-7 --overlay      # 저중층만
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_aerial_imagery"))

import config
from available import (TileMosaic, disk, load_buildings, lonlat_to_global_px,
                       polygon_rings, shoelace_m2)

CKPT = {"vit_h": "models/sam_vit_h_4b8939.pth",
        "vit_b": "models/sam_vit_b_01ec64.pth"}

N_SEEDS = 5          # footprint 내부 positive 포인트 개수
N_NEG = 8            # footprint 바깥 negative 포인트 개수 (도로/식생 번짐 억제)
MAX_COVER = 0.85     # 크롭의 이 비율 이상을 덮는 마스크는 배경으로 보고 버린다
PAD_PX = 40          # 크롭 여유. 건물이 잘리지 않게


def seed_points(roof: np.ndarray, n: int) -> np.ndarray:
    """footprint 내부에서 고르게 떨어진 positive 포인트를 뽑는다.

    거리변환 최대점(가장 안쪽)을 먼저 잡고, 나머지는 침식된 영역에서
    균등 샘플링한다. 가장자리 점은 옆 건물·도로를 물기 쉬워 피한다.
    """
    dist = ndimage.distance_transform_edt(roof)
    pts = [np.unravel_index(int(np.argmax(dist)), dist.shape)[::-1]]

    inner = dist > max(dist.max() * 0.35, 3)
    ys, xs = np.nonzero(inner)
    if len(xs) > 1:
        idx = np.linspace(0, len(xs) - 1, min(n - 1, len(xs))).astype(int)
        pts += [(int(xs[i]), int(ys[i])) for i in idx]
    return np.array(pts[:n], dtype=int)


def negative_points(roof: np.ndarray, n: int) -> np.ndarray:
    """footprint 바깥 고리에서 negative 포인트를 뽑는다.

    건물 밖 도로/식생을 '여기는 아니다'로 알려주면 마스크가 인접
    포장면으로 번지는 것을 막는다. 너무 멀면 정보가 없으므로
    경계에서 적당히 떨어진 고리에서만 뽑는다.
    """
    outside = ndimage.distance_transform_edt(~roof)
    ring = (outside > 6) & (outside < 20)
    ys, xs = np.nonzero(ring)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)
    idx = np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int)
    return np.array([(int(xs[i]), int(ys[i])) for i in idx], dtype=int)


def pick_mask(masks: np.ndarray, scores: np.ndarray, crop_px: int,
              seeds: np.ndarray) -> tuple[np.ndarray, float, int]:
    """SAM 후보 3개 중 옥상다운 것을 고른다."""
    best, best_score, best_i = None, -1.0, -1
    for i, (m, s) in enumerate(zip(masks, scores)):
        cover = m.sum() / crop_px
        if cover > MAX_COVER or m.sum() < 200:
            continue                                  # 배경 전체 / 너무 작음
        if not all(m[y, x] for x, y in seeds[:1]):    # 중심 씨앗을 품어야 함
            continue
        if s > best_score:
            best, best_score, best_i = m, float(s), i
    if best is None:                                  # 전부 탈락하면 최고점 사용
        best_i = int(np.argmax(scores))
        best, best_score = masks[best_i], float(scores[best_i])
    return best, best_score, best_i


MIN_COURT_M2 = 25.0   # 이보다 작은 구멍은 중정이 아니라 노이즈로 본다
COURT_DARK = 78       # 중정 후보: 이보다 어둡거나
COURT_VEG = 8         # G 가 R,B 보다 이만큼 크면 식생


def refine_mask(m: np.ndarray, fp: np.ndarray, rgb: np.ndarray,
                seeds: np.ndarray, gsd: float) -> tuple[np.ndarray, float]:
    """SAM 원마스크를 정제한다. 육안 검수에서 나온 증상들에 대응.

    1) footprint 와 교집합
       부지 유출, 외벽, 건물 밖 짜잘한 조각은 전부 footprint 바깥에 있다.
       footprint 면적은 대장 건축면적과 ±1% 로 맞으므로(측정치) 범위를
       가두는 기준으로 신뢰할 수 있다. SAM 은 그 안에서 모서리만 말한다.
    2) 시드가 든 연결성분만 유지 -> 남은 부스러기 제거
    3) 중정(도넛 가운데)을 구멍으로 제외
       buildings.geojson 에는 내부 링이 0개라 중정 정보가 아예 없다.
       영상에서 찾는 수밖에 없다. 어둡거나 식생인 영역 중 **마스크에 완전히
       둘러싸인 것**만 중정으로 본다. 가장자리에 닿은 어두운 영역은 그냥
       어두운 지붕(방수 도장)이므로 건드리지 않는다.

    반환: (정제 마스크, 중정 면적 m2)
    """
    px_m2 = gsd * gsd
    m = m & fp                                            # 1
    if not m.any():
        return m, 0.0

    lab, n = ndimage.label(m)                             # 2
    keep_ids = {lab[y, x] for x, y in seeds if lab[y, x] > 0}
    if keep_ids:
        m = np.isin(lab, list(keep_ids))
    elif n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == int(sizes.argmax())

    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    gray = rgb.astype(np.float32).mean(axis=2)
    void = m & ((gray < COURT_DARK) |
                ((g - r > COURT_VEG) & (g - b > COURT_VEG)))
    void = ndimage.binary_opening(void, structure=disk(2))

    # 마스크 경계에 닿지 않는 것만 중정으로 인정한다.
    # 성분마다 전체 배열을 비교하면 큰 건물에서 급격히 느려진다(10초/동).
    # 라벨 배열을 한 번만 훑어서 '경계에 닿은 라벨'과 크기를 구한다.
    border = m ^ ndimage.binary_erosion(m, structure=disk(3))
    vlab, vn = ndimage.label(void)
    if vn == 0:
        return m, 0.0
    touching = set(np.unique(vlab[border])) - {0}
    sizes = np.bincount(vlab.ravel(), minlength=vn + 1)
    keep = [i for i in range(1, vn + 1)
            if i not in touching and sizes[i] * px_m2 >= MIN_COURT_M2]
    if not keep:
        return m, 0.0
    court = np.isin(vlab, keep)
    return m & ~court, float(court.sum()) * px_m2


def load_sharpness(region) -> dict[tuple[int, int], float]:
    """qc_report.csv 의 타일별 선명도(lapvar). 신뢰도 등급 산정에 쓴다."""
    path = region.qc_report
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(int(r["x"]), int(r["y"])): float(r["lapvar"])
                for r in csv.DictReader(f)}


def to_f(d: dict, k: str) -> float:
    try:
        return float(d.get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def conf_grade(lap: float) -> str:
    """영상 선명도 등급. 전체 중앙값 130, 하위 10% 는 54 였다."""
    if lap <= 0:
        return "?"
    return "A" if lap >= 130 else "B" if lap >= 80 else "C"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=config.DEFAULT_REGION)
    ap.add_argument("--model", default="vit_h", choices=list(CKPT))
    ap.add_argument("--min-roof", type=float, default=500.0)
    ap.add_argument("--floors", default="", help="예: 1-7 (지상층수 범위)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlay", action="store_true")
    args = ap.parse_args()

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    region = config.get_region(args.region)
    z, tp = config.ZOOM, config.TILE_PX
    lat_mid = (region.n + region.s) / 2
    import tiles as T
    gsd = T.ground_resolution_m_per_px(lat_mid, z, tp)
    px_m2 = gsd * gsd

    flo_lo, flo_hi = 0, 10_000
    if args.floors:
        flo_lo, flo_hi = (int(v) for v in args.floors.split("-"))

    feats = load_buildings(region.buildings)
    cand = []
    for ft in feats:
        rings = polygon_rings(ft["geometry"])
        a = sum(shoelace_m2(r, lat_mid) for r in rings)
        if a < args.min_roof:
            continue
        try:
            fl = int(ft["properties"].get("gro_flo_co") or 0)
        except ValueError:
            fl = 0
        if not (flo_lo <= fl <= flo_hi):
            continue
        cand.append((a, fl, ft, rings))
    cand.sort(key=lambda t: -t[0])
    if args.limit:
        cand = cand[:args.limit]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{args.model} on {dev} | 대상 {len(cand)}동 "
          f"(>={args.min_roof:.0f}m2, {flo_lo}-{flo_hi}층)")
    sam = sam_model_registry[args.model](checkpoint=CKPT[args.model]).to(dev)
    predictor = SamPredictor(sam)

    mosaic = TileMosaic(region.raw_tiles, z, tp)
    out_dir = region.dir / "roof"
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / "overlay"
    if args.overlay:
        ov_dir.mkdir(exist_ok=True)

    # 건축물대장의 건축면적(archArea)은 측량된 법정 수치다. 우리가 만든 값이
    # 아니라 외부에서 온 독립 기준이라, 라벨 없이 면적 정확도를 잴 수 있다.
    # 한계: 면적이 맞아도 모양이 틀릴 수 있고(필요조건이지 충분조건이 아니다),
    #      건축면적은 1층 수평투영이라 저층부가 넓은 타워형은 옥상보다 크다.
    sharp = load_sharpness(region)
    if sharp:
        print(f"타일 선명도 {len(sharp)}장 로드 (신뢰도 등급)")

    ledger: dict[str, dict] = {}
    led_path = region.dir / "ledger" / "ledger.csv"
    if led_path.exists():
        with open(led_path, encoding="utf-8-sig") as f:
            ledger = {r["building_id"]: r for r in csv.DictReader(f)}
        print(f"대장 {len(ledger)}동 조인 (면적 정확도 측정)")

    rows, t0 = [], time.time()
    for i, (fp_m2, fl, ft, rings) in enumerate(cand):
        pid = ft["id"].split(".")[-1]
        pxs, pys = [], []
        for ring in rings:
            for lon, lat in ring:
                gx, gy = lonlat_to_global_px(lon, lat, z, tp)
                pxs.append(gx)
                pys.append(gy)
        px0, px1 = int(min(pxs)) - PAD_PX, int(max(pxs)) + PAD_PX + 1
        py0, py1 = int(min(pys)) - PAD_PX, int(max(pys)) + PAD_PX + 1
        rgb = mosaic.crop(px0, py0, px1, py1)

        mask_img = Image.new("L", (px1 - px0, py1 - py0), 0)
        drw = ImageDraw.Draw(mask_img)
        for ring in rings:
            pts = [(lonlat_to_global_px(lon, lat, z, tp)[0] - px0,
                    lonlat_to_global_px(lon, lat, z, tp)[1] - py0)
                   for lon, lat in ring]
            drw.polygon(pts, fill=255)
        fp = np.asarray(mask_img) > 0
        if fp.sum() < 100:
            continue

        seeds = seed_points(fp, N_SEEDS)
        neg = negative_points(fp, N_NEG)
        pts = np.vstack([seeds, neg]) if len(neg) else seeds
        labels = np.concatenate([np.ones(len(seeds)), np.zeros(len(neg))])

        # 박스로 마스크를 footprint 주변에 가둔다. footprint 가 거칠어도
        # bbox 는 건물을 포함하므로, 도로로 번지는 것을 막는 데 유효하다.
        ys, xs = np.nonzero(fp)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)

        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=pts.astype(float),
            point_labels=labels,
            box=box,
            multimask_output=True)
        m, score, which = pick_mask(masks, scores, rgb.shape[0] * rgb.shape[1], seeds)
        raw_m2 = float(m.sum()) * px_m2
        m, court_m2 = refine_mask(m, fp, rgb, seeds, gsd)

        # 이 건물을 덮는 타일들의 선명도 -> 신뢰도 등급
        laps = [sharp[(tx, ty)]
                for ty in range(py0 // tp, py1 // tp + 1)
                for tx in range(px0 // tp, px1 // tp + 1)
                if (tx, ty) in sharp]
        lap = float(np.mean(laps)) if laps else 0.0

        inter = float((m & fp).sum())
        union = float((m | fp).sum())
        led = ledger.get(pid, {})
        try:
            arch = float(led.get("arch_max") or 0)
        except ValueError:
            arch = 0.0
        sam_m2 = float(m.sum()) * px_m2
        rows.append({"building_id": pid, "gro_flo_co": fl,
                     "footprint_m2": round(fp_m2),
                     "sam_m2": round(sam_m2, 1),
                     "ratio": round(m.sum() / fp.sum(), 3),
                     "iou": round(inter / union, 3) if union else 0.0,
                     "score": round(score, 3), "mask_idx": which,
                     "raw_sam_m2": round(raw_m2, 1),
                     "court_m2": round(court_m2, 1),
                     "sharp": round(lap, 1),
                     "conf": conf_grade(lap),
                     "arch_max": round(arch, 1),
                     "sam_vs_arch": round(sam_m2 / arch, 3) if arch else 0.0,
                     "fp_vs_arch": round(fp_m2 / arch, 3) if arch else 0.0,
                     "jibun_n": led.get("jibun_n", ""),
                     "verdict": led.get("verdict", "")})

        if args.overlay:
            over = rgb.astype(np.float32).copy()
            over[m] = over[m] * 0.5 + np.array([60, 220, 90], np.float32) * 0.5
            edge = fp ^ ndimage.binary_erosion(fp, structure=disk(1))
            over[edge] = (255, 255, 255)
            out = np.concatenate([rgb, np.full((rgb.shape[0], 6, 3), 20, np.uint8),
                                  over.astype(np.uint8)], axis=1)
            im = Image.fromarray(out)
            d = ImageDraw.Draw(im)
            for x, y in seeds:
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 60, 60))
            im.save(ov_dir / f"{i:03d}_{pid}_{fl}f.png")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cand)}  ({(time.time() - t0) / (i + 1):.2f}s/동)")

    with open(out_dir / "roof.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    ious = [r["iou"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    print(f"\n{len(rows)}동, {time.time() - t0:.0f}s")
    print(f"footprint 대비 IoU: 중앙값 {np.median(ious):.3f}  "
          f"평균 {np.mean(ious):.3f}")
    print(f"면적비(SAM/footprint): 중앙값 {np.median(ratios):.3f}")
    print(f"-> {out_dir / 'roof.csv'}")
    if args.overlay:
        print(f"검증 이미지 -> {ov_dir}")
    print("\n주의: footprint 자체가 부정확하므로 IoU 는 정확도가 아니라"
          " '얼마나 다른가'의 지표다. 판정은 오버레이를 눈으로 본다.")

    # --- 면적 정확도 (대장 건축면적 기준) ---------------------------------
    # 지번에 건물이 하나뿐이고 '정상' 판정인 건만 본다. 여러 동이 선 지번은
    # 어느 동이 우리 폴리곤인지 특정할 수 없어 비교가 성립하지 않는다.
    ev = [r for r in rows if r["arch_max"] > 0
          and str(r["jibun_n"]) == "1" and r["verdict"] == "정상"]
    if not ev:
        return
    print(f"\n=== 면적 정확도 (단일건물 지번 {len(ev)}동, 기준: 대장 건축면적) ===")
    for name, key in (("SAM 옥상", "sam_vs_arch"), ("footprint", "fp_vs_arch")):
        v = np.array([r[key] for r in ev])
        within = 100.0 * np.mean(np.abs(v - 1.0) <= 0.20)
        mape = 100.0 * np.mean(np.abs(v - 1.0))
        print(f"  {name:10s} 중앙값 {np.median(v):.3f}  "
              f"±20% 이내 {within:5.1f}%  평균오차 {mape:5.1f}%")
    print("  (면적이 맞아도 모양이 맞다는 뜻은 아니다. 경계 정확도는 손라벨 필요)")

    # 정제 효과: 높이 구간별로 봐야 한다. 저층일수록 부지로 번졌었다.
    print("\n=== 높이 구간별 SAM 면적비 (정제 후) ===")
    for lo, hi in ((0, 15), (15, 25), (25, 40), (40, 300)):
        s = [r for r in ev if lo <= to_f(ledger.get(r["building_id"], {}), "heit") < hi]
        if len(s) >= 5:
            print(f"  {lo:3d}-{hi:3d}m  n={len(s):3d}   "
                  f"SAM {np.median([r['sam_vs_arch'] for r in s]):.3f}   "
                  f"footprint {np.median([r['fp_vs_arch'] for r in s]):.3f}")

    import collections
    cc = collections.Counter(r["conf"] for r in rows)
    court = [r for r in rows if r["court_m2"] > 0]
    print(f"\n신뢰도(영상 선명도): {dict(sorted(cc.items()))}")
    print(f"중정 검출: {len(court)}동, 합계 {sum(r['court_m2'] for r in court):,.0f} m²")


if __name__ == "__main__":
    main()
