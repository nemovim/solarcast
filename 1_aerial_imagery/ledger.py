"""건축물대장 표제부를 조인해 '이 폴리곤이 진짜 건물인가'를 판정한다.

FINDINGS.md 1번 문제 해결용. buildings.geojson 폴리곤의 49%가 단순 사각형이고
일부는 건물이 아니라 부지 전체다(골프연습장 12,520 m2). SAM 도 이건 못 고친다.
가리키는 곳을 충실히 자를 뿐이라 부지를 가리키면 부지를 자른다.

대장의 **건축면적(archArea)** 과 폴리곤 면적을 비교하면 가려진다.

    폴리곤 12,520 m2  vs  건축면적 1,800 m2  -> 부지형, 제외
    폴리곤  7,003 m2  vs  건축면적 6,800 m2  -> 정상

덤으로 주용도(공공건물 필터), 높이(3번 파트 음영 계산), 지붕구조(평/경사)도 온다.

bd_mgt_sn(25자리) 구조
    4617013400 1 0908 0000 000002
    시군구(5) 법정동(5) 대지구분(1) 본번(4) 부번(4) 일련번호(6)

API 는 지번 단위라 같은 지번의 건물들이 한 번에 온다. 지번으로 캐시하면
호출 수가 크게 줄어든다(개발계정 하루 10,000건 한도).

사용법
    python src/ledger.py --limit 20      # 소량 시험
    python src/ledger.py                 # 전체
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "2_roof_detection"))

import config
from available import load_buildings, polygon_rings, shoelace_m2

ENDPOINT = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"

# 폴리곤/건축면적 비율 판정 기준.
# 건축면적은 1층 바닥 투영이고 폴리곤은 대충 그린 외곽이라 정확히 안 맞는다.
# 크게 어긋날 때만 걸러내는 것이 목적이다.
RATIO_MIN = 0.45     # 폴리곤이 건축면적보다 훨씬 작으면 이상
RATIO_MAX = 2.20     # 폴리곤이 건축면적의 2배를 넘으면 부지형으로 의심


def load_key() -> str:
    """.env 에서 인증키를 읽는다. 키는 절대 커밋되지 않는다(.gitignore).

    포털이 인코딩/디코딩 키를 구분해 보여주지 않는 경우가 있어 둘 다 받는다.
    인코딩 키는 '%2B', '%3D%3D' 처럼 % 이스케이프가 들어 있는데,
    requests 가 params= 를 다시 인코딩하므로 그대로 쓰면 이중 인코딩이 되어
    인증에 실패한다. % 가 보이면 한 번 풀어서 디코딩 형태로 맞춘다.
    (키는 base64 라 '+/=' 만 나오고 '%' 가 원래 들어갈 일이 없다.)
    """
    env = Path(config.ROOT) / ".env"
    if not env.exists():
        raise SystemExit(
            ".env 가 없다. 프로젝트 루트에 아래 한 줄을 넣어라:\n"
            "  DATA_GO_KR_KEY=발급받은_인증키")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("DATA_GO_KR_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if not v or v.startswith("여기에"):
                raise SystemExit(".env 의 DATA_GO_KR_KEY 가 아직 플레이스홀더다.")
            if "%" in v:
                v = unquote(v)
            return v
    raise SystemExit(".env 에 DATA_GO_KR_KEY 가 없다.")


# 2026-07-01 전남광주통합특별시 출범으로 시도코드가 바뀌었다.
#   전라남도 46 / 광주광역시 29  ->  전남광주통합특별시 12
# 뒤 3자리는 유지된다(나주시 46170 -> 12170).
# buildings.geojson 의 bd_mgt_sn 은 구 코드라 변환해야 대장이 조회된다.
# 구 코드로 조회하면 오류 없이 0건이 와서 원인을 찾기 어렵다.
SIDO_MIGRATION = {"46": "12", "29": "12"}


def migrate_sigungu(code: str) -> str:
    return SIDO_MIGRATION.get(code[:2], code[:2]) + code[2:]


def parse_mgt_sn(sn: str) -> tuple[str, str, str, str, str] | None:
    """bd_mgt_sn 25자리를 (시군구, 법정동, 대지구분, 본번, 부번)으로 쪼갠다.

    시군구는 통합특별시 신코드로 변환해서 돌려준다.
    """
    if not sn or len(sn) < 19 or not sn[:19].isdigit():
        return None
    # 11번째 자리는 산여부가 아니다. 빛가람 건물이 전부 '1' 인데 신도시에
    # 산 지번이 있을 리 없고, platGbCd='0'(대지) 으로 조회해야 실제로 나온다.
    return (migrate_sigungu(sn[0:5]), sn[5:10], "0",
            sn[11:15], sn[15:19])


def fetch_jibun(key: str, sigungu: str, bjdong: str, plat: str,
                bun: str, ji: str, cache_dir: Path,
                timeout: int = 20) -> list[dict]:
    """지번 하나의 표제부 레코드들을 받아온다. 응답은 파일로 캐시한다."""
    cache = cache_dir / f"{sigungu}_{bjdong}_{plat}_{bun}_{ji}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "serviceKey": key,          # params= 로 넘기므로 '디코딩' 키여야 한다
        "sigunguCd": sigungu,
        "bjdongCd": bjdong,
        "platGbCd": plat,
        "bun": bun,
        "ji": ji,
        "numOfRows": "100",
        "pageNo": "1",
        "_type": "json",
    }
    r = requests.get(ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    try:
        body = r.json()["response"]["body"]
    except (ValueError, KeyError):
        # 인증 실패 등은 XML 로 온다. 본문을 그대로 보여주는 편이 낫다.
        raise SystemExit(f"응답 파싱 실패. 원문 앞부분:\n{r.text[:400]}")

    items = body.get("items") or {}
    item = items.get("item") if isinstance(items, dict) else None
    if item is None:
        recs = []
    elif isinstance(item, dict):
        recs = [item]
    else:
        recs = list(item)

    cache.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    return recs


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def assign_records(members: list, recs: list[dict]) -> list[tuple]:
    """같은 지번의 폴리곤들과 대장 레코드들을 면적이 가까운 것끼리 1:1 로 짝짓는다.

    mgmBldrgstPk(14자리)와 bd_mgt_sn(25자리)은 키 체계가 달라 직접 매칭이
    안 된다. 대신 폴리곤 면적과 건축면적의 로그 비를 비용으로 두고 최적
    할당을 풀어 1:1 대응을 복원한다.

    **이 매칭은 속성(높이·용도·주소·건물명)을 붙이는 데만 쓴다.**
    부지형 판정에 쓰면 '면적이 가장 가까운 레코드'를 고르는 셈이라 비율이
    1 로 끌려가 판정이 무의미해진다. 판정은 계속 지번 최대 건축면적으로 한다.
    """
    if not recs:
        return [(None, "대장없음")] * len(members)
    if len(recs) == 1:
        kind = "단독" if len(members) == 1 else "지번공유"
        return [(recs[0], kind)] * len(members)

    areas = [to_float(r.get("archArea")) for r in recs]
    cost = np.zeros((len(members), len(recs)), dtype=float)
    for i, (poly_m2, _, _) in enumerate(members):
        for j, a in enumerate(areas):
            cost[i, j] = abs(math.log(max(poly_m2, 1.0) / max(a, 1.0)))

    biggest = max(recs, key=lambda r: to_float(r.get("archArea")))
    out: list[tuple] = [(biggest, "대표값")] * len(members)
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        out[i] = (recs[j], "면적매칭")
    return out


def verdict(poly_m2: float, arch: float) -> tuple[str, float]:
    if arch <= 0:
        return "대장없음", 0.0
    ratio = poly_m2 / arch
    if ratio > RATIO_MAX:
        return "부지형", ratio
    if ratio < RATIO_MIN:
        return "폴리곤과소", ratio
    return "정상", ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=config.DEFAULT_REGION)
    ap.add_argument("--min-roof", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.12, help="호출 간격(초)")
    args = ap.parse_args()

    key = load_key()
    region = config.get_region(args.region)
    lat_mid = (region.n + region.s) / 2

    out_dir = region.dir / "ledger"
    cache_dir = out_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats = load_buildings(region.buildings)
    targets = []
    for ft in feats:
        rings = polygon_rings(ft["geometry"])
        a = sum(shoelace_m2(r, lat_mid) for r in rings)
        if a >= args.min_roof:
            targets.append((a, ft))
    targets.sort(key=lambda t: -t[0])
    if args.limit:
        targets = targets[:args.limit]

    print(f"대상 {len(targets)}동 (>= {args.min_roof:.0f} m2)")

    # 지번 단위로 묶는다. 같은 지번의 폴리곤들과 대장 레코드들을 한자리에
    # 모아야 면적 기반 1:1 매칭을 풀 수 있다.
    groups: dict[tuple, list] = {}
    orphan = []
    for poly_m2, ft in targets:
        sn = ft["properties"].get("bd_mgt_sn", "")
        parsed = parse_mgt_sn(sn)
        if parsed is None:
            orphan.append((poly_m2, ft, sn))
        else:
            groups.setdefault(parsed, []).append((poly_m2, ft, sn))
    print(f"지번 {len(groups)}개로 묶임 (번호이상 {len(orphan)}동)")

    def blank(poly_m2, ft, sn, v="번호이상"):
        return {"building_id": ft["id"].split(".")[-1], "bd_mgt_sn": sn,
                "poly_m2": round(poly_m2), "jibun_n": 0, "arch_sum": 0.0,
                "arch_max": 0.0, "arch_matched": 0.0, "match_kind": "",
                "tot_area": 0.0, "heit": 0.0, "main_purps": "", "roof_cd": "",
                "bld_nm": "", "plat_plc": "", "new_plat_plc": "",
                "area_ratio": 0.0, "verdict": v}

    rows, calls, t0 = [], 0, time.time()
    for n, (parsed, members) in enumerate(groups.items()):
        sigungu, bjdong, plat, bun, ji = parsed
        cache = cache_dir / f"{sigungu}_{bjdong}_{plat}_{bun}_{ji}.json"
        was_cached = cache.exists()
        try:
            recs = fetch_jibun(key, sigungu, bjdong, plat, bun, ji, cache_dir)
        except requests.RequestException as e:
            print(f"  호출 실패 {sigungu}_{bjdong}_{bun}_{ji}: {e}")
            rows += [blank(p, f, s, "호출실패") for p, f, s in members]
            continue
        if not was_cached:
            calls += 1
            time.sleep(args.sleep)

        arch_sum = sum(to_float(r.get("archArea")) for r in recs)
        arch_max = max((to_float(r.get("archArea")) for r in recs), default=0.0)

        # 판정은 '그 지번의 가장 큰 건물'과 비교한다. 합계로 비교하면 한 지번에
        # 여러 동이 선 시설에서 부지형을 놓친다.
        #  골프연습장 908번지: 폴리곤 12,520 / 합계 7,547 = 1.66 -> 통과해버림
        #                      폴리곤 12,520 / 최대 2,337 = 5.36 -> 걸림
        # 속성(높이·용도·주소)은 아래 면적 매칭으로 동별로 붙인다.
        for (poly_m2, ft, sn), (rec, kind) in zip(members,
                                                  assign_records(members, recs)):
            v, ratio = verdict(poly_m2, arch_max)
            r = rec or {}
            rows.append({
                "building_id": ft["id"].split(".")[-1],
                "bd_mgt_sn": sn,
                "poly_m2": round(poly_m2),
                "jibun_n": len(recs),
                "arch_sum": round(arch_sum, 1),
                "arch_max": round(arch_max, 1),
                "arch_matched": round(to_float(r.get("archArea")), 1),
                "match_kind": kind,
                "tot_area": round(to_float(r.get("totArea")), 1),
                "heit": round(to_float(r.get("heit")), 1),
                "main_purps": r.get("mainPurpsCdNm", "") or "",
                "roof_cd": r.get("roofCdNm", "") or "",
                "bld_nm": (r.get("bldNm", "") or "").strip(),
                "plat_plc": (r.get("platPlc", "") or "").strip(),
                "new_plat_plc": (r.get("newPlatPlc", "") or "").strip(),
                "area_ratio": round(ratio, 3),
                "verdict": v,
            })

        if (n + 1) % 50 == 0:
            print(f"  지번 {n + 1}/{len(groups)}  호출 {calls}건  "
                  f"{time.time() - t0:.0f}s")

    rows += [blank(p, f, s) for p, f, s in orphan]

    csv_path = out_dir / "ledger.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    import collections
    vc = collections.Counter(r["verdict"] for r in rows)
    print(f"\n{len(rows)}동, 실제 API 호출 {calls}건, {time.time() - t0:.0f}s")
    print("판정:", dict(vc.most_common()))
    print("매칭:", dict(collections.Counter(r["match_kind"] for r in rows).most_common()))
    named = sum(1 for r in rows if r["bld_nm"])
    addr = sum(1 for r in rows if r["new_plat_plc"])
    print(f"건물명 확보 {named}동 ({100 * named / len(rows):.0f}%), "
          f"도로명주소 확보 {addr}동 ({100 * addr / len(rows):.0f}%)")
    pc = collections.Counter(r["main_purps"] for r in rows if r["main_purps"])
    print("주용도 상위:", pc.most_common(10))
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
