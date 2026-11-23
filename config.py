"""데이터 사양 및 경로 설정.

이 패키지는 **수집이 끝난 영상 데이터**를 담고 있다. 추가 다운로드가 필요 없으므로
VWorld 인증키는 들어 있지 않다. 다시 수집해야 한다면 환경변수로 넘긴다.

    set VWORLD_KEY=발급받은키          (Windows CMD)
    $env:VWORLD_KEY="발급받은키"       (PowerShell)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- VWorld API ---------------------------------------------------------
# 인증키는 저장소에 커밋하지 않는다. 재수집이 필요할 때만 환경변수로 준다.
VWORLD_KEY = os.environ.get("VWORLD_KEY", "")
VWORLD_DOMAIN = "http://localhost:8080"

WMTS_URL = "https://api.vworld.kr/req/wmts/1.0.0/{key}/Satellite/{z}/{y}/{x}.jpeg"
DATA_URL = "https://api.vworld.kr/req/data"

# Satellite 레이어 유효 줌은 6~19. 19가 서버 상한이다.
ZOOM = 19
MAX_ZOOM = 19

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass(frozen=True)
class Region:
    name: str
    role: str
    w: float
    s: float
    e: float
    n: float
    desc: str

    @property
    def dir(self) -> Path:
        return DATA / self.name

    @property
    def raw_tiles(self) -> Path:
        return self.dir / "raw_tiles" / f"z{ZOOM}"

    @property
    def qc_report(self) -> Path:
        return self.dir / "qc" / "qc_report.csv"

    @property
    def buildings(self) -> Path:
        return self.dir / "vector" / "buildings.geojson"

    @property
    def mosaic(self) -> Path:
        return self.dir / "mosaic" / f"{self.name}_z{ZOOM}.tif"

    @property
    def chips_meta(self) -> Path:
        return self.dir / "chips" / "meta.csv"

    @property
    def chips_img(self) -> Path:
        return self.dir / "chips" / "images"

    @property
    def verify_dir(self) -> Path:
        return self.dir / "verify"

    def ensure_dirs(self) -> None:
        for d in (self.raw_tiles, self.qc_report.parent, self.buildings.parent,
                  self.mosaic.parent, self.chips_img, self.verify_dir):
            d.mkdir(parents=True, exist_ok=True)


REGIONS: dict[str, Region] = {
    # VWorld LT_C_ADEMD_INFO 행정동 폴리곤에서 직접 조회한 실측 경계
    # (빛가람동, emd_cd 12170134). 추정값이 아니다.
    "bitgaram": Region(
        name="bitgaram",
        role="positive",
        w=126.76610,
        s=35.00603,
        e=126.80793,
        n=35.03644,
        desc="빛가람동(나주혁신도시). 옥상 태양광 대상 지역.",
    ),
}

DEFAULT_REGION = "bitgaram"


def get_region(name: str) -> Region:
    if name not in REGIONS:
        raise SystemExit(f"알 수 없는 지역: {name} (가능: {', '.join(REGIONS)})")
    return REGIONS[name]


TILE_PX = 256
CHIP_SIZE = 512
CHIP_STRIDE = 256
CHIP_BLOCK = 6
DOWNLOAD_THREADS = 16
DOWNLOAD_RETRIES = 3
REQUEST_TIMEOUT = 30
TARGET_CLASS = "rooftop_pv"
