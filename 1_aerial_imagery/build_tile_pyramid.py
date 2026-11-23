"""z19 VWorld 원본 타일로 z14~18 축소 타일(피라미드)을 만든다.

웹 지도는 줌 16 근처에서 시작하는데 원본은 z19 한 단계뿐이라, 그대로 두면
VWorld 영상이 z19 이상에서만 보인다. 하위 줌 타일은 자식 2×2 를 이어 붙여
절반으로 줄이는 것을 반복해 만든다.

원본이 없는 자리는 투명으로 두어야 그 아래 Mapbox 위성이 비쳐 보인다.
JPEG 은 알파가 없으므로 축소 타일은 알파를 지원하면서 PNG 보다 훨씬 작은 WebP 로 저장한다.

    python 1_aerial_imagery/build_tile_pyramid.py [--region bitgaram] [--min-zoom 14]

출력: data/<region>/tiles/z{z}/{x}_{y}.webp  (재생성 가능한 산출물 — git 에 올리지 않는다)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

TILE = config.TILE_PX


def parse_xy(path: Path) -> tuple[int, int]:
    x, y = path.stem.split("_")
    return int(x), int(y)


def build_level(children: dict[tuple[int, int], Path], out_dir: Path) -> dict[tuple[int, int], Path]:
    """자식 줌 타일들로 한 단계 위 줌 타일을 만들고 {(x, y): 경로} 를 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parents = sorted({(x // 2, y // 2) for x, y in children})
    result: dict[tuple[int, int], Path] = {}

    for px, py in parents:
        canvas = Image.new("RGBA", (TILE * 2, TILE * 2), (0, 0, 0, 0))
        for dx in (0, 1):
            for dy in (0, 1):
                src = children.get((px * 2 + dx, py * 2 + dy))
                if src is None:
                    continue
                with Image.open(src) as im:
                    canvas.paste(im.convert("RGBA"), (dx * TILE, dy * TILE))
        out = out_dir / f"{px}_{py}.webp"
        canvas.resize((TILE, TILE), Image.LANCZOS).save(out, "WEBP", quality=85, method=6)
        result[(px, py)] = out

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", default=config.DEFAULT_REGION)
    ap.add_argument("--min-zoom", type=int, default=14)
    args = ap.parse_args()

    region = config.get_region(args.region)
    src_dir = region.raw_tiles
    tiles = {parse_xy(p): p for p in src_dir.glob("*.jpeg")}
    if not tiles:
        raise SystemExit(f"원본 타일이 없습니다: {src_dir}")
    print(f"z{config.ZOOM}: 원본 {len(tiles):,}장 ({src_dir})")

    out_root = region.dir / "tiles"
    level = tiles
    for z in range(config.ZOOM - 1, args.min_zoom - 1, -1):
        level = build_level(level, out_root / f"z{z}")
        print(f"z{z}: {len(level):,}장 -> {out_root / f'z{z}'}")


if __name__ == "__main__":
    main()
