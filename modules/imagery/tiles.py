"""Web Mercator pixel transform for VWorld z19 tiles."""
from affine import Affine

WORLD = 40075016.685578488
Z19_PIXEL = WORLD / (256 * (2 ** 19))
# The existing Bitgaram study covers this complete z19 rectangle (3,410 tiles).
BITGARAM_BOUNDS = (446760, 207604, 446821, 207658)


def tile_transform(x: int, y: int) -> Affine:
    return Affine(Z19_PIXEL, 0, -WORLD / 2 + x * 256 * Z19_PIXEL,
                  0, -Z19_PIXEL, WORLD / 2 - y * 256 * Z19_PIXEL)
