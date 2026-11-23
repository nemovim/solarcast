"""Small utilities shared by the three inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_BITGARAM_INPUT = (
    Path(__file__).resolve().parents[2] / "data" / "bitgaram" / "raw_tiles" / "z19"
)


@dataclass
class Prediction:
    label: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray
    offset: tuple[int, int] = (0, 0)

    @property
    def mask_pixels(self) -> int:
        return int(np.asarray(self.mask, dtype=bool).sum())


def discover_images(input_path: str | Path) -> tuple[Path, list[Path]]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    images = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return path, images


def image_key(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    return "__".join(part.replace(" ", "_") for part in relative.with_suffix("").parts)


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def tile_windows(
    width: int, height: int, tile_size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    return [
        (left, top, min(left + tile_size, width), min(top + tile_size, height))
        for top in _axis_starts(height, tile_size, overlap)
        for left in _axis_starts(width, tile_size, overlap)
    ]


def offset_prediction(prediction: Prediction, left: int, top: int) -> Prediction:
    x1, y1, x2, y2 = prediction.box
    return Prediction(
        label=prediction.label,
        score=float(prediction.score),
        box=(x1 + left, y1 + top, x2 + left, y2 + top),
        mask=np.asarray(prediction.mask, dtype=bool),
        offset=(left, top),
    )


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_nms(
    predictions: Iterable[Prediction], iou_threshold: float = 0.55
) -> list[Prediction]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    ordered = sorted(predictions, key=lambda item: item.score, reverse=True)
    kept: list[Prediction] = []
    for candidate in ordered:
        if not any(
            candidate.label == accepted.label
            and _box_iou(candidate.box, accepted.box) >= iou_threshold
            for accepted in kept
        ):
            kept.append(candidate)
    return kept
