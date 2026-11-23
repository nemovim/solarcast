"""Compact, resumable mask export shared by all inference backends."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.inference import Prediction


SUMMARY_FIELDS = (
    "key",
    "source_filename",
    "tile_id",
    "tile_x",
    "tile_y",
    "zoom",
    "center_latitude",
    "center_longitude",
    "width",
    "height",
    "instances",
    "max_confidence",
    "mean_confidence",
    "covered_fraction",
    "counts",
    "mask_path",
)

INSTANCE_FIELDS = (
    "key",
    "source_filename",
    "tile_id",
    "tile_x",
    "tile_y",
    "zoom",
    "instance_id",
    "label",
    "confidence",
    "mask_pixels",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "mask_path",
)


def slippy_tile_metadata(source: Path) -> dict[str, object]:
    """Extract XYZ indices and the Web Mercator tile-center coordinate."""

    tile_match = re.fullmatch(r"(-?\d+)_(-?\d+)", source.stem)
    zoom_match = next(
        (
            candidate
            for part in reversed(source.parts)
            if (candidate := re.fullmatch(r"z(\d+)", part, flags=re.IGNORECASE))
        ),
        None,
    )
    if not tile_match or not zoom_match:
        return {
            "tile_id": source.stem,
            "tile_x": None,
            "tile_y": None,
            "zoom": None,
            "center_latitude": None,
            "center_longitude": None,
        }
    tile_x, tile_y = (int(value) for value in tile_match.groups())
    zoom = int(zoom_match.group(1))
    scale = 2**zoom
    longitude = (tile_x + 0.5) / scale * 360.0 - 180.0
    mercator_y = math.pi * (1.0 - 2.0 * (tile_y + 0.5) / scale)
    latitude = math.degrees(math.atan(math.sinh(mercator_y)))
    return {
        "tile_id": source.stem,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "zoom": zoom,
        "center_latitude": round(latitude, 8),
        "center_longitude": round(longitude, 8),
    }


def _full_mask(size: tuple[int, int], prediction: Prediction) -> np.ndarray:
    width, height = size
    output = np.zeros((height, width), dtype=bool)
    local = np.asarray(prediction.mask, dtype=bool)
    if local.ndim != 2:
        raise ValueError(f"Mask for {prediction.label!r} must be 2D, got {local.shape}")
    left, top = prediction.offset
    right = min(width, left + local.shape[1])
    bottom = min(height, top + local.shape[0])
    if right > left and bottom > top:
        output[top:bottom, left:right] = local[: bottom - top, : right - left]
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: tuple[str, ...], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    temporary.replace(path)


class CompactDatasetWriter:
    """Write 1-bit union/instance masks plus two small CSV manifests."""

    def __init__(self, output: Path, resume: bool = False) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        summaries = _read_csv(output / "summary.csv") if resume else []
        instances = _read_csv(output / "instances.csv") if resume else []
        self.summaries = {row["key"]: row for row in summaries if row.get("key")}
        self.instances: dict[str, list[dict[str, object]]] = {}
        for row in instances:
            self.instances.setdefault(row.get("key", ""), []).append(row)

    def is_complete(self, key: str) -> bool:
        row = self.summaries.get(key)
        if not row:
            return False
        paths = [row.get("mask_path", "")]
        paths.extend(item.get("mask_path", "") for item in self.instances.get(key, []))
        try:
            expected = int(row.get("instances", -1))
        except (TypeError, ValueError):
            return False
        return (
            expected == len(self.instances.get(key, []))
            and all(path and (self.output / str(path)).is_file() for path in paths)
        )

    def add(self, source: Path, key: str, predictions: Sequence[Prediction]) -> None:
        with Image.open(source) as opened:
            size = opened.size
        width, height = size
        union = np.zeros((height, width), dtype=bool)
        metadata = slippy_tile_metadata(source)
        instance_rows: list[dict[str, object]] = []
        counts: dict[str, int] = {}
        scores: list[float] = []
        instance_dir = self.output / "instance_masks" / key
        if instance_dir.is_dir():
            for stale_mask in instance_dir.glob("*.png"):
                stale_mask.unlink()

        for index, prediction in enumerate(predictions, start=1):
            mask = _full_mask(size, prediction)
            union |= mask
            instance_id = f"{index:04d}"
            relative = Path("instance_masks") / key / f"{instance_id}.png"
            destination = self.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(destination, optimize=True)
            counts[prediction.label] = counts.get(prediction.label, 0) + 1
            scores.append(float(prediction.score))
            box = [round(float(value), 2) for value in prediction.box]
            instance_rows.append(
                {
                    "key": key,
                    "source_filename": source.name,
                    "tile_id": metadata["tile_id"],
                    "tile_x": metadata["tile_x"],
                    "tile_y": metadata["tile_y"],
                    "zoom": metadata["zoom"],
                    "instance_id": instance_id,
                    "label": prediction.label,
                    "confidence": round(float(prediction.score), 6),
                    "mask_pixels": int(mask.sum()),
                    "box_x1": box[0],
                    "box_y1": box[1],
                    "box_x2": box[2],
                    "box_y2": box[3],
                    "mask_path": relative.as_posix(),
                }
            )

        union_relative = Path("masks") / f"{key}.png"
        union_destination = self.output / union_relative
        union_destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(union).save(union_destination, optimize=True)
        self.summaries[key] = {
            "key": key,
            "source_filename": source.name,
            **metadata,
            "width": width,
            "height": height,
            "instances": len(predictions),
            "max_confidence": round(max(scores), 6) if scores else "",
            "mean_confidence": round(sum(scores) / len(scores), 6) if scores else "",
            "covered_fraction": round(float(union.mean()), 8),
            "counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
            "mask_path": union_relative.as_posix(),
        }
        self.instances[key] = instance_rows

    def flush(self) -> None:
        keys = sorted(self.summaries)
        summary_rows = [self.summaries[key] for key in keys]
        instance_rows = [row for key in keys for row in self.instances.get(key, [])]
        _write_csv(self.output / "summary.csv", SUMMARY_FIELDS, summary_rows)
        _write_csv(self.output / "instances.csv", INSTANCE_FIELDS, instance_rows)
