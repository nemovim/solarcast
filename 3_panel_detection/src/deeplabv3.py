"""Run the Fraunhofer multi-resolution DeepLabV3 solar-panel model."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    DEFAULT_BITGARAM_INPUT,
    Prediction,
    discover_images,
    image_key,
)
from src.results import CompactDatasetWriter  # noqa: E402
from src.deeplab_core import ONNXPanelSegmenter, binary_mask, predict_tiled  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models" / "deeplabv3.onnx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "deeplabv3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_BITGARAM_INPUT, help="image file or directory"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.5, help="probability threshold")
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--closing-radius", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def component_predictions(mask: np.ndarray, probability: np.ndarray) -> list[Prediction]:
    """Convert one semantic mask into connected-component predictions."""

    candidate = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(candidate)
    height, width = candidate.shape
    predictions: list[Prediction] = []
    for start_y, start_x in zip(*np.nonzero(candidate)):
        if visited[start_y, start_x]:
            continue
        queue = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and candidate[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        coordinates = np.asarray(pixels, dtype=np.int32)
        ys, xs = coordinates[:, 0], coordinates[:, 1]
        left, top = int(xs.min()), int(ys.min())
        right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
        local_mask = np.zeros((bottom - top, right - left), dtype=bool)
        local_mask[ys - top, xs - left] = True
        predictions.append(
            Prediction(
                label="photovoltaic panel",
                score=float(np.asarray(probability)[ys, xs].mean()),
                box=(float(left), float(top), float(right), float(bottom)),
                mask=local_mask,
                offset=(left, top),
            )
        )
    return predictions


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}\nRun: python -m src.download deeplabv3"
        )
    root, images = discover_images(args.input)
    if args.max_images is not None:
        images = images[: args.max_images]
    writer = CompactDatasetWriter(args.output, resume=args.resume)
    segmenter = ONNXPanelSegmenter(args.model, use_gpu=args.device == "cuda")
    print(f"DeepLabV3 providers: {segmenter.providers}")

    for index, source in enumerate(images, start=1):
        key = image_key(source, root)
        if args.resume and writer.is_complete(key):
            print(f"[{index}/{len(images)}] reuse: {source.name}", flush=True)
            continue
        print(f"[{index}/{len(images)}] DeepLabV3: {source}", flush=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        probability = predict_tiled(image, segmenter, args.tile_size, args.overlap)
        mask = binary_mask(
            probability,
            threshold=args.conf,
            min_pixels=args.min_pixels,
            closing_radius=args.closing_radius,
        )
        predictions = component_predictions(mask, probability)
        writer.add(source, key, predictions)
        if index % 25 == 0:
            writer.flush()
    writer.flush()
    print(f"Done: {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
