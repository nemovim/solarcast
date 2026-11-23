"""Run the agademer YOLO26-L solar-panel instance-segmentation checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    DEFAULT_BITGARAM_INPUT,
    Prediction,
    class_aware_nms,
    discover_images,
    image_key,
    offset_prediction,
    tile_windows,
)
from src.results import CompactDatasetWriter  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_BITGARAM_INPUT, help="image file or directory"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "yolo"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO network input size")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["photovoltaic panel"],
        help="class names to retain; use 'all' to keep building predictions too",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=(16, 32),
        default=16,
        help="inference precision: 16 (FP16) or 32 (FP32)",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _normalized_label(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def predict_tile(model, tile: Image.Image, args: argparse.Namespace) -> list[Prediction]:
    result = model.predict(
        # Passing PIL is deliberate: Ultralytics converts RGB PIL input to the
        # OpenCV-compatible BGR array expected by its preprocessing pipeline.
        source=tile.convert("RGB"),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        quantize=32 if str(args.device).casefold() == "cpu" else args.quantize,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.boxes is None or result.masks is None:
        return []
    boxes = _as_numpy(result.boxes.xyxy)
    scores = _as_numpy(result.boxes.conf)
    class_ids = _as_numpy(result.boxes.cls).astype(int)
    masks = _as_numpy(result.masks.data).astype(bool)
    names = result.names
    predictions: list[Prediction] = []
    for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        if mask.shape != (tile.height, tile.width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    tile.size, Image.Resampling.NEAREST
                )
            ) > 0
        label = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else names[int(class_id)]
        requested = {_normalized_label(value) for value in args.classes}
        if "all" not in requested and _normalized_label(str(label)) not in requested:
            continue
        predictions.append(
            Prediction(
                label=str(label),
                score=float(score),
                box=tuple(float(value) for value in box),
                mask=mask,
            )
        )
    return predictions


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}\nRun: python -m src.download yolo"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Run through run_yolo.sh to install the dependencies") from exc

    root, images = discover_images(args.input)
    if args.max_images is not None:
        images = images[: args.max_images]
    writer = CompactDatasetWriter(args.output, resume=args.resume)
    model = YOLO(str(args.model))
    for index, source in enumerate(images, start=1):
        key = image_key(source, root)
        if args.resume and writer.is_complete(key):
            print(f"[{index}/{len(images)}] reuse: {source.name}", flush=True)
            continue
        print(f"[{index}/{len(images)}] YOLO: {source}", flush=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        predictions: list[Prediction] = []
        for left, top, right, bottom in tile_windows(
            image.width, image.height, args.tile_size, args.overlap
        ):
            tile = image.crop((left, top, right, bottom))
            predictions.extend(
                offset_prediction(item, left, top) for item in predict_tile(model, tile, args)
            )
        predictions = class_aware_nms(predictions, iou_threshold=args.iou)
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
