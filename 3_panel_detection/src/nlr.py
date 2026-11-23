"""Run the NLR/NatLabRockies RTMDet-Ins-X panel segmentation checkpoint."""

from __future__ import annotations

import argparse
import os
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


DEFAULT_CONFIG = PROJECT_ROOT / "models" / "nlr_config.py"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "nlr.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_BITGARAM_INPUT, help="image file or directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse tiles already present in the compact CSV manifests",
    )
    parser.add_argument(
        "--hard-exit-after-success",
        action="store_true",
        help=(
            "exit without running native-library shutdown hooks after all outputs are saved; "
            "use only on hosts that segfault in nvmlShutdown"
        ),
    )
    return parser.parse_args()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "to_ndarray"):
        value = value.to_ndarray()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def predict_tile(model, inference_detector, tile: Image.Image, conf: float) -> list[Prediction]:
    # MMDetection ndarray input follows OpenCV's BGR convention.
    bgr = np.asarray(tile.convert("RGB"))[:, :, ::-1].copy()
    result = inference_detector(model, bgr)
    instances = result.pred_instances
    scores = _as_numpy(instances.scores)
    keep = scores >= conf
    if not np.any(keep) or not hasattr(instances, "masks"):
        return []
    boxes = _as_numpy(instances.bboxes)[keep]
    class_ids = _as_numpy(instances.labels).astype(int)[keep]
    masks = _as_numpy(instances.masks)[keep].astype(bool)
    scores = scores[keep]
    names = model.dataset_meta.get("classes", ())
    predictions: list[Prediction] = []
    for box, score, class_id, mask in zip(boxes, scores, class_ids, masks):
        if mask.shape != (tile.height, tile.width):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    tile.size, Image.Resampling.NEAREST
                )
            ) > 0
        label = names[int(class_id)] if int(class_id) < len(names) else str(class_id)
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
    if args.output is None:
        args.output = PROJECT_ROOT / "outputs" / "nlr"
    missing = [str(path) for path in (args.config, args.model) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing NLR file(s): " + ", ".join(missing) +
            "\nRun: python -m src.download nlr"
        )
    try:
        import torch
        from mmengine import Config
        from mmdet.apis import inference_detector, init_detector
    except ImportError as exc:
        raise RuntimeError("Install the MMDetection environment described in README.md") from exc

    root, images = discover_images(args.input)
    if args.max_images is not None:
        images = images[: args.max_images]
    writer = CompactDatasetWriter(args.output, resume=args.resume)
    # Follow the upstream NLR example exactly: construct from config, then load
    # the checkpoint state_dict. This also avoids depending on checkpoint
    # metadata for the five domain-specific class names.
    config = Config.fromfile(str(args.config))
    # Use the same score floor as calibration; the CLI applies the final cutoff.
    config.model.backbone.init_cfg = None
    config.model.test_cfg.score_thr = min(0.05, args.conf)
    config.model.test_cfg.max_per_img = 100
    # Inference must not build a training/test dataset just to select a palette.
    model = init_detector(config, checkpoint=None, palette='random', device=args.device)
    checkpoint = torch.load(str(args.model), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.dataset_meta = {
        **getattr(model, "dataset_meta", {}),
        "classes": tuple(config.metainfo["classes"]),
    }
    for index, source in enumerate(images, start=1):
        key = image_key(source, root)
        if args.resume and writer.is_complete(key):
            print(f"[{index}/{len(images)}] reuse: {source.name}", flush=True)
            continue
        print(f"[{index}/{len(images)}] NLR: {source}", flush=True)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        predictions: list[Prediction] = []
        for left, top, right, bottom in tile_windows(
            image.width, image.height, args.tile_size, args.overlap
        ):
            tile = image.crop((left, top, right, bottom))
            predictions.extend(
                offset_prediction(item, left, top)
                for item in predict_tile(model, inference_detector, tile, args.conf)
            )
        predictions = class_aware_nms(predictions, iou_threshold=args.iou)
        writer.add(source, key, predictions)
        if index % 25 == 0:
            writer.flush()
    writer.flush()
    print(f"Done: {args.output.resolve()}")
    if args.hard_exit_after_success:
        # Some managed GPU hosts preload libcudahook, which can segfault inside
        # nvmlShutdown during normal interpreter teardown. All Python-managed
        # outputs are closed above, and synchronizing first ensures CUDA work is
        # complete before bypassing only the faulty native shutdown hooks.
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
