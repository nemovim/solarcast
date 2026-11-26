"""Fine-tune NLR RTMDet-Ins as a one-class photovoltaic-panel model."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPO_ROOT / "data" / "bitgaram" / "training" / "human_v1" / "coco"
DEFAULT_CONFIG = REPO_ROOT / "modules" / "panel" / "configs" / "training.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "panel" / "nlr.pth"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "training" / "nlr_human_v3"
DEFAULT_OVERFIT_OUTPUT = REPO_ROOT / "outputs" / "training" / "nlr_overfit"
CLASS_NAME = "photovoltaic-panel"
ORIGINAL_CLASS_COUNT = 5
CLASSIFIER_KEY_MARKERS = ("rtm_cls", "cls_score", "conv_cls", "classifier")


def empty_safe_coco_metric(base):
    """Only genuine zero detections become zero AP; other errors stay fatal."""
    class EmptySafeCocoMetric(base):
        def compute_metrics(self, results):
            if not results:
                raise ValueError("No validation samples were evaluated")
            if (not self.format_only and set(self.metrics) <= {'bbox', 'segm'}
                    and all(len(pred['bboxes']) == 0 for _, pred in results)):
                logging.getLogger(__name__).warning(
                    "Zero detections across validation: recording AP=0 (not missing metrics)")
                items = self.metric_items or ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
                return {f'{metric}_{item}': 0.0 for metric in self.metrics for item in items}
            return super().compute_metrics(results)
    return EmptySafeCocoMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--val-dataset", type=Path, help="Separate COCO root containing val/; training remains under --dataset/train")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--overfit", action="store_true", help="diagnostic: no augmentation/early stopping, final epoch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--unfreeze-features", action="store_true", help="also train backbone/neck")
    parser.add_argument("--feature-mode", choices=("head", "neck", "full"))
    parser.add_argument("--classifier-init", choices=("average", "fresh", "single"), default="average")
    parser.add_argument("--small-panel-crop", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--skip-test", action="store_true", help="reserve test metrics until model selection")
    parser.add_argument("--train-val-only", action="store_true", help="two-split dataset; no test files or loop")
    parser.add_argument("--bitmap-masks", action="store_true", help="load exact COCO RLE masks as bitmaps")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate COCO files and write the derived config without importing MMDetection",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="skip training and evaluate an existing best checkpoint in the output directory",
    )
    parser.add_argument(
        "--hard-exit-after-success",
        action="store_true",
        help="bypass faulty native CUDA shutdown hooks after all outputs are flushed",
    )
    args = parser.parse_args()
    if args.overfit and (args.small_panel_crop or args.classifier_init != 'average' or args.test_only):
        parser.error('--overfit requires average initialization, no crop, and a new training run')
    args.epochs = args.epochs if args.epochs is not None else (100 if args.overfit else 12)
    args.feature_mode = args.feature_mode or ('full' if args.overfit else 'head')
    if args.overfit:
        args.skip_test = True
    if args.train_val_only:
        args.skip_test = True
        if args.test_only:
            parser.error('--test-only cannot be used with --train-val-only')
    if args.val_dataset:
        args.train_val_only = True
        args.skip_test = True
        if args.test_only:
            parser.error('--val-dataset does not support --test-only')
    return args


def coco_split_stats(dataset: Path, splits=("train", "val", "test")) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    split_names: dict[str, set[str]] = {}
    for split in splits:
        annotation_path = dataset / split / "annotations.json"
        image_dir = dataset / split / "images"
        if not annotation_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing COCO split={split}: need {annotation_path} and {image_dir}"
            )
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = [category.get("name") for category in payload.get("categories", [])]
        if categories != [CLASS_NAME]:
            raise ValueError(f"Expected category [{CLASS_NAME}], got {categories}")
        images = payload.get("images", [])
        annotations = payload.get("annotations", [])
        if not images:
            raise ValueError(f"Empty dataset split: {split}")
        split_names[split] = {image["file_name"] for image in images}
        annotated_ids = {annotation["image_id"] for annotation in annotations}
        missing_images = [
            image["file_name"]
            for image in images
            if not (image_dir / image["file_name"]).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(f"Missing {len(missing_images)} image file(s) in split={split}")
        stats[split] = {
            "images": len(images),
            "instances": len(annotations),
            "empty_images": sum(image["id"] not in annotated_ids for image in images),
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a not in split_names or b not in split_names:
            continue
        if split_names[a] & split_names[b]:
            raise ValueError(f"Image leakage between {a} and {b}")
    if "train" in stats and (not stats["train"]["instances"] or not stats["train"]["empty_images"]):
        raise ValueError("Training needs both panel annotations and reviewed empty images")
    return stats


def config_text(
    base_config: Path,
    dataset: Path,
    output: Path,
    checkpoint: Path,
    epochs: int,
    batch_size: int,
    accumulate: int,
    workers: int,
    learning_rate: float,
    seed: int,
    small_panel_crop: bool = False,
    patience: int = 3,
    overfit: bool = False,
    train_val_only: bool = False,
    bitmap_masks: bool = False,
    val_dataset: Path | None = None,
) -> str:
    if epochs < 1 or batch_size < 1 or accumulate < 1 or workers < 0:
        raise ValueError("epochs, batch-size and accumulate must be positive; workers cannot be negative")
    if learning_rate <= 0 or patience < 1:
        raise ValueError("learning-rate and patience must be positive")
    base = base_config.resolve().as_posix()
    root = dataset.resolve().as_posix() + "/"
    work_dir = output.resolve().as_posix()
    model_path = checkpoint.resolve().as_posix()
    persistent = workers > 0
    crop_step = "" if not small_panel_crop else """
    dict(type='RandomCrop', crop_type='absolute_range', crop_size=(160, 256),
         allow_negative_crop=True, recompute_bbox=True),"""
    if overfit and small_panel_crop:
        raise ValueError('Overfit diagnosis disables crop augmentation')
    text = f'''# Generated by src.train_nlr. Edit the CLI arguments, not this file.
_base_ = {base!r}

classes = ({CLASS_NAME!r},)
metainfo = dict(classes=classes)
data_root = {root!r}
work_dir = {work_dir!r}
load_from = None
resume = False
launcher = 'none'
randomness = dict(seed={seed}, deterministic=False)

# The reviewed masks are binary, so the original five-category head becomes one class.
model = dict(backbone=dict(init_cfg=None), bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask={bitmap_masks}),{crop_step}
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask={bitmap_masks}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    _delete_=True,
    batch_size={batch_size},
    num_workers={workers},
    persistent_workers={persistent},
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers={workers},
    persistent_workers={persistent},
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers={workers},
    persistent_workers={persistent},
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    _delete_=True,
    type='EmptySafeCocoMetric',
    ann_file=data_root + 'val/annotations.json',
    metric=['bbox', 'segm'],
    classwise=True,
)
test_evaluator = dict(
    _delete_=True,
    type='EmptySafeCocoMetric',
    ann_file=data_root + 'test/annotations.json',
    metric=['bbox', 'segm'],
    classwise=True,
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={epochs}, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    accumulative_counts={accumulate},
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type='AdamW', lr={learning_rate!r}, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={{'backbone': dict(lr_mult=0.1)}},
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=100),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end={epochs},
        T_max={epochs},
        eta_min={learning_rate * 0.05!r},
    ),
]
custom_hooks = [dict(
    type='EarlyStoppingHook', monitor='coco/segm_mAP', rule='greater',
    patience={patience}, min_delta=0.001, strict=True, check_finite=True,
)]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
)
visualizer = dict(
    name='training_' + {output.name!r},
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ]
)

# Loaded after MMEngine initialization. Five-class classifier tensors are
# collapsed to one class; any other incompatible tensors are skipped safely.
source_checkpoint = {model_path!r}
'''
    if overfit:
        text += '''
# Diagnostic only: train memorization, not a deployment model selection rule.
train_pipeline = [step for step in train_pipeline
                  if step['type'] not in ('YOLOXHSVRandomAug', 'RandomFlip', 'RandomCrop')]
train_dataloader['dataset']['pipeline'] = train_pipeline
optim_wrapper['optimizer']['weight_decay'] = 0.0
custom_hooks = []
default_hooks['checkpoint'].update(
    interval=10, max_keep_ckpts=-1, save_best=None, save_last=True,
)
'''
    if train_val_only:
        text += '''
# This dataset has train/val only; never alias validation as a test split.
test_dataloader = None
test_evaluator = None
test_cfg = None
'''
    if val_dataset is not None:
        val_root = val_dataset.resolve().as_posix() + '/'
        text += f"\nval_dataloader['dataset']['data_root'] = {val_root!r}\n"
        text += f"val_evaluator['ann_file'] = {val_root!r} + 'val/annotations.json'\n"
    return text


def collapse_classification_tensor(
    key: str,
    value: object,
    target_value: object,
    original_classes: int,
):
    """Average class channels when converting the original head to one class."""

    source_shape = tuple(getattr(value, "shape", ()))
    target_shape = tuple(getattr(target_value, "shape", ()))
    is_classifier = any(marker in key for marker in CLASSIFIER_KEY_MARKERS)
    if (
        not is_classifier
        or not source_shape
        or len(source_shape) != len(target_shape)
        or source_shape[0] != target_shape[0] * original_classes
        or source_shape[1:] != target_shape[1:]
    ):
        return None
    reshaped = value.reshape((target_shape[0], original_classes, *source_shape[1:]))
    try:
        return reshaped.mean(dim=1)
    except TypeError:
        return reshaped.mean(axis=1)


def compatible_state_dict(
    source: dict[str, object],
    target: dict[str, object],
    collapse_classes: int | None = None,
):
    compatible: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for original_key, value in source.items():
        key = original_key.removeprefix("module.")
        if key not in target:
            skipped[original_key] = "missing_target_key"
            continue
        source_shape = tuple(getattr(value, "shape", ()))
        target_shape = tuple(getattr(target[key], "shape", ()))
        if source_shape != target_shape:
            collapsed = (
                collapse_classification_tensor(key, value, target[key], collapse_classes)
                if collapse_classes
                else None
            )
            if collapsed is not None:
                compatible[key] = collapsed
                continue
            skipped[original_key] = f"shape_{source_shape}_to_{target_shape}"
            continue
        compatible[key] = value
    return compatible, skipped


def load_compatible_checkpoint(
    model,
    checkpoint_path: Path,
    torch,
    collapse_classes: int | None = None,
):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, dict):
        raise TypeError("NLR checkpoint does not contain a state_dict")
    target = model.state_dict()
    compatible, skipped = compatible_state_dict(source, target, collapse_classes)
    normalized_source = {key.removeprefix("module."): value for key, value in source.items()}
    collapsed_keys = [
        key
        for key, value in compatible.items()
        if key in target
        and tuple(getattr(normalized_source.get(key, value), "shape", ()))
        != tuple(getattr(target[key], "shape", ()))
    ]
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "source_tensors": len(source),
        "loaded_tensors": len(compatible),
        "skipped_tensors": len(skipped),
        "missing_after_load": list(missing),
        "unexpected_after_load": list(unexpected),
        "loaded_keys": list(compatible),
        "collapsed_classification_keys": collapsed_keys,
        "skipped": skipped,
    }


def initialize_then_load_transfer(model, checkpoint_path: Path, torch, classifier_init="average"):
    """Initialize MMEngine first, then install weights that training must retain."""

    if not hasattr(model, "init_weights"):
        raise TypeError("MMDetection model does not provide init_weights()")
    model.init_weights()
    if hasattr(model, "is_init") and not model.is_init:
        raise RuntimeError("MMEngine model did not mark itself initialized")
    report = load_compatible_checkpoint(
        model,
        checkpoint_path,
        torch,
        collapse_classes=ORIGINAL_CLASS_COUNT if classifier_init == "average" else None,
    )
    report["initialized_before_transfer"] = True
    return report


def transfer_guard_keys(loaded_keys: list[str], collapsed_keys: list[str]) -> list[str]:
    """Choose small representative tensors for a pre-training integrity guard."""

    selected: list[str] = []
    for marker in ("backbone", "neck"):
        match = next((key for key in loaded_keys if marker in key), None)
        if match:
            selected.append(match)
    if collapsed_keys:
        selected.append(collapsed_keys[0])
    elif loaded_keys:
        selected.append(loaded_keys[-1])
    return list(dict.fromkeys(selected))


def validate_transfer_report(report: dict[str, object], classifier_init="average") -> None:
    loaded = list(report.get("loaded_keys", []))
    collapsed = list(report.get("collapsed_classification_keys", []))
    missing_groups = [
        marker for marker in ("backbone", "neck")
        if not any(marker in key for key in loaded)
    ]
    if missing_groups:
        raise RuntimeError(
            "Transfer did not load required model groups: " + ", ".join(missing_groups)
        )
    if classifier_init == "single":
        if report.get("missing_after_load") or report.get("skipped_tensors"):
            raise RuntimeError("Single-class transfer requires a fully matching checkpoint")
        if not any(any(marker in key for marker in CLASSIFIER_KEY_MARKERS) for key in loaded):
            raise RuntimeError("Single-class transfer did not load classifier tensors")
    elif classifier_init == "fresh":
        missing = list(report.get("missing_after_load", []))
        if not missing or any(not any(marker in key for marker in CLASSIFIER_KEY_MARKERS)
                              for key in missing):
            raise RuntimeError("Fresh initialization must leave only classifier tensors unloaded")
    elif not collapsed:
        raise RuntimeError(
            "No five-class classifier tensor was collapsed into the one-class head"
        )


def install_transfer_guard(runner, keys: list[str], Hook, torch) -> None:
    """Abort if Runner initialization changes transferred tensors before epoch 1."""

    expected = {
        key: runner.model.state_dict()[key].detach().cpu().clone()
        for key in keys
    }

    class TransferGuardHook(Hook):
        priority = "HIGHEST"

        def before_train(self, guarded_runner) -> None:
            current = guarded_runner.model.state_dict()
            changed = [
                key
                for key, value in expected.items()
                if key not in current
                or not torch.equal(current[key].detach().cpu(), value)
            ]
            if changed:
                raise RuntimeError(
                    "Transferred weights changed before training: " + ", ".join(changed)
                )
            print("transfer_guard=ok keys=" + ",".join(expected), flush=True)

    runner.register_hook(TransferGuardHook(), priority="HIGHEST")


def existing_best_checkpoint(output: Path) -> Path:
    canonical = output / "best_model.pth"
    if canonical.is_file():
        return canonical
    candidates = list(output.glob("best_coco_segm_mAP_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {output}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def select_trained_checkpoint(output, epochs, best_value, overfit=False):
    if overfit:
        final = output / f'epoch_{epochs}.pth'
        if not final.is_file():
            raise FileNotFoundError(f'Overfit diagnosis requires the final checkpoint: {final}')
        return final.resolve()
    return Path(best_value).resolve() if best_value else existing_best_checkpoint(output).resolve()


def inference_dataset_config(pipeline, metainfo):
    """A valid metadata-only dataset for init_detector; no test annotations needed."""
    return dict(type='CocoDataset', metainfo=copy.deepcopy(dict(metainfo)),
                test_mode=True, lazy_init=True,
                pipeline=[copy.deepcopy(step) for step in pipeline if step['type'] != 'LoadAnnotations'])


def freeze_features(runner, Hook, mode="head") -> dict[str, int]:
    """Train the head while preserving transferred features and BN statistics."""

    names = {"head": ("backbone", "neck"), "neck": ("backbone",), "full": ()}[mode]
    modules = [getattr(runner.model, name) for name in names]
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    class FrozenFeaturesHook(Hook):
        def before_train_iter(self, guarded_runner, batch_idx, data_batch=None):
            # The loop sets model.train() after before_train_epoch().
            for module in modules:
                module.eval()

    runner.register_hook(FrozenFeaturesHook(), priority="HIGHEST")
    counts = {
        "trainable_parameters": sum(p.numel() for p in runner.model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in runner.model.parameters() if not p.requires_grad),
    }
    if not counts["trainable_parameters"]:
        raise RuntimeError("No trainable head parameters")
    return counts


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    base_config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if args.output == DEFAULT_OUTPUT and not args.validate_only and not args.test_only:
        if args.overfit:
            output = DEFAULT_OVERFIT_OUTPUT
        output = output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.val_dataset:
        stats = coco_split_stats(dataset, ('train',))
        stats.update(coco_split_stats(args.val_dataset.resolve(), ('val',)))
        from modules.panel.training.leakage import check_evaluation_leakage
        check_evaluation_leakage(dataset, args.val_dataset.resolve())
    else:
        stats = coco_split_stats(dataset, ("train", "val") if args.train_val_only else ("train", "val", "test"))
    output.mkdir(parents=True, exist_ok=True)
    generated_config = output / "nlr_config.py"
    if args.overfit and generated_config.exists():
        raise FileExistsError('Use a new output directory for the overfit diagnosis')
    if not args.test_only:
        generated_config.write_text(
            config_text(
                base_config, dataset, output, checkpoint, args.epochs,
                args.batch_size, args.accumulate, args.workers,
                args.learning_rate, args.seed,
                args.small_panel_crop, args.patience, args.overfit,
                args.train_val_only, args.bitmap_masks,
                args.val_dataset,
            ),
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"config={generated_config}")
    if args.validate_only:
        print("validate_only=ok")
        return
    required_files = (base_config,) if args.test_only else (base_config, checkpoint)
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing NLR model file(s): "
            + ", ".join(str(path) for path in missing)
            + "\nProvide --config and --checkpoint for the NLR model."
        )
    if not args.device.startswith("cuda"):
        raise ValueError("NLR RTMDet-Ins-X fine-tuning currently requires a CUDA device")

    try:
        import torch
        from mmengine import Config
        from mmengine.hooks import Hook
        from mmengine.runner import Runner
        from mmdet.utils import register_all_modules
    except ImportError as exc:
        raise RuntimeError("Install the NLR MMDetection environment described in README.md") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    torch.cuda.set_device(device_index)
    os.environ.setdefault("LOCAL_RANK", "0")
    register_all_modules(init_default_scope=True)
    from mmdet.evaluation.metrics import CocoMetric
    from mmdet.registry import METRICS
    METRICS.register_module(module=empty_safe_coco_metric(CocoMetric))
    config = Config.fromfile(str(generated_config))
    if args.test_only:
        runner = Runner.from_cfg(config)
        # Runner.test() initializes weights too. Mark initialization complete
        # BEFORE loading the trained weights, just as in the training path.
        runner.model.init_weights()
        best_checkpoint = existing_best_checkpoint(output).resolve()
        print(f"training=skipped checkpoint={best_checkpoint}", flush=True)
    else:
        # Run the epoch-0 probe in a disposable Runner. CheckpointHook is only
        # initialized by before_train(), so allowing it into a standalone val
        # run raises before epoch 1. A separate Runner also prevents val's
        # after_run hook from closing the training visualizer.
        baseline_config = copy.deepcopy(config)
        baseline_config.work_dir = str(output / "pretraining_validation")
        baseline_config.experiment_name = "pretraining_validation"
        baseline_config.visualizer.name = "before_training_" + output.name
        baseline_config.default_hooks = copy.deepcopy(config.default_hooks)
        baseline_config.default_hooks.checkpoint = None
        baseline_config.custom_hooks = []
        baseline_runner = Runner.from_cfg(baseline_config)
        baseline_report = initialize_then_load_transfer(
            baseline_runner.model, checkpoint, torch, args.classifier_init
        )
        validate_transfer_report(baseline_report, args.classifier_init)
        baseline_metrics = baseline_runner.val()
        (output / "before_training_val_metrics.json").write_text(
            json.dumps(baseline_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"before_training_val_metrics={baseline_metrics}", flush=True)
        del baseline_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        runner = Runner.from_cfg(config)
        report = initialize_then_load_transfer(runner.model, checkpoint, torch, args.classifier_init)
        validate_transfer_report(report, args.classifier_init)
        (output / "transfer_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"checkpoint_loaded={report['loaded_tensors']} skipped={report['skipped_tensors']}",
            flush=True,
        )
        guard_keys = transfer_guard_keys(
            report["loaded_keys"], report["collapsed_classification_keys"]
        )
        install_transfer_guard(runner, guard_keys, Hook, torch)
        feature_mode = "full" if args.unfreeze_features else args.feature_mode
        frozen = {"mode": feature_mode, **freeze_features(runner, Hook, feature_mode)}
        print(f"training_features={frozen}", flush=True)
        (output / "training_setup.json").write_text(
            json.dumps({"stats": stats, "features": frozen, "epochs": args.epochs,
                        "learning_rate": args.learning_rate, "best_metric": "coco/segm_mAP",
                        "classifier_init": args.classifier_init, "small_panel_crop": args.small_panel_crop,
                        "patience": args.patience, "seed": args.seed,
                        "overfit_diagnostic": args.overfit,
                        "checkpoint_selection": "final_epoch" if args.overfit else "best_val_segm_mAP",
                        "augmentation": not args.overfit,
                        "train_val_only": args.train_val_only, "bitmap_masks": args.bitmap_masks,
                        "early_stopping": not args.overfit}, indent=2),
            encoding="utf-8",
        )
        runner.train()
        best_value = runner.message_hub.get_info("best_ckpt")
        best_checkpoint = select_trained_checkpoint(output, args.epochs, best_value, args.overfit)

    # MMEngine 0.10.7 delegates to torch.load without weights_only=False.
    # PyTorch 2.6+ therefore rejects MMEngine's own HistoryBuffer metadata.
    # These checkpoints were produced locally by this training run, so load
    # them explicitly through the trusted checkpoint path used above.
    best_report = load_compatible_checkpoint(runner.model, best_checkpoint, torch)
    if best_report['missing_after_load'] or best_report['unexpected_after_load']:
        raise RuntimeError('Best checkpoint must load all model tensors; refusing partial evaluation')
    (output / "best_checkpoint_load_report.json").write_text(
        json.dumps(best_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    canonical_checkpoint = output / ("final_model.pth" if args.overfit else "best_model.pth")
    if best_checkpoint != canonical_checkpoint:
        shutil.copy2(best_checkpoint, canonical_checkpoint)
    print(f"inference_checkpoint={canonical_checkpoint}", flush=True)
    # Keep inference initialization identical to validation (no load_from/pretrained).
    inference_config = copy.deepcopy(config)
    inference_config.model.backbone.init_cfg = None
    for name in ("val_dataloader", "test_dataloader"):
        if inference_config.get(name) is None:
            continue
        pipeline = inference_config[name].dataset.pipeline
        inference_config[name].dataset.pipeline = [
            step for step in pipeline if step['type'] != 'LoadAnnotations'
        ]
    inference_config.test_pipeline = [
        step for step in inference_config.test_pipeline if step['type'] != 'LoadAnnotations'
    ]
    if inference_config.get("test_dataloader") is None:
        # inference_detector looks up the pipeline under test_dataloader.dataset.
        # init_detector may also build it lazily to obtain palette metadata.
        inference_config.test_dataloader = dict(dataset=inference_dataset_config(
            inference_config.test_pipeline, inference_config.metainfo))
    inference_config.dump(str(output / "inference_config.py"))
    if not args.skip_test:
        metrics = runner.test()
        (output / "test_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"test_metrics={output / 'test_metrics.json'}")
    if args.hard_exit_after_success:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
