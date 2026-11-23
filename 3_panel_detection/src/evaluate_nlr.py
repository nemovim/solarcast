"""Compare background false alarms and panel recall; calibrate on val only."""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference import class_aware_nms
from src.nlr import DEFAULT_CONFIG, DEFAULT_MODEL, predict_tile
from src.train_nlr import DEFAULT_DATASET


THRESHOLDS = [round(value / 100, 2) for value in range(5, 100, 5)]


def evaluation_dataset(run, explicit=None):
    """Read only literal paths from our generated config without loading CUDA."""
    if explicit is not None:
        return Path(explicit).resolve()
    path = Path(run) / 'nlr_config.py'
    if not path.is_file():
        raise FileNotFoundError(f'Cannot recover validation dataset: {path}; specify --dataset')
    root = override = None
    for statement in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == 'data_root':
                root = ast.literal_eval(statement.value)
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript):
                inner = target.value
                if (isinstance(inner.value, ast.Name) and inner.value.id == 'val_dataloader'
                        and isinstance(inner.slice, ast.Constant) and inner.slice.value == 'dataset'
                        and isinstance(target.slice, ast.Constant) and target.slice.value == 'data_root'):
                    override = ast.literal_eval(statement.value)
    selected = override or root
    if not selected:
        raise ValueError(f'No literal validation data_root in {path}; specify --dataset')
    return Path(selected).resolve()


def model_specs(run, checkpoint=None, extra_models=()):
    specs = [('original', DEFAULT_CONFIG, DEFAULT_MODEL),
             ('finetuned', run / 'inference_config.py', checkpoint or run / 'best_model.pth')]
    for name, config, weights in extra_models:
        if name in {item[0] for item in specs}:
            raise ValueError(f'Duplicate model name: {name}')
        specs.append((name, Path(config), Path(weights)))
    for name, config, weights in specs:
        for path in (config, weights):
            if not Path(path).is_file():
                raise FileNotFoundError(f'{name}: {path}')
    return specs


def write_comparison(output, report, names):
    from src.compare_nlr_results import markdown_table
    sections = []
    for key in ('selected_threshold', 'common_threshold'):
        rows = []
        for name in names:
            model = report[name]
            metrics = model['selected_val'] if key == 'selected_threshold' else model['val_at_0_15']
            rows.append({'model': name, **metrics})
        with (output / f'{key}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        sections.append(f'## {key}\n\n' + markdown_table(rows))
    (output / 'comparison.md').write_text(
        '# Model comparison on the same validation dataset\n\n'
        'Confidence is selected on validation; these are not independent test results.\n\n'
        + '\n\n'.join(sections) + '\n', encoding='utf-8')


def mask_iou_matrix(predictions, truth):
    output = np.zeros((len(predictions), len(truth)), dtype=float)
    for i, prediction in enumerate(predictions):
        mask = np.asarray(prediction.mask, dtype=bool)
        for j, target in enumerate(truth):
            union = np.count_nonzero(mask | target)
            output[i, j] = np.count_nonzero(mask & target) / union if union else 0
    return output


def image_counts(scores, ious, threshold):
    """Match instances once in confidence order at mask IoU >= 0.5."""
    matched = set()
    tp = fp = 0
    for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True):
        if scores[i] < threshold:
            continue
        candidates = [j for j in range(ious.shape[1]) if j not in matched and ious[i, j] >= 0.5]
        if candidates:
            matched.add(max(candidates, key=lambda j: ious[i, j]))
            tp += 1
        else:
            fp += 1
    return tp, fp, ious.shape[1] - tp


def summarize(records, threshold):
    tp = fp = fn = empty = empty_hits = empty_fp = 0
    for record in records:
        a, b, c = image_counts(record['scores'], record['ious'], threshold)
        tp += a; fp += b; fn += c
        if record['ious'].shape[1] == 0:
            empty += 1
            empty_hits += int(b > 0)
            empty_fp += b
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'confidence': threshold, 'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall,
        'f0_5': 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0,
        'empty_images': empty, 'empty_images_with_fp': empty_hits,
        'empty_fp_instances': empty_fp,
        'empty_image_fp_rate': empty_hits / empty if empty else None,
        'fp_per_empty_image': empty_fp / empty if empty else None,
    }


def choose_threshold(curve, minimum_recall=0.8):
    feasible = [row for row in curve if row['recall'] >= minimum_recall]
    if not feasible:
        return max(curve, key=lambda row: (row['f0_5'], row['recall'], row['confidence'])), False
    return min(feasible, key=lambda row: (
        row['empty_images_with_fp'], -row['f0_5'], -row['confidence']
    )), True


def train_diagnostic_summary(records, val_confidence):
    """No training-set threshold selection; calibration remains on validation."""
    return {
        'at_val_selected': summarize(records, val_confidence),
        'at_0_15': summarize(records, 0.15),
        'at_0_25': summarize(records, 0.25),
        'curve': [summarize(records, threshold) for threshold in THRESHOLDS],
    }


def collect(model, inference_detector, dataset, split, coco_class):
    coco = coco_class(str(dataset / split / 'annotations.json'))
    records = []
    for image_id in sorted(coco.imgs):
        info = coco.imgs[image_id]
        path = dataset / split / 'images' / info['file_name']
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        if max(image.size) > 640:
            raise ValueError('Evaluation expects raw tiles no larger than 640 pixels')
        predictions = predict_tile(model, inference_detector, image, 0.05)
        # Compare original 5-class and trained 1-class in the same panel label space.
        for prediction in predictions:
            prediction.label = 'panel'
        predictions = class_aware_nms(predictions, 0.55)
        truth = [coco.annToMask(annotation).astype(bool)
                 for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))]
        records.append({'filename': info['file_name'], 'scores': [p.score for p in predictions],
                        'ious': mask_iou_matrix(predictions, truth)})
        print(f'{split}: {info["file_name"]}', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, help='Evaluation COCO root; default: validation root saved in the training config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--val-only', action='store_true', help='do not inspect test during experiment selection')
    parser.add_argument('--train-diagnostic', action='store_true', help='measure memorization separately; never calibrate on train')
    parser.add_argument('--checkpoint', type=Path, help='explicit trained checkpoint (e.g. final_model.pth)')
    parser.add_argument('--extra-model', nargs=3, action='append', default=[],
                        metavar=('NAME', 'CONFIG', 'CHECKPOINT'), help='evaluate another model on the same dataset')
    parser.add_argument('--output', type=Path, help='new directory for comparison results')
    parser.add_argument('--hard-exit-after-success', action='store_true')
    args = parser.parse_args()
    args.dataset = evaluation_dataset(args.run, args.dataset)
    from src.train_nlr import coco_split_stats
    splits = ('train', 'val') if args.train_diagnostic else ('val',)
    if not args.val_only:
        splits = (*splits, 'test')
    coco_split_stats(args.dataset, splits)
    print(f'Evaluation dataset: {args.dataset}', flush=True)
    specs = model_specs(args.run, args.checkpoint, args.extra_model)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    import torch
    from mmengine import Config
    from mmdet.apis import init_detector, inference_detector
    from pycocotools.coco import COCO

    output = args.output or args.run / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    trained_checkpoint = args.checkpoint or args.run / 'best_model.pth'
    report = {'selection_split': 'val', 'mask_iou': 0.5, 'target_recall': 0.8,
              'dataset': str(args.dataset.resolve()),
              'model_sources': {name: {'config': str(Path(config).resolve()),
                                      'checkpoint': str(Path(weights).resolve())}
                                for name, config, weights in specs},
              'trained_checkpoint': str(trained_checkpoint.resolve()),
              'train_is_diagnostic_only': args.train_diagnostic,
              'note': ('Validation-only model/threshold selection; no independent test evaluation.'
                       if args.val_only else 'Previously inspected test split; use a new region for final confirmation.')}
    image_rows = []
    for name, config_path, checkpoint_path in specs:
        config = Config.fromfile(str(config_path))
        config.model.backbone.init_cfg = None
        # Low score floor for calibration, fixed class-agnostic panel NMS afterward.
        config.model.test_cfg.score_thr = 0.05
        config.model.test_cfg.max_per_img = 100
        # Explicit palette bypasses dataset construction for visualization metadata.
        # Also supports already-saved train/val-only configs with a pipeline-only dataset.
        model = init_detector(config, checkpoint=None, palette='random', device=args.device)
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
        model.dataset_meta = {'classes': tuple(config.metainfo['classes'])}
        model.eval()
        with torch.no_grad():
            val_records = collect(model, inference_detector, args.dataset, 'val', COCO)
            curve = [summarize(val_records, threshold) for threshold in THRESHOLDS]
            selected, met = choose_threshold(curve)
            test_records = [] if args.val_only else collect(model, inference_detector, args.dataset, 'test', COCO)
            train_records = collect(model, inference_detector, args.dataset, 'train', COCO) if args.train_diagnostic else []
        report[name] = {'val_curve': curve, 'selected_val': selected, 'recall_target_met': met,
                        'val_at_0_15': summarize(val_records, 0.15),
                        'val_at_0_25': summarize(val_records, 0.25),
                        'test_evaluated': not args.val_only}
        if not args.val_only:
            report[name].update(test_at_selected=summarize(test_records, selected['confidence']),
                                test_at_0_15=summarize(test_records, 0.15),
                                test_at_0_25=summarize(test_records, 0.25))
        if args.train_diagnostic:
            report[name]['train_diagnostic'] = train_diagnostic_summary(train_records, selected['confidence'])
        for split, records in (('val', val_records), ('test', test_records), ('train', train_records)):
            for record in records:
                tp, fp, fn = image_counts(record['scores'], record['ious'], selected['confidence'])
                image_rows.append({'model': name, 'split': split, 'filename': record['filename'],
                                   'confidence': selected['confidence'], 'tp': tp, 'fp': fp, 'fn': fn,
                                   'empty': record['ious'].shape[1] == 0})
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(image_rows[0]))
        writer.writeheader(); writer.writerows(image_rows)
    summary_key = 'selected_val' if args.val_only else 'test_at_selected'
    write_comparison(output, report, [spec[0] for spec in specs])
    print(json.dumps({name: report[name][summary_key] for name, _, _ in specs}, indent=2))
    print(f'evaluation={output.resolve()}', flush=True)
    if args.hard_exit_after_success:
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
