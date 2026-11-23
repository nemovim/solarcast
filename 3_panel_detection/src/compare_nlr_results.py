"""Compare existing NLR evaluation reports on verified identical validation data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


FIELDS = ('model', 'confidence', 'tp', 'fp', 'fn', 'precision', 'recall', 'f0_5',
          'empty_images', 'empty_images_with_fp', 'empty_image_fp_rate', 'empty_fp_instances')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def validation_identity(run):
    root = Path(run) / 'dataset/coco/val'
    payload = read_json(root / 'annotations.json')
    images = payload['images']
    if not images or len({item['file_name'] for item in images}) != len(images):
        raise ValueError(f'Empty or duplicate validation images: {root}')
    # Compare labels independently of JSON spacing, list order and COCO numeric IDs.
    by_id = {item['id']: item for item in images}
    if len(by_id) != len(images):
        raise ValueError(f'Duplicate COCO image IDs: {root}')
    categories = {item['id']: item['name'] for item in payload['categories']}
    labels = {item['file_name']: [] for item in images}
    for annotation in payload['annotations']:
        name = by_id[annotation['image_id']]['file_name']
        labels[name].append({key: annotation[key] for key in ('segmentation', 'bbox', 'area', 'iscrowd')})
        labels[name][-1]['category'] = categories[annotation['category_id']]
    entries = []
    for item in sorted(images, key=lambda row: row['file_name']):
        name = item['file_name']
        entries.append({'file_name': name, 'width': item['width'], 'height': item['height'],
                        'sha256': hashlib.sha256((root / 'images' / name).read_bytes()).hexdigest(),
                        'annotations': sorted(labels[name], key=canonical_hash)})
    return {'sha256': canonical_hash(entries), 'images': len(images),
            'instances': len(payload['annotations']),
            'empty_images': sum(not value for value in labels.values())}


def checked_metrics(metrics, identity):
    for key in FIELDS[1:]:
        if key not in metrics:
            raise ValueError(f'Missing evaluation metric: {key}')
    if metrics['tp'] + metrics['fn'] != identity['instances'] or metrics['empty_images'] != identity['empty_images']:
        raise ValueError('Report counts do not match this run\'s validation ground truth')
    return {key: metrics[key] for key in FIELDS[1:]}


def compare_runs(runs, common_confidence=0.15):
    names = [name for name, _ in runs]
    if not names or len(names) != len(set(names)) or 'original' in names:
        raise ValueError('Use unique model names other than the reserved name original')
    reference, baseline, selected, common, sources = None, None, [], [], []
    baseline_consistent = True
    for name, run in runs:
        run = Path(run).resolve()
        identity = validation_identity(run)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f'{name}: validation images/labels differ. Re-evaluate on exactly the same dataset before comparison.')
        report_path = run / 'evaluation/report.json'
        report = read_json(report_path)
        if report.get('selection_split') != 'val' or report.get('mask_iou') != 0.5:
            raise ValueError(f'{name}: incompatible evaluation protocol')
        if baseline is None:
            baseline = report['original']
            models = [('original', baseline), (name, report['finetuned'])]
        else:
            baseline_consistent &= canonical_hash(baseline) == canonical_hash(report['original'])
            models = [(name, report['finetuned'])]
        for model_name, model in models:
            selected.append({'model': model_name, **checked_metrics(model['selected_val'], identity),
                             'recall_target_met': model.get('recall_target_met')})
            curve = model['val_curve']
            fixed = next((row for row in curve if math.isclose(row['confidence'], common_confidence, abs_tol=1e-8)), None)
            if fixed is None:
                raise ValueError(f'{model_name}: confidence {common_confidence} is not present in val_curve')
            common.append({'model': model_name, **checked_metrics(fixed, identity)})
        sources.append({'model': name, 'run': str(run), 'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        'trained_checkpoint': report.get('trained_checkpoint')})
    return {'dataset': reference, 'mask_iou': 0.5, 'selection_split': 'val',
            'note': 'Validation results used for checkpoint/threshold selection; not independent test performance.',
            'baseline_consistent_across_reports': baseline_consistent,
            'common_confidence': common_confidence, 'selected_threshold': selected,
            'common_threshold': common, 'sources': sources}


def markdown_table(rows):
    lines = ['| Model | Conf | TP | FP | FN | Precision | Recall | Empty images with FP |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['confidence']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} | "
                     f"{row['precision']:.1%} | {row['recall']:.1%} | {row['empty_images_with_fp']}/{row['empty_images']} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=2, action='append', metavar=('NAME', 'RUN_DIRECTORY'), required=True)
    parser.add_argument('--common-confidence', type=float, default=0.15)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    report = compare_runs(args.run, args.common_confidence)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for key in ('selected_threshold', 'common_threshold'):
        rows = report[key]
        with (args.output / f'{key}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    text = '# NLR model comparison\n\n## Per-model selected confidence\n\n' + markdown_table(report['selected_threshold'])
    text += '\n\n## Common confidence\n\n' + markdown_table(report['common_threshold']) + '\n'
    if not report['baseline_consistent_across_reports']:
        text += '\nWarning: original-model results differ across runs; check model versions and evaluation settings.\n'
    (args.output / 'comparison.md').write_text(text, encoding='utf-8')
    print(text)
    print(f'Result: {args.output.resolve()}')


if __name__ == '__main__':
    main()
