"""Controlled NLR ablations on one reviewed snapshot; select using val only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.train_nlr import DEFAULT_DATASET, PROJECT_ROOT, coco_split_stats


# Adjacent experiments change one factor. No predictions become new labels.
EXPERIMENTS = (
    ('head_average', 'head', 'average', False),
    ('neck_average', 'neck', 'average', False),
    ('neck_average_crop', 'neck', 'average', True),
    ('neck_fresh_crop', 'neck', 'fresh', True),
)


def choose_experiment(rows):
    """Never call a model a winner when none meets the val recall target."""
    feasible = [row for row in rows if row['recall_target_met']]
    rank = lambda row: (row['val']['empty_images_with_fp'], -row['val']['f0_5'], row['name'])
    if feasible:
        return min(feasible, key=rank)['name']
    return None


def run_command(command, path):
    # Capture print() as well as MMEngine logs, including transfer guards.
    with path.open('w', encoding='utf-8') as log:
        try:
            subprocess.run([sys.executable, '-u', '-m', *command], cwd=PROJECT_ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Command failed. Read {path}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true', help='reuse completed experiments; restart failed ones')
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    stats = coco_split_stats(dataset)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name, *_ in EXPERIMENTS):
        raise FileExistsError('Use a new comparison directory; existing experiments are preserved')
    digest = hashlib.sha256()
    for split in ('train', 'val', 'test'):
        digest.update((dataset / split / 'annotations.json').read_bytes())
    report = {'selection_split': 'val', 'test_used_for_selection': False,
              'target_recall': 0.8, 'dataset': str(dataset), 'dataset_annotation_sha256': digest.hexdigest(),
              'stats': stats, 'epochs': args.epochs, 'patience': args.patience,
              'experiments': [], 'selected_experiment': None, 'status': 'running'}
    report_path = output / 'comparison.json'
    if args.resume:
        saved = json.loads(report_path.read_text(encoding='utf-8'))
        for key in ('dataset', 'dataset_annotation_sha256', 'epochs', 'patience', 'stats'):
            if saved[key] != report[key]:
                raise ValueError(f'Cannot resume: {key} changed')
        if saved.get('status') == 'complete':
            print(f'Already complete: {report_path}', flush=True)
            return
        report['experiments'] = saved['experiments']
    completed = {row['name']: row for row in report['experiments']}
    if len(completed) != len(report['experiments']) or set(completed) - {e[0] for e in EXPERIMENTS}:
        raise ValueError('Invalid completed experiments in comparison.json')
    for name, feature_mode, classifier_init, crop in EXPERIMENTS:
        run = output / name
        if name in completed:
            row = completed[name]
            if (Path(row['run']).resolve() != run or row['feature_mode'] != feature_mode
                    or row['classifier_init'] != classifier_init or row['small_panel_crop'] != crop):
                raise ValueError(f'Cannot reuse changed experiment: {name}')
            for artifact in ('best_model.pth', 'inference_config.py', 'evaluation/report.json'):
                if not (run / artifact).is_file():
                    raise FileNotFoundError(f'Completed experiment missing {run / artifact}')
            print(f'Reusing completed experiment: {name}', flush=True)
            continue
        if run.exists():
            archive = run.with_name(name + '.failed_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            run.rename(archive)
            print(f'Preserved previous attempt: {archive}', flush=True)
        run.mkdir()
        print(f'Training {name}; live output: {run / "train_console.log"}', flush=True)
        command = ['src.train_nlr', '--dataset', str(dataset), '--output', str(run),
                   '--epochs', str(args.epochs), '--patience', str(args.patience),
                   '--feature-mode', feature_mode, '--classifier-init', classifier_init,
                   '--device', args.device, '--skip-test', '--hard-exit-after-success']
        if crop:
            command.append('--small-panel-crop')
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        try:
            run_command(command, run / 'train_console.log')
            print(f'Validating {name}', flush=True)
            run_command(['src.evaluate_nlr', '--dataset', str(dataset), '--run', str(run),
                         '--device', args.device, '--val-only', '--hard-exit-after-success'],
                        run / 'evaluation_console.log')
        except RuntimeError as exc:
            report.update(status='failed', failed_experiment=name, error=str(exc))
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise
        evaluation = json.loads((run / 'evaluation/report.json').read_text(encoding='utf-8'))
        trained = evaluation['finetuned']
        row = {'name': name, 'run': str(run), 'feature_mode': feature_mode,
               'classifier_init': classifier_init, 'small_panel_crop': crop,
               'val': trained['selected_val'], 'recall_target_met': trained['recall_target_met'],
               'val_at_0_15': trained['val_at_0_15'], 'val_at_0_25': trained['val_at_0_25']}
        report['experiments'].append(row)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'{name}: val recall={row["val"]["recall"]:.3f} '
              f'precision={row["val"]["precision"]:.3f} '
              f'empty FP images={row["val"]["empty_images_with_fp"]}', flush=True)
    report['selected_experiment'] = choose_experiment(report['experiments'])
    report['status'] = 'testing_selected_model' if report['selected_experiment'] else 'recall_target_not_met'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report['selected_experiment']:
        selected = output / report['selected_experiment']
        print(f'Selected on val: {selected}', flush=True)
        # The held-out test is evaluated only after selection, never ranked.
        run_command(['src.train_nlr', '--test-only', '--dataset', str(dataset),
                     '--output', str(selected), '--device', args.device, '--hard-exit-after-success'],
                    selected / 'test_console.log')
        run_command(['src.evaluate_nlr', '--dataset', str(dataset), '--run', str(selected),
                     '--device', args.device, '--hard-exit-after-success'],
                    selected / 'test_evaluation_console.log')
        report['status'] = 'complete'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    else:
        print('No model met 80% val recall. Test remains untouched; inspect comparison.json.', flush=True)
    print(f'Comparison: {report_path}', flush=True)


if __name__ == '__main__':
    main()
