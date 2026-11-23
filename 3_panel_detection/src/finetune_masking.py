"""Fine-tune the existing NLR pipeline on completed masking-v2 data only."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.export_masking import DEFAULT_MERGED, IMAGES, prepare_dataset
from src.train_nlr import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, PROJECT_ROOT


def run_step(command, logfile):
    print('Running: ' + ' '.join(map(str, command)), flush=True)
    with Path(logfile).open('w', encoding='utf-8') as log:
        with subprocess.Popen(list(map(str, command)), cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(f'Command exited with {code}; inspect {logfile}')


def training_command(args, dataset, output):
    command = [sys.executable, '-u', '-m', 'src.train_nlr', '--dataset', str(dataset),
               '--output', str(output), '--config', str(args.config.resolve()),
               '--checkpoint', str(args.checkpoint.resolve()), '--epochs', str(args.epochs),
               '--feature-mode', args.feature_mode, '--classifier-init', 'average',
               '--learning-rate', str(args.learning_rate), '--batch-size', str(args.batch_size),
               '--accumulate', str(args.accumulate), '--workers', str(args.workers),
               '--patience', str(args.patience), '--seed', str(args.seed), '--device', args.device,
               '--train-val-only', '--bitmap-masks']
    if args.overfit:
        command.append('--overfit')
    return command


def evaluation_command(args, dataset, output):
    command = [sys.executable, '-u', '-m', 'src.evaluate_nlr', '--dataset', str(dataset),
               '--run', str(output), '--device', args.device, '--val-only', '--hard-exit-after-success']
    if args.overfit:
        command += ['--checkpoint', str(output / 'final_model.pth'), '--train-diagnostic']
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--merged', type=Path, default=DEFAULT_MERGED, help='merged v2 folder with annotations/ and masks/')
    source.add_argument('--archives', type=Path, nargs=4, help='four exported part ZIPs; validate and merge into this run')
    parser.add_argument('--images', type=Path, default=IMAGES)
    parser.add_argument('--output', type=Path, help='new run directory; existing directories are never overwritten')
    parser.add_argument('--prepare-only', action='store_true', help='CPU data conversion and config validation; no download/GPU training')
    parser.add_argument('--epochs', type=int, help='default: 12; 100 with --overfit')
    parser.add_argument('--overfit', action='store_true', help='full model, no augmentation/weight decay/early stopping; select final epoch')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--feature-mode', choices=('head', 'neck', 'full'))
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args(argv)
    if args.overfit and args.feature_mode not in (None, 'full'):
        parser.error('--overfit requires --feature-mode full')
    args.feature_mode = args.feature_mode or ('full' if args.overfit else 'neck')
    args.epochs = args.epochs if args.epochs is not None else (100 if args.overfit else 12)
    if min(args.epochs, args.patience, args.batch_size, args.accumulate) < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error('epochs/patience/batch-size/accumulate/learning-rate must be positive; workers must be >=0')
    return args


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    args = parse_args()
    family = 'nlr_masking_v2_overfit' if args.overfit else 'nlr_masking_v2'
    output = (args.output or PROJECT_ROOT / 'outputs/fine_tune' / family /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new run directory; refusing to overwrite: {output}')
    sources = args.archives or [args.merged]
    for source in sources:
        if not source.exists():
            raise FileNotFoundError(f'Missing source: {source}; use --archives with the four exported ZIPs if the merged folder is not copied')
    output.mkdir(parents=True, exist_ok=False)
    merged = args.merged.resolve()
    if args.archives:
        merged = output / 'source'
        run_step([sys.executable, str(PROJECT_ROOT / 'masking/merge.py'),
                  *[str(path.resolve()) for path in args.archives], '--output', str(merged)], output / 'merge_console.log')
    print(f'Preparing completed annotations: {merged}', flush=True)
    report = prepare_dataset(merged, output / 'dataset', args.images.resolve(), seed=args.seed)
    dataset = output / 'dataset/coco'
    train = training_command(args, dataset, output)
    evaluate = evaluation_command(args, dataset, output)
    plan = {'source': str(merged), 'archives': [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                             for p in (args.archives or [])],
            'split_stats': report['stats'], 'snapshot_sha256': report['snapshot_sha256'],
            'train_command': train + ['--hard-exit-after-success'], 'evaluation_command': evaluate,
            'status': 'validating',
            'checkpoint_initialization': 'original five-class NLR checkpoint, averaged to one panel class',
            'overfit_diagnostic': args.overfit,
            'selection': ('final epoch; train diagnostics separate; confidence calibrated on validation; no test split'
                          if args.overfit else 'best validation mask mAP; confidence calibrated on validation only; no test split')}
    (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    validate = list(train)
    if args.overfit:
        # Overfit training refuses a directory already containing a generated config.
        validate[validate.index('--output') + 1] = str(output / 'preflight')
    run_step(validate + ['--validate-only'], output / 'validation_console.log')
    print(json.dumps(report['stats'], indent=2), flush=True)
    if args.prepare_only:
        plan['status'] = 'prepared'
        (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Prepared (no GPU training): {output}', flush=True)
        return
    try:
        plan['status'] = 'training'
        (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        if args.config == DEFAULT_CONFIG and args.checkpoint == DEFAULT_CHECKPOINT:
            run_step([sys.executable, '-u', '-m', 'src.download', 'nlr'], output / 'download_console.log')
        run_step(train + ['--hard-exit-after-success'], output / 'train_console.log')
        plan['status'] = 'evaluating'
        (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        run_step(evaluate, output / 'evaluation_console.log')
    except Exception as error:
        plan.update(failed_stage=plan['status'], status='failed', error=str(error))
        (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    plan['status'] = 'complete'
    (output / 'run_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    model_name = 'final_model.pth' if args.overfit else 'best_model.pth'
    print(f'Model: {output / model_name}\nValidation report: {output / "evaluation/report.json"}', flush=True)


if __name__ == '__main__':
    main()
