"""Evaluate saved NLR experiments on one local validation set, without training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

from src.evaluate_nlr import evaluation_dataset
from src.finetune_masking import run_step
from src.train_nlr import PROJECT_ROOT, coco_split_stats
from src.compare_nlr_results import markdown_table


def discover(root, reference):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    models, skipped = [], []
    for config in sorted(root.rglob('inference_config.py')):
        run = config.parent
        if run == reference:
            continue
        setup_path = run/'training_setup.json'
        setup = json.loads(setup_path.read_text(encoding='utf-8')) if setup_path.is_file() else {}
        overfit = setup.get('overfit_diagnostic', 'overfit' in str(run.relative_to(root)))
        checkpoints = ['final_model.pth'] if overfit else ['best_model.pth']
        for filename in checkpoints:
            checkpoint = run/filename
            if not checkpoint.is_file():
                skipped.append(dict(run=str(run), reason=f'Missing {filename}'))
                continue
            name = '__'.join(run.relative_to(root).parts)
            models.append(dict(name=name, run=str(run), checkpoint=str(checkpoint),
                               feature_mode=setup.get('features',{}).get('mode','unknown'),
                               overfit=overfit))
    return models, skipped


def summarize(output, completed):
    selected, common, details = [], [], {}
    for name, report_path in completed:
        report = json.loads(Path(report_path).read_text(encoding='utf-8'))
        metrics = report['finetuned']
        details[name] = metrics
        selected.append(dict(model=name, **metrics['selected_val'],
                             recall_target_met=metrics.get('recall_target_met')))
        common.append(dict(model=name, **metrics['val_at_0_15']))
        if 'original' not in details:
            baseline = report['original']
            details['original'] = baseline
            selected.insert(0,dict(model='original', **baseline['selected_val'],
                                   recall_target_met=baseline.get('recall_target_met')))
            common.insert(0,dict(model='original', **baseline['val_at_0_15']))
    if not selected:
        return
    for filename, rows in [('selected_threshold.csv',selected),('common_threshold.csv',common)]:
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'comparison.md').write_text(
        '# Saved NLR model comparison\n\nSame validation dataset; mask IoU=0.5. '
        'Thresholds selected on validation. Not independent test performance. '
        'Older experiments may have trained on these images; verify train overlap before interpreting generalization.\n\n'
        '## Selected confidence\n\n'+markdown_table(selected)+'\n\n## Common confidence 0.15\n\n'+markdown_table(common)+'\n',encoding='utf-8')
    (output/'comparison.json').write_text(json.dumps(details,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=PROJECT_ROOT/'outputs/aihub_train/run01')
    parser.add_argument('--search-root',type=Path,default=PROJECT_ROOT/'outputs/fine_tune')
    parser.add_argument('--dataset',type=Path,help='Defaults to validation saved in --run/nlr_config.py')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'outputs/comparisons'/('all_nlr_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    dataset=evaluation_dataset(args.run,args.dataset)
    coco_split_stats(dataset,('val',))
    if not args.search_root.is_dir():
        raise FileNotFoundError(args.search_root)
    models,skipped=discover(args.search_root,args.run)
    if not (args.run/'best_model.pth').is_file() or not (args.run/'inference_config.py').is_file():
        raise FileNotFoundError(f'Missing reference best_model.pth/inference_config.py: {args.run}')
    models.insert(0,dict(name='aihub_current',run=str(args.run.resolve()),
                         checkpoint=str((args.run/'best_model.pth').resolve())))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    plan=dict(dataset=str(dataset),models=models,skipped=skipped,status='prepared',failed=[])
    def save(): (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    save()
    for model in models: print(f"{model['name']}: {model['checkpoint']}",flush=True)
    if args.prepare_only:
        return
    completed=[]
    for index,model in enumerate(models):
        directory=output/f'{index:02d}_{model["name"]}'
        command=[sys.executable,'-u','-m','src.evaluate_nlr','--run',model['run'],
                 '--checkpoint',model['checkpoint'],'--dataset',str(dataset),'--val-only',
                 '--output',str(directory),'--device',args.device,'--hard-exit-after-success']
        plan.update(status='running',current=model['name']);save()
        try:
            run_step(command,output/f'{index:02d}_evaluation.log')
            completed.append((model['name'],directory/'report.json'))
            summarize(output,completed)
        except Exception as exc:
            plan['failed'].append(dict(model=model['name'],error=str(exc)))
        save()
    plan['status']='complete' if not plan['failed'] else 'completed_with_failures'
    save()
    print(f"Results: {output}; successful={len(completed)}, failed={len(plan['failed'])}",flush=True)
    if plan['failed']:
        raise SystemExit(1)


if __name__=='__main__': main()
