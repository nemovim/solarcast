"""CPU regressions for transfer, feature freezing and val-only selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src import compare_nlr
from src.train_nlr import (
    compatible_state_dict, config_text, freeze_features, validate_transfer_report, empty_safe_coco_metric,
    parse_args, select_trained_checkpoint,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        self.neck = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))


class Runner:
    def __init__(self):
        self.model = Model()
        self.hooks = []

    def register_hook(self, hook, priority):
        self.hooks.append(hook)


class TrainingTests(unittest.TestCase):
    def test_overfit_disables_regularization_and_preserves_heldout_splits(self):
        configs = []
        for overfit in (False, True):
            namespace = {}
            exec(config_text(Path('base.py'), Path('data'), Path('work'), Path('weights.pth'),
                             100, 1, 4, 0, 5e-5, 2026, overfit=overfit), namespace)
            configs.append(namespace)
        normal, diagnostic = configs
        self.assertEqual(diagnostic['custom_hooks'], [])
        self.assertEqual(diagnostic['optim_wrapper']['optimizer']['weight_decay'], 0)
        self.assertEqual(diagnostic['train_cfg']['max_epochs'], 100)
        self.assertNotIn('RandomFlip', [s['type'] for s in diagnostic['train_pipeline']])
        self.assertNotIn('YOLOXHSVRandomAug', [s['type'] for s in diagnostic['train_pipeline']])
        self.assertEqual(diagnostic['train_dataloader']['dataset']['pipeline'], diagnostic['train_pipeline'])
        for key in ('val_dataloader', 'test_dataloader', 'val_evaluator', 'test_evaluator'):
            self.assertEqual(normal[key], diagnostic[key])
        self.assertEqual(diagnostic['train_dataloader']['dataset']['ann_file'], 'train/annotations.json')
        with patch('sys.argv', ['train_nlr', '--overfit']):
            args = parse_args()
            self.assertEqual((args.epochs, args.feature_mode, args.classifier_init), (100, 'full', 'average'))
            self.assertTrue(args.skip_test)
        with patch('sys.argv', ['train_nlr']):
            args = parse_args()
            self.assertEqual((args.epochs, args.feature_mode), (12, 'head'))
        with self.assertRaises(ValueError):
            config_text(Path('base.py'), Path('data'), Path('work'), Path('weights.pth'),
                        100, 1, 4, 0, 5e-5, 2026, small_panel_crop=True, overfit=True)

    def test_overfit_uses_final_checkpoint_not_best_validation(self):
        parent = Path(__file__).resolve().parents[1] / 'outputs'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            best = root / 'best_epoch_5.pth'
            best.touch()
            with self.assertRaises(FileNotFoundError):
                select_trained_checkpoint(root, 100, best, True)
            final = root / 'epoch_100.pth'
            final.touch()
            self.assertEqual(select_trained_checkpoint(root, 100, best, True), final.resolve())
            self.assertEqual(select_trained_checkpoint(root, 100, best), best.resolve())

    def test_train_diagnostic_uses_validation_threshold(self):
        import numpy as np
        from src.evaluate_nlr import train_diagnostic_summary
        records = [{'scores': [.2], 'ious': np.array([[1.]])}]
        summary = train_diagnostic_summary(records, .25)
        self.assertEqual(summary['at_val_selected']['tp'], 0)
        self.assertEqual(summary['at_0_15']['tp'], 1)
        self.assertNotIn('selected_train', summary)

    def test_zero_detections_produce_metrics_without_hiding_errors(self):
        class Base:
            metrics = ['bbox', 'segm']
            metric_items = None
            format_only = False

            def compute_metrics(self, results):
                raise RuntimeError('real evaluation failure')

        metric = empty_safe_coco_metric(Base)()
        zeros = metric.compute_metrics([({}, {'bboxes': []})] * 17)
        self.assertEqual(len(zeros), 12)
        self.assertEqual(zeros['segm_mAP'], 0.0)
        with self.assertRaises(ValueError):
            metric.compute_metrics([])
        with self.assertRaises(RuntimeError):
            metric.compute_metrics([({}, {'bboxes': [[0, 0, 1, 1]]})])
        with self.assertRaises(KeyError):
            metric.compute_metrics([({}, {})])

    def test_freezing_preserves_weights_and_bn_statistics(self):
        for mode, frozen_names in [('head', ('backbone', 'neck')), ('neck', ('backbone',))]:
            runner = Runner()
            counts = freeze_features(runner, object, mode)
            self.assertGreater(counts['trainable_parameters'], 0)
            before = {k: v.clone() for k, v in runner.model.state_dict().items()}
            runner.model.train()
            for hook in runner.hooks:
                hook.before_train_iter(runner, 0)
            optim = torch.optim.SGD(runner.model.parameters(), lr=.1)
            runner.model(torch.randn(4, 2)).sum().backward()
            optim.step()
            after = runner.model.state_dict()
            for k in before:
                if k.split('.')[0] in frozen_names:
                    self.assertTrue(torch.equal(before[k], after[k]), k)
            self.assertFalse(torch.equal(before['head.weight'], after['head.weight']))
            if mode == 'neck':
                self.assertTrue(runner.model.neck.weight.requires_grad)

    def test_fresh_classifier_does_not_skip_features(self):
        source = {'backbone.weight': torch.ones(2, 2), 'neck.weight': torch.ones(2, 2),
                  'bbox_head.rtm_cls.weight': torch.arange(10.).reshape(5, 2)}
        target = {**source, 'bbox_head.rtm_cls.weight': torch.zeros(1, 2)}
        avg, _ = compatible_state_dict(source, target, 5)
        self.assertTrue(torch.equal(avg['bbox_head.rtm_cls.weight'], torch.tensor([[4., 5.]])))
        fresh, skipped = compatible_state_dict(source, target)
        self.assertEqual(set(fresh), {'backbone.weight', 'neck.weight'})
        self.assertEqual(set(skipped), {'bbox_head.rtm_cls.weight'})
        report = {'loaded_keys': list(fresh), 'missing_after_load': list(skipped)}
        validate_transfer_report(report, 'fresh')
        report['missing_after_load'].append('bbox_head.mask.weight')
        with self.assertRaises(RuntimeError):
            validate_transfer_report(report, 'fresh')

    def test_crop_only_changes_train_pipeline(self):
        cfgs = []
        for crop in (False, True):
            namespace = {}
            exec(config_text(Path('base.py'), Path('data'), Path('work'), Path('weights.pth'),
                             12, 1, 4, 0, 5e-5, 2026, crop, 3), namespace)
            cfgs.append(namespace)
        self.assertEqual(cfgs[0]['test_pipeline'], cfgs[1]['test_pipeline'])
        crop = [s for s in cfgs[1]['train_pipeline'] if s['type'] == 'RandomCrop'][0]
        self.assertTrue(crop['allow_negative_crop'])
        self.assertTrue(crop['recompute_bbox'])
        self.assertEqual(cfgs[0]['custom_hooks'][0]['patience'], 3)

    def test_selection_never_uses_test_and_rejects_unmet_recall(self):
        a = {'name': 'a', 'recall_target_met': False,
             'val': {'empty_images_with_fp': 0, 'f0_5': .99}, 'test': {'f0_5': 1}}
        self.assertIsNone(compare_nlr.choose_experiment([a]))
        b = {'name': 'b', 'recall_target_met': True,
             'val': {'empty_images_with_fp': 2, 'f0_5': .6}, 'test': {'f0_5': 0}}
        self.assertEqual(compare_nlr.choose_experiment([a, b]), 'b')

    def test_orchestration_holds_test_until_selection(self):
        temp_parent = Path(__file__).resolve().parents[1] / 'outputs'
        temp_parent.mkdir(exist_ok=True)
        for feasible in (False, True):
            with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
                root = Path(directory)
                root.resolve().relative_to(temp_parent.resolve())
                dataset = root / 'dataset'
                for split in ('train', 'val', 'test'):
                    (dataset / split).mkdir(parents=True)
                    (dataset / split / 'annotations.json').write_text('{}')
                output = root / 'compare'
                calls = []

                def fake_run(command, log):
                    calls.append(command)
                    if command[0] == 'src.train_nlr':
                        run = Path(command[command.index('--output') + 1])
                        (run / 'best_model.pth').touch()
                        (run / 'inference_config.py').touch()
                    if command[0] == 'src.evaluate_nlr':
                        run = Path(command[command.index('--run') + 1])
                        dest = run / 'evaluation'
                        dest.mkdir(exist_ok=True)
                        val = {'empty_images_with_fp': 1, 'f0_5': .5,
                               'recall': .8 if feasible else .4, 'precision': .6}
                        data = {'selected_val': val, 'recall_target_met': feasible,
                                'val_at_0_15': val, 'val_at_0_25': val}
                        (dest / 'report.json').write_text(json.dumps({'finetuned': data}))

                argv = ['compare_nlr', '--dataset', str(dataset), '--output', str(output)]
                with patch('sys.argv', argv), patch.object(compare_nlr, 'coco_split_stats', return_value={}), \
                        patch.object(compare_nlr, 'run_command', side_effect=fake_run):
                    compare_nlr.main()
                self.assertEqual(len(calls), 10 if feasible else 8)
                for cmd in calls[:8]:
                    self.assertIn('--skip-test' if cmd[0] == 'src.train_nlr' else '--val-only', cmd)
                if feasible:
                    self.assertIn('--test-only', calls[8])
                report = json.loads((output / 'comparison.json').read_text())
                self.assertEqual(report['status'], 'complete' if feasible else 'recall_target_not_met')
                if not feasible:
                    # Model the user's fourth-run crash, preserving the first three.
                    report['experiments'].pop()
                    report['status'] = 'running'
                    (output / 'comparison.json').write_text(json.dumps(report))
                    previous = output / 'neck_fresh_crop' / 'crash.log'
                    previous.write_text('keep this failed attempt')
                    calls.clear()
                    with patch('sys.argv', argv + ['--resume']), \
                            patch.object(compare_nlr, 'coco_split_stats', return_value={}), \
                            patch.object(compare_nlr, 'run_command', side_effect=fake_run):
                        compare_nlr.main()
                    self.assertEqual(len(calls), 2)
                    self.assertIn('neck_fresh_crop', calls[0][calls[0].index('--output') + 1])
                    archived = list(output.glob('neck_fresh_crop.failed_*/crash.log'))
                    self.assertEqual(len(archived), 1)
                    self.assertEqual(archived[0].read_text(), 'keep this failed attempt')
                    (dataset / 'train/annotations.json').write_text('{"changed": true}')
                    with patch('sys.argv', argv + ['--resume']), \
                            patch.object(compare_nlr, 'coco_split_stats', return_value={}), \
                            patch.object(compare_nlr, 'run_command') as execute:
                        with self.assertRaises(ValueError):
                            compare_nlr.main()
                        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
