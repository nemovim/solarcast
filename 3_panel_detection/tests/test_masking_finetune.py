"""CPU regressions for the new reviewed-v2 train/validation pipeline."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.export_masking import (
    prepare_dataset, stratified_split, instance_annotation, bitmap_from_png,
)
from src.finetune_masking import parse_args as masking_args, training_command, evaluation_command, run_step
from src.train_nlr import coco_split_stats, config_text, parse_args as training_args, inference_dataset_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from masking.core import Store, mask_png


class MaskingTrainingTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.merged = self.root / 'merged'
        self.images.mkdir()
        (self.merged / 'annotations').mkdir(parents=True)
        (self.merged / 'masks').mkdir()
        self.assignment = self.root / 'assignments.json'
        items = [{'id': f'{i+1}_1', 'file': f'{i+1}_1.jpeg', 'part': i % 4 + 1,
                  'width': 8, 'height': 8} for i in range(23)]
        self.assignment.write_text(json.dumps({'schema': 2, 'images': items}), encoding='utf-8')
        self.store = Store(self.assignment, self.images, self.root / 'work')
        self.records = []
        for i, item in enumerate(items):
            key = item['id']
            status = 'done' if i < 10 else 'empty' if i < 20 else ['pending','deferred','in_progress'][i-20]
            record = self.store.blank(key)
            record.update(status=status, polygons=[[[1,1],[6,1],[6,5],[1,5]]] if status == 'done' else [])
            self.records.append(record)
            (self.merged / 'annotations' / f'{key}.json').write_text(json.dumps(record), 'utf-8')
            Image.new('RGB', (8,8), (i*9, i*7, i*5)).save(self.images / item['file'])
            if status in {'done', 'empty'}:
                (self.merged / 'masks' / f'{key}.png').write_bytes(mask_png(8,8,record['polygons']))

    def prepare(self, destination='dataset'):
        return prepare_dataset(self.merged, self.root / destination, self.images, self.assignment)

    def test_stratification_is_per_class_deterministic_and_order_independent(self):
        split, stats = stratified_split(self.records)
        self.assertEqual(stats['panel'], {'total':10,'train':8,'val':2})
        self.assertEqual(stats['empty'], {'total':10,'train':8,'val':2})
        self.assertEqual(split, stratified_split(list(reversed(self.records)))[0])
        self.assertNotEqual(split, stratified_split(self.records, 3)[0])
        self.assertEqual(len(split),20)
        self.assertEqual(len(set(split)),20)

    def test_requested_124_positive_680_empty_rounding(self):
        records = [{'image_id': f'p{i}', 'status':'done'} for i in range(124)]
        records += [{'image_id': f'n{i}', 'status':'empty'} for i in range(680)]
        _, stats = stratified_split(records)
        self.assertEqual((stats['panel']['train'],stats['panel']['val']),(99,25))
        self.assertEqual((stats['empty']['train'],stats['empty']['val']),(544,136))

    def test_export_has_only_reviewed_two_splits_and_keeps_empty_images(self):
        report = self.prepare()
        self.assertEqual(report['completed_images'],20)
        self.assertEqual(report['excluded_status_counts'],{'pending':1,'deferred':1,'in_progress':1})
        self.assertFalse((self.root/'dataset/coco/test').exists())
        stats = coco_split_stats(self.root/'dataset/coco',('train','val'))
        self.assertEqual(stats['train'],{'images':16,'instances':8,'empty_images':8})
        self.assertEqual(stats['val'],{'images':4,'instances':2,'empty_images':2})
        with self.assertRaises(FileNotFoundError):
            coco_split_stats(self.root/'dataset/coco')
        self.assertTrue((self.root/'dataset/split_manifest.csv').is_file())
        self.assertEqual(len(list((self.root/'dataset/reviewed_annotations').glob('*.json'))),20)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_rle_decodes_to_exact_polygon_mask_without_small_object_filter(self):
        for points in ([[1,1],[6,1],[6,5],[1,5]], [[2,2],[3,2],[3,3],[2,3]],
                       [[0,0],[8,0],[0,6]], [[1.2,1.1],[6.6,2.4],[5.8,6.7],[.7,5.3]]):
            annotation=instance_annotation(points,8,8)
            rle=annotation['segmentation']
            flat=np.concatenate([np.full(count,index%2,dtype=np.uint8) for index,count in enumerate(rle['counts'])])
            decoded=flat.reshape(rle['size'],order='F')>0
            truth=bitmap_from_png(mask_png(8,8,[points]),8,8)
            np.testing.assert_array_equal(decoded,truth)
            self.assertEqual(annotation['area'],int(truth.sum()))
        self.assertEqual(instance_annotation([[2,2],[3,2],[3,3],[2,3]],8,8)['area'],1)

    def test_corrupted_mask_is_rejected_before_output_creation(self):
        (self.merged/'masks/1_1.png').write_bytes(mask_png(8,8,[]))
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.prepare()
        self.assertFalse((self.root/'dataset').exists())

    def test_duplicate_content_in_train_and_val_is_rejected(self):
        splits,_=stratified_split(self.records)
        a=next(r['image_id'] for r in self.records if r['status']=='done' and splits[r['image_id']]=='train')
        b=next(r['image_id'] for r in self.records if r['status']=='done' and splits[r['image_id']]=='val')
        (self.images/f'{b}.jpeg').write_bytes((self.images/f'{a}.jpeg').read_bytes())
        with self.assertRaisesRegex(ValueError,'Identical image content'):
            self.prepare()
        self.assertFalse((self.root/'dataset').exists())

    def test_new_config_uses_bitmap_masks_no_test_and_keeps_negatives(self):
        namespace={}
        exec(config_text(Path('base.py'),self.root/'data',self.root/'run',Path('weights.pth'),
                         12,1,4,0,5e-5,2026,train_val_only=True,bitmap_masks=True),namespace)
        self.assertIsNone(namespace['test_dataloader'])
        self.assertIsNone(namespace['test_evaluator'])
        self.assertIsNone(namespace['test_cfg'])
        for pipeline in ('train_pipeline','test_pipeline'):
            loader=next(s for s in namespace[pipeline] if s['type']=='LoadAnnotations')
            self.assertTrue(loader['poly2mask'])
        self.assertFalse(namespace['train_dataloader']['dataset']['filter_cfg']['filter_empty_gt'])
        self.assertEqual(namespace['default_hooks']['checkpoint']['save_best'],'coco/segm_mAP')
        legacy={}
        exec(config_text(Path('base.py'),self.root/'data',self.root/'run',Path('weights.pth'),
                         12,1,4,0,5e-5,2026),legacy)
        self.assertIsInstance(legacy['test_dataloader'],dict)
        self.assertFalse(legacy['train_pipeline'][1]['poly2mask'])

    def test_orchestrator_passes_train_val_only_and_expected_training_parameters(self):
        args=masking_args(['--prepare-only'])
        self.assertEqual(args.feature_mode,'neck')
        self.assertEqual((args.epochs,args.patience),(12,3))
        command=training_command(args,self.root/'dataset/coco',self.root/'run')
        self.assertIn('--train-val-only',command)
        self.assertIn('--bitmap-masks',command)
        self.assertNotIn('--overfit',command)
        self.assertNotIn('test',command)
        from unittest.mock import patch
        with patch('sys.argv',['train_nlr','--train-val-only']):
            self.assertTrue(training_args().skip_test)

    def test_overfit_full_and_final_checkpoint(self):
        args = masking_args(['--overfit'])
        self.assertEqual((args.feature_mode, args.epochs), ('full', 100))
        self.assertIn('--overfit', training_command(args, self.root/'data', self.root/'run'))
        evaluate = evaluation_command(args, self.root/'data', self.root/'run')
        self.assertEqual(evaluate[evaluate.index('--checkpoint')+1], str(self.root/'run/final_model.pth'))
        self.assertIn('--train-diagnostic', evaluate)
        self.assertIn('--val-only', evaluate)
        self.assertEqual(masking_args(['--overfit','--epochs','20']).epochs, 20)
        namespace = {}
        exec(config_text(Path('base.py'), self.root/'data', self.root/'run', Path('weights.pth'),
                         100, 1, 4, 0, 5e-5, 2026, overfit=True,
                         train_val_only=True, bitmap_masks=True), namespace)
        self.assertEqual(namespace['custom_hooks'], [])
        checkpoint = namespace['default_hooks']['checkpoint']
        self.assertEqual(checkpoint['interval'], 10)
        self.assertEqual(checkpoint['max_keep_ckpts'], -1)
        self.assertIsNone(checkpoint['save_best'])
        self.assertTrue(checkpoint['save_last'])
        self.assertEqual(namespace['optim_wrapper']['optimizer']['weight_decay'], 0)
        types = {step['type'] for step in namespace['train_pipeline']}
        self.assertTrue(types.isdisjoint({'RandomFlip','RandomCrop','YOLOXHSVRandomAug'}))
        self.assertFalse(namespace['train_dataloader']['dataset']['filter_cfg']['filter_empty_gt'])

    def test_subprocess_logs_utf8_independently_of_windows_console_codepage(self):
        from contextlib import redirect_stdout
        log = self.root / 'unicode.log'
        with redirect_stdout(io.StringIO()):
            run_step([sys.executable, '-c', "print('한글 로그 \\u2713')"], log)
        self.assertEqual(log.read_text('utf-8').strip(), '한글 로그 \u2713')

    def test_inference_dataset_has_registry_type_and_needs_no_annotation_file(self):
        pipeline = [{'type':'LoadImageFromFile'}, {'type':'LoadAnnotations', 'with_mask':True},
                    {'type':'Resize','scale':(640,640)}]
        metadata = {'classes':('photovoltaic-panel',)}
        dataset = inference_dataset_config(pipeline,metadata)
        self.assertEqual(dataset['type'],'CocoDataset')
        self.assertTrue(dataset['lazy_init'])
        self.assertTrue(dataset['test_mode'])
        self.assertEqual(dataset['metainfo'],metadata)
        self.assertNotIn('ann_file',dataset)
        self.assertEqual([step['type'] for step in dataset['pipeline']],['LoadImageFromFile','Resize'])
        dataset['pipeline'][1]['scale']=(320,320)
        self.assertEqual(pipeline[2]['scale'],(640,640))


if __name__ == '__main__':
    unittest.main()
