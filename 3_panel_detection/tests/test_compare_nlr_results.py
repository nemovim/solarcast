import json
from pathlib import Path
import tempfile
import unittest

from src.compare_nlr_results import compare_runs, validation_identity


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = [('neck',self.root/'neck'),('full',self.root/'full')]
        for name, run in self.runs:
            val = run/'dataset/coco/val'
            (val/'images').mkdir(parents=True)
            (val/'images/p.jpeg').write_bytes(b'positive-image')
            (val/'images/n.jpeg').write_bytes(b'negative-image')
            data = {'images':[{'id':1,'file_name':'p.jpeg','width':8,'height':8},
                              {'id':2,'file_name':'n.jpeg','width':8,'height':8}],
                    'annotations':[{'id':1,'image_id':1,'category_id':1,'segmentation':[[1,1,5,1,5,5]],
                                    'bbox':[1,1,4,4],'area':8,'iscrowd':0}],
                    'categories':[{'id':1,'name':'photovoltaic-panel'}]}
            (val/'annotations.json').write_text(json.dumps(data),encoding='utf-8')
            def metrics(conf,fp):
                return dict(confidence=conf,tp=1,fp=fp,fn=0,precision=1/(1+fp),recall=1.,f0_5=.5,
                            empty_images=1,empty_images_with_fp=int(fp>0),empty_image_fp_rate=float(fp>0),
                            empty_fp_instances=fp)
            original = dict(selected_val=metrics(.05,10),val_curve=[metrics(.05,10),metrics(.15,6)],recall_target_met=True)
            finetuned = dict(selected_val=metrics(.1 if name=='neck' else .15,4 if name=='neck' else 1),
                             val_curve=[metrics(.05,5),metrics(.15,2 if name=='neck' else 1)],recall_target_met=True)
            (run/'evaluation').mkdir()
            (run/'evaluation/report.json').write_text(json.dumps(dict(original=original,finetuned=finetuned,
                selection_split='val',mask_iou=.5,trained_checkpoint=str(run/'best_model.pth'))),encoding='utf-8')

    def test_compares_original_once_and_distinguishes_selected_and_common_threshold(self):
        report = compare_runs(self.runs)
        self.assertEqual([r['model'] for r in report['selected_threshold']],['original','neck','full'])
        self.assertEqual([r['fp'] for r in report['selected_threshold']],[10,4,1])
        self.assertEqual([r['fp'] for r in report['common_threshold']],[6,2,1])
        self.assertTrue(all(r['confidence']==.15 for r in report['common_threshold']))
        self.assertTrue(report['baseline_consistent_across_reports'])
        self.assertEqual(report['dataset']['instances'],1)

    def test_rejects_different_image_content_or_labels(self):
        image = self.runs[1][1]/'dataset/coco/val/images/p.jpeg'
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'differ'):
            compare_runs(self.runs)
        image.write_bytes(b'positive-image')
        path = self.runs[1][1]/'dataset/coco/val/annotations.json'
        data = json.loads(path.read_text())
        data['annotations'][0]['segmentation'][0][0]=2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'differ'):
            compare_runs(self.runs)

    def test_identity_ignores_json_spacing_list_order_and_coco_ids(self):
        run = self.runs[1][1]
        path = run/'dataset/coco/val/annotations.json'
        data = json.loads(path.read_text())
        data['images'].reverse()
        for image in data['images']:
            image['id']+=10
        data['annotations'][0]['image_id']+=10
        data['annotations'][0]['id']=99
        path.write_text(json.dumps(data,indent=4))
        self.assertEqual(validation_identity(self.runs[0][1]),validation_identity(run))

    def test_rejects_stale_report_counts_and_missing_threshold(self):
        with self.assertRaisesRegex(ValueError,'not present'):
            compare_runs(self.runs,.23)
        path = self.runs[1][1]/'evaluation/report.json'
        report = json.loads(path.read_text())
        report['finetuned']['selected_val']['fn']=9
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'counts'):
            compare_runs(self.runs)

    def test_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            compare_runs([self.runs[0],self.runs[0]])


if __name__ == '__main__':
    unittest.main()
