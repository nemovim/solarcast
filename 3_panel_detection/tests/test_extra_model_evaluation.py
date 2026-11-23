import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluate_nlr import model_specs, write_comparison


class ExtraModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp_parent = Path(__file__).resolve().parents[2] / 'tmp'
        self.temp_parent.mkdir(exist_ok=True)

    def test_models_keep_their_own_configs_and_missing_weights_fail_early(self):
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as folder:
            root = Path(folder)
            for name in ('base.py', 'base.pth', 'inference_config.py', 'best_model.pth', 'old.py', 'final_model.pth'):
                (root / name).touch()
            extra = [('previous', str(root / 'old.py'), str(root / 'final_model.pth'))]
            with patch('src.evaluate_nlr.DEFAULT_CONFIG', root / 'base.py'), patch('src.evaluate_nlr.DEFAULT_MODEL', root / 'base.pth'):
                specs = model_specs(root, extra_models=extra)
                self.assertEqual(specs[-1], ('previous', root / 'old.py', root / 'final_model.pth'))
                with self.assertRaises(ValueError):
                    model_specs(root, extra_models=extra + extra)
                (root / 'final_model.pth').unlink()
                with self.assertRaises(FileNotFoundError):
                    model_specs(root, extra_models=extra)

    def test_selected_and_common_confidence_tables_include_all_models(self):
        names = ['original', 'finetuned', 'neck', 'previous']
        metrics = dict(confidence=.15, tp=3, fp=4, fn=1, precision=3/7,
                       recall=.75, empty_images_with_fp=2, empty_images=5)
        report = {name: {'selected_val': {**metrics, 'confidence': .3},
                         'val_at_0_15': metrics} for name in names}
        with tempfile.TemporaryDirectory(dir=self.temp_parent) as folder:
            output = Path(folder)
            write_comparison(output, report, names)
            for key, expected in [('selected_threshold', .3), ('common_threshold', .15)]:
                with (output / f'{key}.csv').open(encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row['model'] for row in rows], names)
                self.assertEqual({float(row['confidence']) for row in rows}, {expected})


if __name__ == '__main__':
    unittest.main()
