from pathlib import Path
import tempfile
import unittest

from src.evaluate_nlr import evaluation_dataset


class EvaluationDatasetTests(unittest.TestCase):
    def test_training_validation_override_and_explicit_argument(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            run = Path(tmp)
            config = run/'nlr_config.py'
            config.write_text("data_root = 'aihub/'\nval_dataloader['dataset']['data_root'] = 'bitgaram/'\n", encoding='utf-8')
            self.assertEqual(evaluation_dataset(run), Path('bitgaram').resolve())
            self.assertEqual(evaluation_dataset(run,Path('explicit')), Path('explicit').resolve())
            config.write_text("data_root = 'legacy/'\n", encoding='utf-8')
            self.assertEqual(evaluation_dataset(run), Path('legacy').resolve())


if __name__ == '__main__':
    unittest.main()
