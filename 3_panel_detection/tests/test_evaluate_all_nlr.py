from pathlib import Path
import tempfile
import unittest

from src.evaluate_all_nlr import discover


class DiscoveryTests(unittest.TestCase):
    def test_uses_final_for_overfit_and_best_for_regular(self):
        parent=Path(__file__).resolve().parents[2]/'tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp)
            for family,checkpoint in [('nlr_masking_v2','best_model.pth'),('nlr_masking_v2_overfit','final_model.pth')]:
                run=root/family/'run';run.mkdir(parents=True)
                (run/'inference_config.py').write_text('')
                (run/checkpoint).write_bytes(b'checkpoint')
            missing=root/'missing';missing.mkdir();(missing/'inference_config.py').write_text('')
            models,skipped=discover(root,root/'reference')
            self.assertEqual(len(models),2)
            self.assertEqual(len(skipped),1)
            self.assertTrue(next(m for m in models if m['overfit'])['checkpoint'].endswith('final_model.pth'))


if __name__=='__main__': unittest.main()
