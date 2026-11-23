import argparse
from pathlib import Path
import unittest

from src.compare_aihub import build_plan


class ComparisonPlanTests(unittest.TestCase):
    def test_three_stages_have_separate_outputs_and_common_evaluation(self):
        args = argparse.Namespace(output=Path("outputs/compare"), config=Path("config.py"),
            checkpoint=Path("original.pth"), mixed_dataset=Path("mixed"), local_dataset=Path("local"),
            epochs=30, local_epochs=20, learning_rate=5e-5, local_learning_rate=1e-5,
            feature_mode="neck", patience=5, seed=2026, device="cuda:0", batch_size=1, accumulate=4)
        stages = build_plan(args)
        self.assertEqual([s["name"] for s in stages], ["mixed_only", "local_only", "mixed_then_local"])
        for stage in stages:
            cmd = stage["evaluate"]
            self.assertEqual(cmd[cmd.index("--dataset") + 1], str(Path("local").resolve()))
            self.assertIn("--val-only", cmd)
        cmd = stages[2]["train"]
        self.assertEqual(cmd[cmd.index("--classifier-init") + 1], "single")
        self.assertEqual(cmd[cmd.index("--checkpoint") + 1], str((args.output / "mixed_only/best_model.pth").resolve()))
        for stage in stages[:2]:
            cmd = stage["train"]
            self.assertEqual(cmd[cmd.index("--checkpoint") + 1], str(args.checkpoint.resolve()))


if __name__ == "__main__":
    unittest.main()
