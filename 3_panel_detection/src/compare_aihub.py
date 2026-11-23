"""Run mixed-only, local-only and mixed-to-local NLR experiments sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import hashlib

from PIL import Image

from src.finetune_masking import run_step
from src.train_nlr import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, PROJECT_ROOT, coco_split_stats


def check_evaluation_leakage(mixed, local):
    def fingerprints(dataset, split):
        payload = json.loads((dataset / split / "annotations.json").read_text(encoding="utf-8"))
        result = set()
        for item in payload["images"]:
            with Image.open(dataset / split / "images" / item["file_name"]) as image:
                rgb = image.convert("RGB")
                result.add(hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest())
        return result
    if fingerprints(mixed, "train") & fingerprints(local, "val"):
        raise ValueError("Local validation images occur in mixed training data")


def build_plan(args):
    root = args.output.resolve()
    common = ["--config", str(args.config.resolve()), "--train-val-only", "--bitmap-masks",
              "--feature-mode", args.feature_mode, "--patience", str(args.patience),
              "--seed", str(args.seed), "--device", args.device,
              "--batch-size", str(args.batch_size), "--accumulate", str(args.accumulate)]
    stages = []
    for name, dataset, checkpoint, initialization, epochs, rate in [
        ("mixed_only", args.mixed_dataset, args.checkpoint, "average", args.epochs, args.learning_rate),
        ("local_only", args.local_dataset, args.checkpoint, "average", args.epochs, args.learning_rate),
        ("mixed_then_local", args.local_dataset, root / "mixed_only/best_model.pth",
         "single", args.local_epochs, args.local_learning_rate),
    ]:
        run = root / name
        train = [sys.executable, "-u", "-m", "src.train_nlr", "--dataset", str(dataset.resolve()),
                 "--output", str(run), "--checkpoint", str(checkpoint.resolve()),
                 "--classifier-init", initialization, "--epochs", str(epochs),
                 "--learning-rate", str(rate), *common, "--hard-exit-after-success"]
        evaluate = [sys.executable, "-u", "-m", "src.evaluate_nlr", "--dataset",
                    str(args.local_dataset.resolve()), "--run", str(run), "--val-only",
                    "--device", args.device, "--hard-exit-after-success"]
        stages.append(dict(name=name, train=train, evaluate=evaluate))
    return stages


def summarize(output, stages):
    results = {}
    rows = []
    for stage in stages:
        report = json.loads((output / stage["name"] / "evaluation/report.json").read_text(encoding="utf-8"))
        if not results:
            results["original"] = report["original"]
        results[stage["name"]] = report["finetuned"]
    import csv
    for name, metrics in results.items():
        rows.append(dict(model=name, **metrics["selected_val"],
                         recall_target_met=metrics["recall_target_met"]))
    summary = dict(models=results, note="All models evaluated on the same local validation; thresholds selected on validation, not independent test performance")
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, data in [("comparison.csv", rows),
                           ("comparison_conf_0p15.csv", [dict(model=n, **m["val_at_0_15"]) for n, m in results.items()])]:
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-dataset", type=Path, required=True)
    parser.add_argument("--local-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/aihub_compare" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prepare-only", action="store_true", help="validate datasets and save commands, no GPU or downloads")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--local-learning-rate", type=float, default=1e-5)
    parser.add_argument("--feature-mode", choices=("head", "neck", "full"), default="neck")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.epochs, args.local_epochs, args.patience, args.batch_size, args.accumulate) < 1 or min(args.learning_rate, args.local_learning_rate) <= 0:
        parser.error("Epochs, patience, batch, accumulation and learning rates must be positive")
    stats = {name: coco_split_stats(path.resolve(), ("train", "val"))
             for name, path in [("mixed", args.mixed_dataset), ("local", args.local_dataset)]}
    check_evaluation_leakage(args.mixed_dataset.resolve(), args.local_dataset.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = build_plan(args)
    plan = dict(status="prepared", datasets=stats, stages=stages,
                evaluation_dataset=str(args.local_dataset.resolve()))
    plan_path = output / "run_plan.json"

    def save():
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    save()
    if args.prepare_only:
        print(f"Prepared commands: {plan_path}")
        return
    try:
        for stage in stages:
            for action in ("train", "evaluate"):
                plan.update(status="running", current_stage=stage["name"], action=action)
                save()
                run_step(stage[action], output / f'{stage["name"]}_{action}.log')
        summarize(output, stages)
        plan["status"] = "complete"
        save()
    except Exception as exc:
        plan.update(status="failed", error=str(exc))
        save()
        raise
    print(f"Comparison: {output / 'comparison.csv'}")


if __name__ == "__main__":
    main()
