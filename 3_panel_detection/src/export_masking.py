"""Export completed masking-v2 records to exact COCO RLE, stratified 80:20."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image

from src.train_nlr import CLASS_NAME, PROJECT_ROOT, REPO_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from masking.core import (  # noqa: E402
    ASSIGNMENTS, COMPLETE, IMAGES, SCHEMA, Store, atomic_json, mask_png, png_pixels,
)

DEFAULT_MERGED = REPO_ROOT / "data/bitgaram/annotations_merged/2026-10-27_batch-01"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratified_split(records, seed=2026):
    """Split each image-level positive/negative stratum separately, never instances."""
    result, counts = {}, {}
    for status, label in (("done", "panel"), ("empty", "empty")):
        group = sorted((r for r in records if r["status"] == status), key=lambda r: r["image_id"])
        if len(group) < 2:
            raise ValueError(f"Need at least two completed {label} images for an 80:20 split")
        random.Random(f"{seed}:{status}").shuffle(group)
        n_train = min(len(group) - 1, max(1, len(group) * 4 // 5))
        for index, record in enumerate(group):
            result[record["image_id"]] = "train" if index < n_train else "val"
        counts[label] = {"total": len(group), "train": n_train, "val": len(group) - n_train}
    return result, counts


def bitmap_from_png(data, width, height):
    raw = png_pixels(data, width, height)
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width + 1)[:, 1:] > 0


def encode_rle(bitmap):
    """COCO uncompressed RLE in column-major order, starting with a zero run."""
    flat = np.asarray(bitmap, dtype=np.uint8).ravel(order="F")
    boundaries = np.concatenate(([0], np.flatnonzero(flat[1:] != flat[:-1]) + 1, [len(flat)]))
    counts = np.diff(boundaries).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(bitmap.shape), "counts": counts}


def instance_annotation(points, width, height):
    bitmap = bitmap_from_png(mask_png(width, height, [points]), width, height)
    ys, xs = np.nonzero(bitmap)
    if not len(xs):
        raise ValueError("A reviewed polygon rasterizes to zero pixels; fix it before training")
    return {"segmentation": encode_rle(bitmap), "area": int(len(xs)),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            "category_id": 1, "iscrowd": 0}


def load_reviewed(merged, images=IMAGES, assignments=ASSIGNMENTS):
    merged, images = Path(merged).resolve(), Path(images).resolve()
    if not (merged / "annotations").is_dir():
        raise FileNotFoundError(f"Missing merged annotations: {merged / 'annotations'}")
    store = Store(assignments, images, merged)
    records, excluded, seen = [], Counter(), set()
    for path in sorted((merged / "annotations").glob("*.json")):
        record = read_json(path)
        key = record.get("image_id")
        if key != path.stem or key not in store.by_id or key in seen:
            raise ValueError(f"Unknown/duplicate image ID or filename mismatch: {path}")
        seen.add(key)
        item = store.by_id[key]
        for field, expected in (("schema", SCHEMA), ("assignment_sha256", store.assignment_hash),
                                ("part", item["part"]), ("width", item["width"]), ("height", item["height"])):
            if record.get(field) != expected:
                raise ValueError(f"{key}: incompatible {field}")
        store.validate(key, record)
        if record["status"] not in COMPLETE:
            excluded[record["status"]] += 1
            continue
        source, mask = images / item["file"], merged / "masks" / f"{key}.png"
        if not source.is_file() or not mask.is_file():
            raise FileNotFoundError(f"{key}: missing original image or reviewed mask")
        with Image.open(source) as image:
            if image.size != (item["width"], item["height"]):
                raise ValueError(f"{key}: original image dimensions changed")
            image.verify()
        expected = mask_png(record["width"], record["height"], record["polygons"])
        if png_pixels(mask.read_bytes(), record["width"], record["height"]) != png_pixels(expected, record["width"], record["height"]):
            raise ValueError(f"{key}: reviewed PNG does not match the saved polygons")
        records.append(record)
    if not records:
        raise ValueError("No done/empty images were found; predictions and unfinished records are never training labels")
    return store, records, dict(excluded)


def prepare_dataset(merged, output, images=IMAGES, assignments=ASSIGNMENTS, seed=2026):
    """Validate first, then create an immutable dataset snapshot; never overwrite."""
    merged, output, images = Path(merged).resolve(), Path(output).resolve(), Path(images).resolve()
    if output.exists():
        raise FileExistsError(f"Dataset output already exists; choose a new run: {output}")
    store, records, excluded = load_reviewed(merged, images, assignments)
    splits, strata = stratified_split(records, seed)
    coco = {split: {"images": [], "annotations": [], "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "solar"}]}
            for split in ("train", "val")}
    rows, digests, image_groups = [], {}, {}
    annotation_id = 1
    for image_id, record in enumerate(sorted(records, key=lambda r: r["image_id"]), 1):
        key, status = record["image_id"], record["status"]
        split, item = splits[key], store.by_id[key]
        source, mask = images / item["file"], merged / "masks" / f"{key}.png"
        image_sha, mask_sha = sha256(source), sha256(mask)
        image_groups.setdefault(image_sha, []).append((key, split))
        digests[key] = {"image": image_sha, "mask": mask_sha, "annotation": sha256(merged / 'annotations' / f'{key}.json')}
        coco[split]["images"].append({"id": image_id, "file_name": item["file"],
                                      "width": item["width"], "height": item["height"],
                                      "review_status": status, "source_part": record["part"]})
        for points in record["polygons"]:
            annotation = instance_annotation(points, item["width"], item["height"])
            coco[split]["annotations"].append({"id": annotation_id, "image_id": image_id,
                                               **annotation, "source_polygon": points})
            annotation_id += 1
        rows.append({"image_id": key, "split": split, "status": status, "part": record["part"],
                     "instances": len(record["polygons"]), "image_sha256": image_sha,
                     "mask_sha256": mask_sha, "annotation_sha256": digests[key]["annotation"]})
    duplicate_leaks = [group for group in image_groups.values() if len({split for _, split in group}) > 1]
    if duplicate_leaks:
        raise ValueError(f"Identical image content occurs in train and val: {duplicate_leaks[:3]}")
    snapshot_hash = hashlib.sha256(json.dumps({"splits": splits, "files": digests}, sort_keys=True).encode()).hexdigest()
    report = {"schema": 2, "source": str(merged), "assignment_sha256": store.assignment_hash,
              "seed": seed, "split_ratio": "80:20 per done/empty image stratum",
              "rounding": "train=floor(0.8*n), validation=remainder; each split has >=1 per stratum",
              "strata": strata, "excluded_status_counts": excluded, "completed_images": len(records),
              "snapshot_sha256": snapshot_hash, "test_split": None,
              "segmentation": "exact pixel-center rasterization, COCO uncompressed RLE, one instance per reviewed polygon",
              "stats": {split: {"images": len(payload["images"]), "instances": len(payload["annotations"]),
                                 "panel_images": sum(i["review_status"] == "done" for i in payload["images"]),
                                 "empty_images": sum(i["review_status"] == "empty" for i in payload["images"])}
                        for split, payload in coco.items()},
              "validation": {"id_overlap": 0, "identical_content_overlap": 0, "png_polygon_agreement": True}}
    output.mkdir(parents=True, exist_ok=False)
    for split, payload in coco.items():
        (output / "coco" / split / "images").mkdir(parents=True)
        (output / "coco" / split / "masks").mkdir()
        atomic_json(output / "coco" / split / "annotations.json", payload)
    (output / "reviewed_annotations").mkdir()
    for row in rows:
        key, split = row["image_id"], row["split"]
        shutil.copy2(images / store.by_id[key]["file"], output / "coco" / split / "images" / store.by_id[key]["file"])
        shutil.copy2(merged / "masks" / f"{key}.png", output / "coco" / split / "masks" / f"{key}.png")
        shutil.copy2(merged / "annotations" / f"{key}.json", output / "reviewed_annotations" / f"{key}.json")
    with (output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    atomic_json(output / "split_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = prepare_dataset(args.merged, args.output, args.images, seed=args.seed)
    print(json.dumps(report["stats"], indent=2))
    print(f"dataset={args.output.resolve() / 'coco'}")


if __name__ == "__main__":
    main()
