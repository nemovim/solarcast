"""Convert explicitly split AI Hub aerial polygons and local COCO into NLR COCO.

Manifest columns: image,label,split,group. Paths are relative to the CSV.
Only training/validation Solar panel polygon labels are accepted; no SAR boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from contextlib import nullcontext

from PIL import Image
import numpy as np

from src.export_masking import instance_annotation, encode_rle
from src.train_nlr import CLASS_NAME, PROJECT_ROOT, coco_split_stats


def folder_manifest(root, destination):
    """Pair train_data/train_label by stem, including nested extraction folders."""
    root = Path(root).resolve()

    def index(directory, suffixes):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")
        found = {}
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.stem.lower().endswith('_meta'):
                continue
            key = path.stem.casefold()
            if key in found:
                raise ValueError(f"Duplicate filename stem: {found[key]} and {path}")
            found[key] = path
        if not found:
            raise ValueError(f"No matching files in {directory}")
        return found

    images = index(root / 'train_data', {'.tif', '.tiff'})
    labels = index(root / 'train_label', {'.json'})
    missing = sorted(images.keys() - labels.keys())
    extra = sorted(labels.keys() - images.keys())
    if missing or extra:
        raise ValueError(f"Image/label mismatch: missing labels={len(missing)} {missing[:5]}, missing images={len(extra)} {extra[:5]}")
    with Path(destination).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'label', 'split', 'group'])
        writer.writeheader()
        for key in sorted(images):
            # No external validation split is created in folder mode.
            writer.writerow(dict(image=str(images[key]), label=str(labels[key]),
                                 split='train', group='aihub_official_training'))
    print(f"Matched {len(images)} training image/label pairs; valid_* and train_meta are unused.", flush=True)
    return len(images)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class AmbiguousTiffPages(ValueError):
    pass


def load_source_rgb(path, frame_records=None):
    """Require one original page; reduced overview pages do not count."""
    with Image.open(path) as image:
        frames = []
        for index in range(getattr(image, 'n_frames', 1)):
            image.seek(index)
            reduced = bool(getattr(image, 'tag_v2', {}).get(254, 0) & 1)
            frames.append((index, image.size, reduced))
        originals = [frame for frame in frames if not frame[2]]
        ambiguous = len(originals) != 1
        index = None if ambiguous else originals[0][0]
        if frame_records is not None and len(frames) > 1:
            frame_records.append(dict(source=str(path), selected_frame=index,
                                      skipped=ambiguous,
                                      frames=[dict(index=i, size=s, reduced=r) for i,s,r in frames]))
        if ambiguous:
            raise AmbiguousTiffPages(f'Cannot associate labels with {len(originals)} original pages: {path}')
        image.seek(index)
        if image.mode not in {'RGB', 'RGBA', 'L'}:
            raise ValueError(f'Unsupported raster mode {image.mode}: {path}; convert explicitly')
        return image.convert('RGB')


def panel_annotations(payload, width, height):
    info = payload.get("Learning Data Info")
    if not isinstance(info, dict) or not isinstance(info.get("objects"), list):
        raise ValueError("Expected AI Hub Learning Data Info.objects array")
    objects = info["objects"]
    if not objects:
        raise ValueError("External empty labels need manual review; use reviewed local negatives")
    result = []
    for obj in objects:
        if str(obj.get("class_name", "")).strip().lower() != "solar panel":
            raise ValueError("Expected Solar panel labels, not roof/top/structure")
        if obj.get("annotation_type") != "polygon":
            raise ValueError("Expected aerial polygon labels; SAR boxes are unsupported")
        points = obj.get("points", [])
        if points and isinstance(points[0], (list, tuple)):
            pairs = points
        else:
            if len(points) % 2:
                raise ValueError("Odd polygon coordinate count")
            pairs = list(zip(points[::2], points[1::2]))
        if len(pairs) < 3:
            raise ValueError("Polygon needs at least three vertices")
        if any(len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) for v in p) for p in pairs):
            raise ValueError("Invalid polygon coordinates")
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in pairs):
            raise ValueError("Polygon outside image; check source dimensions")
        result.append(instance_annotation(pairs, width, height))
    return result


def crop_annotations(annotations, box):
    """Clip exact rasterized instance masks; preserve even one-pixel fragments."""
    left, top, right, bottom = box
    result = []
    for annotation in annotations:
        rle = annotation["segmentation"]
        flat = np.zeros(math.prod(rle["size"]), dtype=np.uint8)
        offset = 0
        for index, count in enumerate(rle["counts"]):
            flat[offset:offset + count] = index % 2
            offset += count
        bitmap = flat.reshape(rle["size"], order="F")[top:bottom, left:right]
        ys, xs = np.nonzero(bitmap)
        if not len(xs):
            continue
        result.append(dict(segmentation=encode_rle(bitmap), area=int(len(xs)),
                           bbox=[int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)],
                           category_id=1, iscrowd=0))
    return result


def prepare(manifest, local_dataset, output, tile_size=256, train_source="mixed"):
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if train_source not in {"mixed", "aihub"}:
        raise ValueError("train_source must be mixed or aihub")
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    local_dataset = Path(local_dataset).resolve() if local_dataset is not None else None
    if local_dataset is None and train_source == 'mixed':
        raise ValueError('mixed requires --local-dataset')
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "label", "split", "group"} <= set(reader.fieldnames or []):
            raise ValueError("CSV needs image,label,split,group columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    groups, seen_paths = {}, set()
    for row in rows:
        split, group = row["split"].strip(), row["group"].strip()
        if split not in {"train", "val"} or not group:
            raise ValueError("Each row needs train/val and a nonempty geographic group")
        row["split"] = split
        if group in groups and groups[group] != split:
            raise ValueError(f"Geographic group leaks across splits: {group}")
        groups[group] = split
        image = (manifest.parent / row["image"]).resolve()
        if image in seen_paths:
            raise ValueError(f"Repeated image: {image}")
        seen_paths.add(image)
    if local_dataset is not None:
        coco_split_stats(local_dataset, ("val",) if train_source == "aihub" else ("train", "val"))
    # A failed conversion retains its partial folder for inspection; never reuse it.
    output.mkdir(parents=True)
    datasets = {s: {"images": [], "annotations": [],
                    "categories": [{"id": 1, "name": CLASS_NAME}]} for s in (("train", "val") if local_dataset else ("train",))}
    seen_content = {}
    provenance = []
    duplicates = []
    frame_records = []
    skipped_sources = []

    def add(split, source, annotations, origin, crop_box=None, group=None, prepared_image=None):
        with (Image.open(source) if prepared_image is None else nullcontext(prepared_image)) as image:
            if image.mode not in {"RGB", "RGBA", "L"}:
                raise ValueError(f"Unsupported raster mode {image.mode}: {source}; convert explicitly")
            rgb = image.convert("RGB")
            if crop_box is not None and prepared_image is None:
                rgb = rgb.crop(crop_box)
            width, height = rgb.size
            if crop_box is not None and rgb.size != (tile_size, tile_size):
                raise ValueError(f'Unexpected crop size: {source}, {crop_box}, {rgb.size}')
            digest = hashlib.sha256(f"{width},{height}:".encode() + rgb.tobytes()).hexdigest()
            labels = sorted(json.dumps({key: ann[key] for key in
                             ('segmentation', 'bbox', 'area', 'category_id', 'iscrowd')},
                             sort_keys=True) for ann in annotations)
            if digest in seen_content:
                previous = seen_content[digest]
                retained = output / previous['split'] / 'images' / previous['file_name']
                with Image.open(retained) as saved:
                    equal_pixels = saved.size == rgb.size and saved.convert('RGB').tobytes() == rgb.tobytes()
                if not equal_pixels:
                    raise RuntimeError(f'Pixel hash matched but saved pixels differ: {source}; {retained}')
                duplicates.append(dict(source=str(source), crop_box=crop_box,
                                       retained_file=previous['file_name'], retained_source=previous['source'],
                                       retained_split=previous['split'], skipped_split=split,
                                       labels_differ=previous['labels'] != labels, pixel_sha256=digest))
                return
            ds = datasets[split]
            image_id = len(ds["images"]) + 1
            filename = f"{origin}_{split}_{image_id:07d}.png"
            seen_content[digest] = dict(source=str(source), split=split, labels=labels, file_name=filename,
                                        origin=origin, crop_box=crop_box, image_id=image_id)
            destination = output / split / "images" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination)
        ds["images"].append(dict(id=image_id, file_name=filename, width=width, height=height))
        for annotation in annotations:
            ds["annotations"].append({**annotation, "id": len(ds["annotations"]) + 1,
                                       "image_id": image_id, "category_id": 1})
        provenance.append(dict(split=split, file_name=filename, source=str(source),
                               pixel_sha256=digest, origin=origin, crop_box=crop_box, group=group))

    for split in datasets:
        if local_dataset is None:
            continue
        if split == "train" and train_source == "aihub":
            continue
        local = read_json(local_dataset / split / "annotations.json")
        by_image = {}
        for ann in local["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for image in local["images"]:
            source = (local_dataset / split / "images" / image["file_name"]).resolve()
            with Image.open(source) as raster:
                if raster.size != (image["width"], image["height"]):
                    raise ValueError(f"Local COCO size mismatch: {source}")
            add(split, source, by_image.get(image["id"], []), "local")
    for row_index, row in enumerate(rows):
        # External validation remains held out; do not silently promote it to train.
        if row["split"] != "train":
            continue
        source = (manifest.parent / row["image"]).resolve()
        label = (manifest.parent / row["label"]).resolve()
        try:
            source_rgb = load_source_rgb(source, frame_records)
        except AmbiguousTiffPages as exc:
            skipped_sources.append(dict(source=str(source), label=str(label), reason=str(exc)))
            print(f'Skipping ambiguous TIFF: {source.name}', flush=True)
            continue
        width, height = source_rgb.size
        if width % tile_size or height % tile_size:
            raise ValueError(f"External image dimensions must be multiples of {tile_size}: {source}")
        payload = read_json(label)
        declared_name = payload.get('Learning Data Info', {}).get('file_name')
        if declared_name and Path(declared_name).stem.casefold() != source.stem.casefold():
            raise ValueError(f'Label file_name does not match image: {label}: {declared_name} vs {source.name}')
        annotations = panel_annotations(payload, width, height)
        cropped_area = 0
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                box = (left, top, left + tile_size, top + tile_size)
                tile_annotations = crop_annotations(annotations, box)
                cropped_area += sum(ann['area'] for ann in tile_annotations)
                add(row["split"], source, tile_annotations, "aihub",
                    crop_box=box, group=row["group"].strip(), prepared_image=source_rgb.crop(box))
        if cropped_area != sum(ann['area'] for ann in annotations):
            raise RuntimeError(f'Mask area changed during tiling: {source}')
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(rows):
            print(f"Converted {row_index + 1}/{len(rows)} external source images", flush=True)
    for split, ds in datasets.items():
        (output / split / "annotations.json").write_text(json.dumps(ds), encoding="utf-8")
    stats = coco_split_stats(output, tuple(datasets))
    report = dict(stats=stats, geographic_groups=groups, images=provenance, tile_size=tile_size,
                  duplicate_tiles_skipped=len(duplicates), duplicate_tiles=duplicates,
                  duplicate_policy='keep_first', converter_version=3,
                  multi_frame_sources=frame_records,
                  skipped_ambiguous_tiffs=skipped_sources,
                  skipped_ambiguous_tiff_count=len(skipped_sources),
                  train_source=train_source, validation_source="local_only" if local_dataset else None,
                  excluded_external_val_images=sum(row["split"] == "val" for row in rows),
                  note="Local split preserved; geographic separation of local tiles must be reviewed separately")
    (output / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(manifest, output / "source_manifest.csv")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--aihub-root", type=Path, help="Root containing train_data/ and train_label/; no manual CSV needed")
    parser.add_argument("--local-dataset", type=Path, help="Optional legacy merge; omit to convert AI Hub independently")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256, help="External crop size; 512px source becomes four 256px tiles")
    parser.add_argument("--train-source", choices=("mixed", "aihub"), default="aihub",
                        help="mixed: AI Hub + local train; aihub: external train only. Validation is always local.")
    args = parser.parse_args()
    if args.aihub_root:
        if args.output.exists():
            raise FileExistsError(f"Use a new output directory: {args.output}")
        if args.local_dataset is not None:
            coco_split_stats(args.local_dataset.resolve(), ('val',) if args.train_source == 'aihub' else ('train', 'val'))
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='aihub_manifest_', dir=args.output.resolve().parent) as temp:
            manifest = Path(temp) / 'manifest.csv'
            folder_manifest(args.aihub_root, manifest)
            report = prepare(manifest, args.local_dataset, args.output, args.tile_size, args.train_source)
    else:
        report = prepare(args.manifest, args.local_dataset, args.output, args.tile_size, args.train_source)
    print(json.dumps(report['stats'], indent=2))


if __name__ == "__main__":
    main()
