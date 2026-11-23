"""CPU coverage for external labels, split leakage and one-class transfer."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.prepare_aihub import panel_annotations, prepare, crop_annotations, folder_manifest, load_source_rgb, AmbiguousTiffPages
from src.train_nlr import validate_transfer_report, config_text


class AIHubTests(unittest.TestCase):
    def test_tiff_overview_allowed_but_multiple_originals_rejected(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            path = Path(tmp)/'overview.tif'
            original = Image.new('RGB', (16,16), (25,50,75))
            overview = Image.new('RGB', (8,8), (0,0,0))
            original.save(path, save_all=True, append_images=[overview])
            # A dedicated one-tag-per-frame fixture is unnecessary: model the
            # standard reduced-image tag while retaining real TIFF decoding.
            from unittest.mock import patch
            real_open = Image.open
            class TaggedFrames:
                def __init__(self): self.image = real_open(path)
                def __enter__(self): return self
                def __exit__(self, *args): self.image.close()
                @property
                def n_frames(self): return self.image.n_frames
                def seek(self, index):
                    self.image.seek(index)
                    self.tag_v2 = {254: 1 if index else 0}
                def __getattr__(self, name): return getattr(self.image, name)
            with patch('src.prepare_aihub.Image.open', return_value=TaggedFrames()):
                loaded = load_source_rgb(path)
            self.assertEqual(loaded.size, (16,16))
            self.assertEqual(loaded.getpixel((0,0)), (25,50,75))
            original.save(Path(tmp)/'ambiguous.tif', save_all=True,
                          append_images=[Image.new('RGB',(16,16))])
            records = []
            with self.assertRaises(AmbiguousTiffPages):
                load_source_rgb(Path(tmp)/'ambiguous.tif', records)
            self.assertIsNone(records[0]['selected_frame'])
            self.assertTrue(records[0]['skipped'])

    def test_folder_manifest_pairs_only_training_and_rejects_mismatch(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root = Path(tmp)
            for folder in ('train_data/nested', 'train_label', 'valid_data', 'train_meta'):
                (root/folder).mkdir(parents=True)
            (root/'train_data/nested/a.tif').write_bytes(b'image')
            (root/'train_label/a.json').write_text('{}')
            (root/'valid_data/unused.tif').write_bytes(b'unused')
            self.assertEqual(folder_manifest(root, root/'manifest.csv'), 1)
            with (root/'manifest.csv').open(encoding='utf-8') as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual(row['split'], 'train')
            self.assertEqual(Path(row['image']), root/'train_data/nested/a.tif')
            (root/'train_label/extra.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                folder_manifest(root, root/'bad.csv')

    def payload(self, **changes):
        obj = dict(class_name="Solar panel", annotation_type="polygon",
                   points=[1, 1, 6, 1, 6, 6, 1, 6])
        obj.update(changes)
        return {"Learning Data Info": {"objects": [obj]}}

    def test_polygon_and_rejections(self):
        ann = panel_annotations(self.payload(), 8, 8)[0]
        self.assertGreater(ann["area"], 0)
        self.assertEqual(ann["segmentation"]["size"], [8, 8])
        for changes in [dict(annotation_type="box"), dict(class_name="Roof"),
                        dict(points=[1, 2, 3]), dict(points=[0, 0, 9, 0, 0, 8]),
                        dict(points=[0, 0, float("nan"), 0, 0, 8])]:
            with self.assertRaises(ValueError):
                panel_annotations(self.payload(**changes), 8, 8)

    def test_merge_preserves_negatives_and_rejects_duplicates_and_groups(self):
        parent = Path(__file__).resolve().parents[2] / 'tmp'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root = Path(tmp)
            local = root / "local"
            for split, offset in [("train", 0), ("val", 50)]:
                directory = local / split / "images"
                directory.mkdir(parents=True)
                for i in range(2):
                    Image.new("RGB", (8, 8), (offset + i, 0, 0)).save(directory / f"{split}_{i}.png")
                ann = panel_annotations(self.payload(), 8, 8)[0]
                ds = dict(categories=[dict(id=1, name="photovoltaic-panel")],
                          images=[dict(id=i, file_name=f"{split}_{i}.png", width=8, height=8) for i in range(2)],
                          annotations=[dict(ann, id=1, image_id=0)])
                (local / split / "annotations.json").write_text(json.dumps(ds), encoding="utf-8")
            Image.new("RGB", (8, 8), (100, 0, 0)).save(root / "a.tif")
            (root / "a.json").write_text(json.dumps(self.payload()), encoding="utf-8")
            manifest = root / "manifest.csv"

            def write(rows):
                with manifest.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["image", "label", "split", "group"])
                    writer.writerows(rows)

            write([["a.tif", "a.json", "train", "site-a"]])
            report = prepare(manifest, local, root / "out", tile_size=8)
            self.assertEqual(report["stats"]["train"], dict(images=3, instances=2, empty_images=1))
            self.assertEqual(report["stats"]["val"]["images"], 2)
            with self.assertRaises(FileExistsError):
                prepare(manifest, local, root / "out")
            Image.new("RGB", (8, 8), (100, 0, 0)).save(root / "b.tif")
            write([["a.tif", "a.json", "train", "same"], ["b.tif", "a.json", "val", "same"]])
            with self.assertRaisesRegex(ValueError, "group leaks"):
                prepare(manifest, local, root / "bad-group")
            write([["a.tif", "a.json", "train", "a"], ["b.tif", "a.json", "val", "b"]])
            heldout = prepare(manifest, local, root / "heldout", tile_size=8)
            self.assertEqual(heldout['stats']['val']['images'], 2)
            self.assertEqual(heldout['excluded_external_val_images'], 1)
            write([["a.tif", "a.json", "train", "a"], ["b.tif", "a.json", "train", "b"]])
            dedup = prepare(manifest, local, root / "dedup", tile_size=8)
            self.assertEqual(dedup['duplicate_tiles_skipped'], 1)
            (root/'b.json').write_text(json.dumps(self.payload(points=[1,1,4,1,4,4,1,4])), encoding='utf-8')
            write([["a.tif", "a.json", "train", "a"], ["b.tif", "b.json", "train", "b"]])
            conflict = prepare(manifest, local, root / "conflict", tile_size=8)
            self.assertEqual(conflict['duplicate_tiles_skipped'], 1)
            self.assertTrue(conflict['duplicate_tiles'][0]['labels_differ'])
            self.assertEqual(conflict['stats']['train']['images'], 3)
            ds = json.loads((root/'conflict/train/annotations.json').read_text())
            kept = [ann for ann in ds['annotations'] if ann['image_id']==3]
            original_ann = panel_annotations(self.payload(),8,8)[0]
            self.assertEqual(kept[0]['segmentation'], original_ann['segmentation'])
            self.assertEqual(conflict['duplicate_policy'], 'keep_first')
            # Distinct quadrant pixels, one panel-containing tile plus three negatives.
            image = Image.new('RGB', (8,8))
            for i, box in enumerate([(0,0,4,4),(4,0,8,4),(0,4,4,8),(4,4,8,8)]):
                image.paste((100+i,0,0), box)
            image.save(root/'a.tif')
            (root/'a.json').write_text(json.dumps(self.payload(points=[1,1,3,1,3,3,1,3])), encoding='utf-8')
            write([["a.tif", "a.json", "train", "a"]])
            # External-only preparation must work with no local training split.
            (local/'train/annotations.json').rename(local/'train/unused.json')
            external = prepare(manifest, local, root/'external', tile_size=4, train_source='aihub')
            self.assertEqual(external['stats']['train'], dict(images=4, instances=1, empty_images=3))
            self.assertTrue(all(p['origin']=='aihub' for p in external['images'] if p['split']=='train'))
            self.assertTrue(all(p['origin']=='local' for p in external['images'] if p['split']=='val'))
            standalone = prepare(manifest, None, root/'standalone', tile_size=4, train_source='aihub')
            self.assertEqual(set(standalone['stats']), {'train'})
            self.assertFalse((root/'standalone/val').exists())
            self.assertEqual(standalone['stats']['train']['images'], 4)

    def test_config_can_use_separate_validation_root(self):
        text = config_text(Path('base.py'), Path('external'), Path('run'), Path('model.pth'),
                           12, 1, 4, 0, 5e-5, 2026, train_val_only=True,
                           bitmap_masks=True, val_dataset=Path('local'))
        namespace = {}
        exec(compile(text, '<config>', 'exec'), namespace)
        self.assertEqual(namespace['train_dataloader']['dataset']['data_root'], Path('external').resolve().as_posix()+'/')
        self.assertEqual(namespace['val_dataloader']['dataset']['data_root'], Path('local').resolve().as_posix()+'/')
        self.assertEqual(namespace['val_evaluator']['ann_file'], Path('local').resolve().as_posix()+'/val/annotations.json')

    def test_four_crops_preserve_area_and_boundary_instances(self):
        annotations = panel_annotations(self.payload(points=[2,2,6,2,6,6,2,6]), 8, 8)
        crops = [crop_annotations(annotations, (x,y,x+4,y+4))
                 for y in (0,4) for x in (0,4)]
        self.assertTrue(all(len(crop) == 1 for crop in crops))
        self.assertEqual(sum(c[0]['area'] for c in crops), annotations[0]['area'])
        for crop in crops:
            self.assertEqual(crop[0]['segmentation']['size'], [4,4])
            x,y,w,h = crop[0]['bbox']
            self.assertTrue(0 <= x < x+w <= 4 and 0 <= y < y+h <= 4)
        small = panel_annotations(self.payload(), 8, 8)
        self.assertEqual(crop_annotations(small, (7,7,8,8)), [])

    def test_single_checkpoint_requires_complete_transfer(self):
        report = dict(loaded_keys=["backbone.w", "neck.w", "bbox_head.rtm_cls.weight"],
                      missing_after_load=[], skipped_tensors=0)
        validate_transfer_report(report, "single")
        with self.assertRaises(RuntimeError):
            validate_transfer_report(dict(report, missing_after_load=["bbox_head.rtm_cls.weight"]), "single")


if __name__ == "__main__":
    unittest.main()
