import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ASSIGNMENTS, IMAGES, Conflict, Store, mask_png, validate_polygons
from merge import collect, merge


def raw_png(data):
    position, compressed, header = 8, b"", None
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    while position < len(data):
        size = struct.unpack(">I", data[position:position+4])[0]
        kind = data[position+4:position+8]
        payload = data[position+8:position+8+size]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        if kind == b"IDAT":
            compressed += payload
        position += 12 + size
    return header, zlib.decompress(compressed)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assignments = self.root / "assignments.json"
        self.assignments.write_text(json.dumps({"schema": 2, "images": [
            {"id": f"{part}_1", "file": f"{part}_1.jpeg", "part": part, "width": 8, "height": 6}
            for part in range(1, 5)]}), "utf-8")
        self.store = Store(self.assignments, self.root, self.root / "work")

    def save(self, key="1_1", status="done", polygons=None):
        record = self.store.read(key)
        record.update(status=status, polygons=polygons if polygons is not None else [[[1, 1], [5, 1], [5, 4], [1, 4]]])
        return self.store.save(key, record)

    def archives(self):
        paths = []
        for part in range(1, 5):
            path = self.root / f"part-{part}.zip"
            path.write_bytes(self.store.export(part))
            paths.append(path)
        return paths

    def test_real_fixed_assignments_cover_all_images_once(self):
        store = Store(workspace=self.root / "real")
        self.assertEqual([sum(i["part"] == p for i in store.items) for p in range(1, 5)], [853, 853, 852, 852])
        self.assertEqual({p.name for p in IMAGES.glob('*.jpeg')}, {i['file'] for i in store.items})
        self.assertEqual(len(store.by_id), 3410)
        self.assertFalse(store.workspace.exists())

    def test_fresh_start_save_reload_and_conflict(self):
        initial = self.store.read("1_1")
        self.assertEqual(initial['status'], 'pending')
        saved = self.save()
        reopened = Store(self.assignments, self.root, self.root / 'work')
        self.assertEqual(saved, reopened.read('1_1'))
        initial.update(status='empty')
        with self.assertRaises(Conflict):
            reopened.save('1_1', initial)

    def test_assignment_hash_ignores_os_line_endings_and_formatting(self):
        alternate = self.root / 'windows.json'
        content = json.loads(self.assignments.read_text('utf-8'))
        alternate.write_bytes(json.dumps(content, indent=2).replace('\n', '\r\n').encode('utf-8'))
        other = Store(alternate, self.root, self.root / 'other')
        self.assertEqual(self.store.assignment_hash, other.assignment_hash)

    def test_reset_any_decision_preserves_geometry_and_excludes_training_mask(self):
        for status in ['empty', 'done', 'deferred', 'in_progress']:
            with self.subTest(status=status):
                before = self.save(status=status, polygons=[] if status == 'empty' else None)
                pending = self.store.save('1_1', {**before, 'status': 'pending', 'note': 'keep note'})
                self.assertEqual(pending['status'], 'pending')
                self.assertEqual(pending['polygons'], before['polygons'])
                self.assertEqual(pending['note'], 'keep note')
                self.assertEqual(self.store.read('1_1'), pending)
                with zipfile.ZipFile(io.BytesIO(self.store.export(1))) as archive:
                    self.assertNotIn('masks/1_1.png', archive.namelist())
                    self.assertEqual(json.loads(archive.read('manifest.json'))['complete'], 0)
                restored = self.store.save('1_1', {**before, 'revision': pending['revision']})
                self.assertEqual(restored['status'], status)
                self.assertEqual(restored['polygons'], before['polygons'])

    def test_invalid_status_geometry_and_part_do_not_write(self):
        for changes in [dict(status='done', polygons=[]), dict(status='empty', polygons=[[[1,1],[3,1],[3,3]]]),
                        dict(part=2), dict(status='unknown'), dict(assignment_sha256='other')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.save('1_1', {**self.store.blank('1_1'), **changes})
        self.assertFalse(self.store.record_path('1_1').exists())

    def test_reject_crossed_outside_duplicate_nan(self):
        invalid = [ [[0,0],[6,5],[0,5],[6,0]], [[0,0],[9,0],[1,3]],
                    [[0,0],[5,0],[5,0],[0,5]], [[0,0],[float('nan'),2],[2,3]],
                    [[0,0],[1,0],[2,0]] ]
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_polygons([points], 8, 6)

    def test_png_dimensions_union_and_empty(self):
        polygon = [[1,1],[5,1],[5,4],[1,4]]
        header, raw = raw_png(mask_png(8, 6, [polygon, polygon]))
        self.assertEqual(header, (8, 6, 8, 0, 0, 0, 0))
        self.assertEqual(raw.count(255), 12)
        self.assertEqual(len(raw), 54)
        self.assertEqual(set(raw), {0, 255})
        self.assertEqual(set(raw_png(mask_png(8, 6, []))[1]), {0})
        full = [[0,0],[8,0],[8,6],[0,6]]
        self.assertEqual(raw_png(mask_png(8,6,[full]))[1].count(255), 48)

    def test_failed_atomic_replace_preserves_previous(self):
        before = self.save()
        changed = {**before, 'note': 'new'}
        original_replace = __import__('os').replace
        def fail_record(source, destination):
            if Path(destination).name == '1_1.json':
                raise OSError('disk failure')
            return original_replace(source, destination)
        with patch('core.os.replace', side_effect=fail_record), self.assertRaises(OSError):
            self.store.save('1_1', changed)
        self.assertEqual(self.store.read('1_1'), before)

    def test_export_only_complete_masks_and_merge_report(self):
        self.save('1_1')
        self.save('2_1', 'empty', [])
        self.save('3_1', 'deferred')
        paths = self.archives()
        with zipfile.ZipFile(paths[2]) as archive:
            self.assertNotIn('masks/3_1.png', archive.namelist())
        report = merge(paths, self.root/'merged', self.assignments)
        self.assertEqual(report['complete'], 2)
        self.assertEqual(set(report['unfinished']), {'3_1','4_1'})
        self.assertEqual(len(list((self.root/'merged/masks').glob('*.png'))), 2)
        with self.assertRaises(ValueError):
            merge(paths, self.root/'merged', self.assignments)
        with self.assertRaises(ValueError):
            merge(paths, self.root/'strict', self.assignments, True)
        self.assertFalse((self.root/'strict').exists())

    def test_missing_duplicate_parts_and_corrupt_mask(self):
        self.save()
        paths = self.archives()
        with self.assertRaises(ValueError):
            collect(paths[:3], self.store)
        with self.assertRaises(ValueError):
            collect([paths[0]]*4, self.store)
        corrupted = self.root/'bad.zip'
        with zipfile.ZipFile(paths[0]) as source, zipfile.ZipFile(corrupted, 'w') as destination:
            for name in source.namelist():
                destination.writestr(name, b'broken' if name.endswith('.png') else source.read(name))
        with self.assertRaises(ValueError):
            collect([corrupted, *paths[1:]], self.store)

    def test_all_complete_roundtrip(self):
        for part in range(1,5):
            self.save(f'{part}_1','empty',[])
        _, report = collect(self.archives(), self.store, True)
        self.assertEqual(report['complete'],4)
        self.assertEqual(report['unfinished'],[])


if __name__ == '__main__':
    unittest.main()
