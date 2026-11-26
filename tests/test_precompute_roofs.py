import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyproj import Transformer
from shapely.geometry import mapping, box
from modules.imagery.tiles import tile_transform
from modules.imagery.buildings import fetch_buildings
from modules.roof.inference.sam import SamRoofModel
from pipelines.precompute_roofs import main
from pipelines.precompute import main as run_pipeline


class RoofPrecomputeTests(unittest.TestCase):
    def test_pipeline_passes_model_names_to_both_steps(self):
        with patch('sys.argv', ['precompute', '--panel', 'panel2', '--roof', 'sam2', '--bounds', '1', '2', '3', '4']), \
             patch('pipelines.precompute.subprocess.run') as run:
            run_pipeline()
        self.assertEqual(run.call_count, 2)
        panel, roof = (call.args[0] for call in run.call_args_list)
        self.assertEqual(panel[2:4], ['pipelines.precompute_panels', 'panel2'])
        self.assertEqual(roof[2:4], ['pipelines.precompute_roofs', 'sam2'])
        self.assertIn('--bounds', roof)

    def test_building_api_rejects_incomplete_pages(self):
        class Response:
            def __init__(self, status, features=None, total=None):
                self.value = {'response': {'status': status, 'page': {'total': total},
                                           'result': {'featureCollection': {'features': features or []}}}}
            def raise_for_status(self): pass
            def json(self): return self.value
        with patch('modules.imagery.buildings.requests.get', side_effect=[Response('OK', [{'id': 1}], 2), Response('OK', [{'id': 2}], 2)]):
            self.assertEqual([item['id'] for item in fetch_buildings(446793, 207630, key='fake')], [1, 2])
        with patch('modules.imagery.buildings.requests.get', return_value=Response('ERROR')):
            with self.assertRaises(RuntimeError):
                fetch_buildings(446793, 207630, key='fake')

    def test_sam_uses_building_prompt_and_returns_center_tile_mask(self):
        class FakePredictor:
            def set_image(self, crop): self.shape = crop.shape[:2]
            def predict(self, **kwargs):
                self.prompt = kwargs
                return np.ones((1, *self.shape), dtype=bool), np.array([0.9]), None

        model = SamRoofModel.__new__(SamRoofModel)
        model.predictor = FakePredictor()
        transform = tile_transform(446793, 207630)
        left, top = transform * (50, 50)
        right, bottom = transform * (100, 100)
        lonlat = Transformer.from_crs(3857, 4326, always_xy=True)
        west, south = lonlat.transform(left, bottom)
        east, north = lonlat.transform(right, top)
        image = np.full((768, 768, 3), 255, dtype=np.uint8)
        tile_left, tile_top = transform * (0, 0)
        tile_right, tile_bottom = transform * (256, 256)
        result = model.predict_tile(image, tile_transform(446792, 207629),
                                    [{'geometry': mapping(box(west, south, east, north))}],
                                    (tile_left, tile_bottom, tile_right, tile_top), 35.0)
        self.assertEqual(result.shape, (256, 256))
        self.assertTrue(result[75, 75])
        self.assertFalse(result[10, 10])
        self.assertIn('box', model.predictor.prompt)

    def test_roof_tiles_resume_and_failures_remain_unprocessed(self):
        class FakeModel:
            def __init__(self, checkpoint, *, device):
                self.device = device
            def predict_tile(self, image, affine, features, bounds, latitude):
                mask = np.zeros((256, 256), dtype=bool)
                mask[40:60, 40:60] = True
                return mask

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'models/roof/model.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fake-sam')
            index = root/'custom.csv'
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text('x,y\n446793,207630\n446794,207630\n', encoding='utf-8')
            argv = ['precompute_roofs']
            with patch('sys.argv', argv), patch('pipelines.precompute_roofs.ROOT', root), \
                 patch('pipelines.precompute_roofs.BITGARAM_BOUNDS', (446793, 207630, 446794, 207630)), \
                 patch.dict('os.environ', {'VWORLD_KEY': 'fake-key', 'VWORLD_DOMAIN': ''}), \
                 patch('pipelines.precompute_roofs.SamRoofModel', FakeModel), \
                 patch('pipelines.precompute_roofs.fetch_tile', return_value=np.zeros((256, 256, 3), np.uint8)), \
                 patch('pipelines.precompute_roofs.fetch_buildings', side_effect=[[{'geometry': {}}], RuntimeError('failed')]):
                with self.assertRaisesRegex(RuntimeError, '1 roof tiles failed'):
                    main()
            output = root/'frontend/public/precomputed/roof/model'
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertIn('446793_207630', manifest['tiles'])
            self.assertNotIn('446794_207630', manifest['tiles'])
            self.assertEqual(json.loads((output.parent/'index.json').read_text())['models'][0]['id'], 'model')
            with patch('sys.argv', argv + ['--tile-index', str(index)]), patch('pipelines.precompute_roofs.ROOT', root), \
                 patch.dict('os.environ', {'VWORLD_KEY': 'fake-key'}), \
                 patch('pipelines.precompute_roofs.SamRoofModel', FakeModel), \
                 patch('pipelines.precompute_roofs.fetch_buildings', return_value=[]) as buildings:
                main()
                self.assertEqual(buildings.call_count, 1)
            manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['tiles']['446794_207630']['count'], 0)


if __name__ == '__main__':
    unittest.main()
