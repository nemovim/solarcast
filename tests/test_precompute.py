import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Point, shape

from modules.imagery.tiles import BITGARAM_BOUNDS, tile_transform
from modules.imagery.vworld import fetch_tile
from modules.panel.inference.rtmdet import RTMDetPanelModel, prepare_inference_config
from pipelines.precompute_panels import export_tile, main


class PrecomputeTests(unittest.TestCase):
    def test_tile_coordinates_and_exported_panel(self):
        mask = np.zeros((256, 256), bool)
        mask[30:40, 50:70] = True
        result = type('Prediction', (), {'masks': [mask], 'scores': [0.73]})()
        features = export_tile(446793, 207630, result)['features']
        self.assertEqual(len(features), 1)
        self.assertAlmostEqual(features[0]['properties']['confidence'], 0.73)
        expected = Transformer.from_crs(3857, 4326, always_xy=True).transform(*(tile_transform(446793, 207630) * (50, 30)))
        self.assertLess(shape(features[0]['geometry']).distance(Point(expected)), 1e-7)

    def test_precompute_resume_records_empty_tile_without_real_inference(self):
        seen = []
        class FakeModel:
            def __init__(self, *args, **kwargs): pass
            def predict(self, image):
                seen.append(image.shape)
                return type('Prediction', (), {'masks': [], 'scores': []})()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiles = root / 'tiles'
            tiles.mkdir()
            Image.new('RGB', (256, 256)).save(tiles / '446793_207630.jpeg')
            checkpoint = root / 'model.pth'
            checkpoint.write_bytes(b'fake model')
            config = root / 'config.py'
            config.write_text('fake config', encoding='utf-8')
            output = root / 'output'
            argv = ['precompute', '--source', 'local', '--tiles', str(tiles), '--config', str(config), '--checkpoint', str(checkpoint), '--output', str(output), '--bounds', '446793', '207630', '446793', '207630']
            with patch('sys.argv', argv), patch('pipelines.precompute_panels.RTMDetPanelModel', FakeModel):
                self.assertEqual(main(), 'cuda:0')
                self.assertIsNone(main())
            manifest = json.loads((output / 'model/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(seen, [(256, 256, 3)])
            self.assertEqual(manifest['tiles']['446793_207630']['count'], 0)
            self.assertEqual(json.loads((output / 'model/tiles/446793_207630.geojson').read_text(encoding='utf-8'))['features'], [])

    def test_default_area_matches_existing_bitgaram_extent(self):
        xmin, ymin, xmax, ymax = BITGARAM_BOUNDS
        self.assertEqual((xmin, ymin, xmax, ymax), (446760, 207604, 446821, 207658))
        self.assertEqual((xmax-xmin+1)*(ymax-ymin+1), 3410)

    def test_default_command_uses_vworld_bounds_and_cuda(self):
        calls = []
        class FakeModel:
            def __init__(self, config, checkpoint, *, device, threshold):
                calls.append((device, threshold))
            def predict(self, image):
                return type('Prediction', (), {'masks': [], 'scores': []})()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'modules/panel/configs/inference.py'
            config.parent.mkdir(parents=True)
            config.write_text('fake config', encoding='utf-8')
            checkpoint = root / 'models/panel/best_model.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'fake model')
            other_checkpoint = checkpoint.with_name('other.pth')
            other_checkpoint.write_bytes(b'other model')
            with patch('sys.argv', ['precompute']), patch('pipelines.precompute_panels.ROOT', root), \
                 patch('pipelines.precompute_panels.BITGARAM_BOUNDS', (446793, 207630, 446793, 207630)), \
                 patch.dict('os.environ', {'VWORLD_KEY': 'fake-key', 'VWORLD_DOMAIN': ''}), \
                 patch('pipelines.precompute_panels.RTMDetPanelModel', FakeModel), \
                 patch('pipelines.precompute_panels.fetch_tile', return_value=np.zeros((256, 256, 3), np.uint8)) as fetch:
                main()
                with patch('sys.argv', ['precompute', 'other']):
                    main()
                    other_checkpoint.write_bytes(b'changed model')
                    with self.assertRaisesRegex(ValueError, 'different model or settings'):
                        main()
            self.assertEqual(fetch.call_count, 2)
            fetch.assert_called_with(446793, 207630, key='fake-key', domain='')
            self.assertEqual(calls, [('cuda:0', 0.05), ('cuda:0', 0.05)])
            output = root / 'frontend/public/precomputed/panel'
            self.assertTrue((output / 'best_model/tiles/446793_207630.geojson').is_file())
            self.assertTrue((output / 'other/tiles/446793_207630.geojson').is_file())
            self.assertEqual([item['id'] for item in json.loads((output / 'index.json').read_text())['models']], ['best_model', 'other'])

    def test_vworld_retries_and_does_not_expose_key_in_errors(self):
        data = io.BytesIO()
        Image.new('RGB', (256, 256)).save(data, format='JPEG')
        response = type('Response', (), {'status_code': 200, 'content': data.getvalue(), 'raise_for_status': lambda self: None})()
        with patch('modules.imagery.vworld.requests.get', side_effect=[requests.Timeout('secret-key'), response]), patch('modules.imagery.vworld.time.sleep'):
            self.assertEqual(fetch_tile(1, 2, key='secret-key').shape, (256, 256, 3))
        with patch('modules.imagery.vworld.requests.get', side_effect=requests.Timeout('secret-key')), patch('modules.imagery.vworld.time.sleep'):
            with self.assertRaises(RuntimeError) as error:
                fetch_tile(1, 2, key='secret-key')
            self.assertNotIn('secret-key', str(error.exception))

    def test_inference_config_removes_ground_truth_steps(self):
        class Config(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        steps = [Config(type='LoadImageFromFile'), Config(type='LoadAnnotations'), Config(type='PackDetInputs')]
        config = Config(test_pipeline=steps[:], test_dataloader=Config(dataset=Config(pipeline=steps[:])))
        prepare_inference_config(config)
        self.assertEqual([step['type'] for step in config.test_pipeline], ['LoadImageFromFile', 'PackDetInputs'])
        self.assertEqual([step['type'] for step in config.test_dataloader.dataset.pipeline], ['LoadImageFromFile', 'PackDetInputs'])

    def test_panel_adapter_uses_rgb_input_and_filters_by_score(self):
        class Tensor:
            def __init__(self, data): self.data = np.asarray(data)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.data
        instances = type('Instances', (), {
            'scores': Tensor([0.2, 0.8]), 'labels': Tensor([0, 0]),
            'masks': Tensor(np.ones((2, 3, 4), dtype=bool)),
            'bboxes': Tensor([[0, 0, 1, 1], [1, 1, 3, 3]]),
        })()
        seen = []
        def predict(model, bgr):
            seen.append(bgr[0, 0].tolist())
            return type('Result', (), {'pred_instances': instances})()
        model = RTMDetPanelModel.__new__(RTMDetPanelModel)
        model.model, model.threshold, model.version, model.inference_detector = object(), 0.5, 'fake', predict
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[0, 0] = [10, 20, 30]
        result = model.predict(image)
        self.assertEqual(seen, [[30, 20, 10]])
        self.assertEqual(result.masks.shape, (1, 3, 4))
        self.assertAlmostEqual(result.scores[0], 0.8)


if __name__ == '__main__':
    unittest.main()
