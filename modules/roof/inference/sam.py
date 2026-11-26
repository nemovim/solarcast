"""Building-prompted SAM ViT-H roof segmentation on an image mosaic."""
import math
from pathlib import Path
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from shapely.geometry import mapping, shape, box
from shapely.ops import transform
from .heuristics import seeds, negatives, select_mask, refine


class SamRoofModel:
    def __init__(self, checkpoint_path, *, device='cuda:0'):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f'SAM checkpoint not found: {path}')
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        # This adapter intentionally supports the original ViT-H architecture only.
        with torch.device('meta'):
            model = sam_model_registry['vit_h'](checkpoint=None)
        state = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(state.get('state_dict', state), strict=True, assign=True)
        model.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        model.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        self.predictor = SamPredictor(model.to(device).eval())

    def predict_tile(self, image, affine, features, tile_bounds, latitude):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('SAM input must be an RGB uint8 image')
        inverse = ~affine
        to_image = Transformer.from_crs(4326, 3857, always_xy=True).transform
        h, w = image.shape[:2]
        roof_all = np.zeros((h, w), dtype=bool)
        tile = box(*tile_bounds)
        for feature in features:
            geometry = transform(to_image, shape(feature['geometry']))
            if not geometry.is_valid or geometry.is_empty:
                raise ValueError('Invalid building geometry; tile was not marked complete')
            if not geometry.intersects(tile):
                continue
            minx, miny, maxx, maxy = geometry.bounds
            corners = [inverse * (x, y) for x in (minx, maxx) for y in (miny, maxy)]
            if min(p[0] for p in corners) < 0 or max(p[0] for p in corners) > w or min(p[1] for p in corners) < 0 or max(p[1] for p in corners) > h:
                raise ValueError('Building exceeds imagery mosaic; tile was not marked complete')
            left = max(0, math.floor(min(p[0] for p in corners)) - 40)
            right = min(w, math.ceil(max(p[0] for p in corners)) + 40)
            top = max(0, math.floor(min(p[1] for p in corners)) - 40)
            bottom = min(h, math.ceil(max(p[1] for p in corners)) + 40)
            if right <= left or bottom <= top:
                raise ValueError('Invalid building crop; tile was not marked complete')
            crop = image[top:bottom, left:right]
            footprint = rasterize([(mapping(geometry), 1)], out_shape=crop.shape[:2],
                                  transform=affine * Affine.translation(left, top)).astype(bool)
            if not footprint.any():
                raise ValueError('Building footprint is smaller than one pixel; tile was not marked complete')
            positive, negative = seeds(footprint), negatives(footprint)
            points = np.vstack([positive, negative])
            labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
            ys, xs = np.nonzero(footprint)
            prompt_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
            self.predictor.set_image(crop)
            masks, scores, _ = self.predictor.predict(point_coords=points.astype(float), point_labels=labels,
                                                       box=prompt_box, multimask_output=True)
            gsd = abs(affine.a) * math.cos(math.radians(latitude))
            roof = refine(select_mask(masks, scores, positive), footprint, crop, positive, gsd)
            if not roof.any():
                raise ValueError('SAM returned an empty roof; tile was not marked complete')
            roof_all[top:bottom, left:right] |= roof
        tile_left, tile_top = inverse * (tile_bounds[0], tile_bounds[3])
        col, row = round(tile_left), round(tile_top)
        if col < 0 or row < 0 or col + 256 > w or row + 256 > h:
            raise ValueError('Tile exceeds imagery mosaic; tile was not marked complete')
        return roof_all[row:row + 256, col:col + 256]
