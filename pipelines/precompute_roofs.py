"""Precompute SAM ViT-H roof masks from VWorld imagery and building footprints."""
import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from dotenv import load_dotenv

from modules.imagery.buildings import fetch_buildings
from modules.imagery.tiles import BITGARAM_BOUNDS, tile_transform
from modules.imagery.vworld import fetch_tile
from modules.roof.inference.sam import SamRoofModel
from pipelines.precompute_panels import atomic_json, export_tile, model_hash, update_catalog
from pipelines.shutdown import hard_exit_after_success

ROOT = Path(__file__).resolve().parents[1]


def mosaic_bounds(x, y, buildings):
    """Include complete intersecting footprints, subject to a memory limit."""
    affine = tile_transform(x, y)
    inverse = ~affine
    left, top = affine * (0, 0)
    right, bottom = affine * (256, 256)
    from shapely.geometry import box
    tile_box = box(left, bottom, right, top)
    project = Transformer.from_crs(4326, 3857, always_xy=True).transform
    xmin, ymin, xmax, ymax = x - 1, y - 1, x + 1, y + 1
    for feature in buildings:
        geometry = transform(project, shape(feature['geometry']))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError('Invalid building geometry; tile was not marked complete')
        if not geometry.intersects(tile_box):
            continue
        west, south, east, north = geometry.bounds
        pixels = [inverse * (px, py) for px in (west, east) for py in (south, north)]
        xmin = min(xmin, x + math.floor((min(p[0] for p in pixels) - 40) / 256))
        xmax = max(xmax, x + math.floor((max(p[0] for p in pixels) + 40) / 256))
        ymin = min(ymin, y + math.floor((min(p[1] for p in pixels) - 40) / 256))
        ymax = max(ymax, y + math.floor((max(p[1] for p in pixels) + 40) / 256))
    if xmin < 0 or ymin < 0 or xmax >= 2**19 or ymax >= 2**19 or (xmax-xmin+1)*(ymax-ymin+1) > 100:
        raise ValueError('Building imagery mosaic exceeds safe limits; tile was not marked complete')
    return xmin, ymin, xmax, ymax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', default='model', help='SAM ViT-H checkpoint in models/roof; .pth may be omitted')
    parser.add_argument('--tile-index', type=Path, help='CSV with x,y columns (overrides the default Bitgaram area)')
    parser.add_argument('--bounds', type=int, nargs=4, metavar=('X_MIN','Y_MIN','X_MAX','Y_MAX'))
    parser.add_argument('--output', type=Path, default=ROOT/'frontend/public/precomputed/roof')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hard-exit-after-success', dest='hard_exit_after_success', action='store_true', default=True, help='Bypass native shutdown hooks after successful processing (default)')
    parser.add_argument('--normal-exit', dest='hard_exit_after_success', action='store_false', help='Use normal Python shutdown')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.model):
        parser.error('Model must be a filename in models/roof')
    if args.bounds and args.tile_index:
        parser.error('Choose either --bounds or --tile-index')
    name = args.model if Path(args.model).suffix else args.model + '.pth'
    checkpoint = ROOT/'models/roof'/name
    if args.bounds or not args.tile_index:
        xmin, ymin, xmax, ymax = args.bounds or BITGARAM_BOUNDS
        if xmin > xmax or ymin > ymax:
            parser.error('invalid bounds')
        coords = [(x, y) for x in range(xmin, xmax+1) for y in range(ymin, ymax+1)]
    else:
        with args.tile_index.open(encoding='utf-8-sig', newline='') as stream:
            coords = sorted({(int(row['x']), int(row['y'])) for row in csv.DictReader(stream)})
    if not coords or any(x < 1 or y < 1 or x >= 2**19-1 or y >= 2**19-1 for x, y in coords):
        raise ValueError('Invalid or empty z19 tile range')
    load_dotenv(ROOT/'.env')
    key, domain = os.getenv('VWORLD_KEY', '').strip(), os.getenv('VWORLD_DOMAIN', '').strip()
    if not key:
        raise ValueError('VWORLD_KEY is required')
    target = args.output/Path(name).stem
    manifest_path = target/'manifest.json'
    manifest = {'version': 1, 'zoom': 19, 'kind': 'roof', 'model': Path(name).stem,
                'model_sha256': model_hash(checkpoint), 'method': 'sam-vit-h-footprint-v1',
                'source': 'VWorld z19 and LT_C_SPBD', 'imagery_observed_at': None, 'tiles': {}}
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(current.get(k) != manifest[k] for k in ('version', 'zoom', 'kind', 'model_sha256', 'method')):
            raise ValueError('Existing output was produced by different model or settings; use a new filename')
        manifest = current
    pending = [(x, y) for x, y in coords if f'{x}_{y}' not in manifest['tiles'] or not (target/'tiles'/f'{x}_{y}.geojson').is_file()]
    if not pending:
        update_catalog(args.output)
        print('All requested roof tiles were already processed.', flush=True)
        return
    model = SamRoofModel(str(checkpoint), device=args.device)

    @lru_cache(maxsize=96)
    def tile(x, y):
        return fetch_tile(x, y, key=key, domain=domain)

    failures = []
    catalog_updated = False
    for index, (x, y) in enumerate(pending, 1):
        tile_id = f'{x}_{y}'
        try:
            buildings = fetch_buildings(x, y, key=key, domain=domain)
            if buildings:
                xmin, ymin, xmax, ymax = mosaic_bounds(x, y, buildings)
                mosaic = np.concatenate([np.concatenate([tile(tx, ty) for tx in range(xmin, xmax+1)], axis=1)
                                         for ty in range(ymin, ymax+1)], axis=0)
                affine = tile_transform(x, y)
                left, top = affine * (0, 0)
                right, bottom = affine * (256, 256)
                latitude = Transformer.from_crs(3857, 4326, always_xy=True).transform((left+right)/2, (top+bottom)/2)[1]
                mask = model.predict_tile(mosaic, tile_transform(xmin, ymin), buildings,
                                          (left, bottom, right, top), latitude)
                prediction = type('Prediction', (), {'masks': [mask], 'scores': [1.0]})()
                result = export_tile(x, y, prediction)
                for feature in result['features']:
                    feature['properties'] = {}
            else:
                result = {'type': 'FeatureCollection', 'features': []}
            now = datetime.now(timezone.utc).isoformat()
            atomic_json(target/'tiles'/f'{tile_id}.geojson', result)
            manifest['tiles'][tile_id] = {'file': f'tiles/{tile_id}.geojson', 'count': len(result['features']), 'processed_at': now}
            manifest['updated_at'] = now
            atomic_json(manifest_path, manifest)
            if not catalog_updated:
                update_catalog(args.output)
                catalog_updated = True
            print(f'[{index}/{len(pending)}] {tile_id}: {len(buildings)} buildings', flush=True)
        except Exception as exc:
            failures.append(tile_id)
            print(f'[{index}/{len(pending)}] {tile_id}: failed ({type(exc).__name__}: {exc})', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} roof tiles failed and remain unprocessed: {", ".join(failures[:10])}')
    return args.device if args.hard_exit_after_success else None


if __name__ == '__main__':
    exit_device = main()
    if exit_device is not None:
        hard_exit_after_success(exit_device)
