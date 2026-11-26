"""Run NLR over z19 tiles and save derived panel geometries for the web."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union
from shapely import make_valid

from modules.imagery.tiles import BITGARAM_BOUNDS, tile_transform
from modules.imagery.vworld import fetch_tile
from modules.panel.inference.rtmdet import RTMDetPanelModel
from pipelines.shutdown import hard_exit_after_success

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False, suffix='.tmp') as stream:
        name = stream.name
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    os.replace(name, path)


def model_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def update_catalog(output_root):
    models = []
    for manifest_path in sorted(output_root.glob('*/manifest.json')):
        model_id = manifest_path.parent.name
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', model_id):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if manifest.get('version') == 1 and isinstance(manifest.get('model'), str):
                models.append({'id': model_id, 'name': manifest['model']})
        except (OSError, ValueError, AttributeError):
            continue
    atomic_json(output_root/'index.json', {'version': 1, 'models': models})


def export_tile(x, y, result):
    affine = tile_transform(x, y)
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True).transform
    features = []
    for mask, score in zip(result.masks, result.scores):
        parts = [shape(g) for g, v in shapes(mask.astype('uint8'), mask=mask.astype(bool), transform=affine) if v]
        if not parts:
            continue
        geometry = make_valid(unary_union(parts)).simplify(0.15, preserve_topology=True)
        geometry = transform(to_wgs, geometry)
        if geometry.is_empty or geometry.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        features.append({'type':'Feature','geometry':mapping(geometry),'properties':{'confidence':round(float(score), 6)}})
    return {'type':'FeatureCollection','features':features}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', help='Checkpoint name in models/panel (default: best_model.pth; .pth may be omitted)')
    parser.add_argument('--tiles', type=Path, default=ROOT/'data/bitgaram/raw_tiles/z19')
    parser.add_argument('--source', choices=('local','vworld'), default='vworld', help='Read local tiles or fetch VWorld tiles in memory (default: vworld)')
    parser.add_argument('--tile-index', type=Path, help='CSV with x,y columns (overrides the default Bitgaram area)')
    parser.add_argument('--config', type=Path, default=ROOT/'modules/panel/configs/inference.py')
    parser.add_argument('--checkpoint', type=Path, help='Checkpoint path outside models/panel')
    parser.add_argument('--model-name', help='Label saved in the output manifest (default: checkpoint name)')
    parser.add_argument('--output', type=Path, default=ROOT/'frontend/public/precomputed/panel', help='Root directory for panel model result folders')
    parser.add_argument('--device', default='cuda:0', help='PyTorch inference device (default: cuda:0)')
    parser.add_argument('--bounds', type=int, nargs=4, metavar=('X_MIN','Y_MIN','X_MAX','Y_MAX'))
    parser.add_argument('--all', action='store_true', help='Process every input tile (default when --bounds is omitted)')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--hard-exit-after-success', dest='hard_exit_after_success', action='store_true', default=True, help='Bypass native shutdown hooks after successful processing (default)')
    parser.add_argument('--normal-exit', dest='hard_exit_after_success', action='store_false', help='Use normal Python shutdown')
    args = parser.parse_args()
    if args.model and args.checkpoint:
        parser.error('Choose a model name or --checkpoint, not both')
    if args.model:
        if args.model in ('.', '..') or '/' in args.model or '\\' in args.model:
            parser.error('Model must be a filename in models/panel; use --checkpoint for a path')
        model_filename = args.model if Path(args.model).suffix else args.model + '.pth'
        args.checkpoint = ROOT/'models/panel'/model_filename
    elif not args.checkpoint:
        args.checkpoint = ROOT/'models/panel/best_model.pth'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.checkpoint.name):
        parser.error('Checkpoint filename must use letters, numbers, dots, underscores or hyphens')
    model_name = args.model_name or args.checkpoint.stem
    output_root = args.output
    args.output = output_root / args.checkpoint.stem
    if args.bounds and args.all:
        parser.error('Choose either --bounds or --all')
    if args.bounds and args.tile_index:
        parser.error('Choose either --bounds or --tile-index')
    if args.threads < 1:
        parser.error('--threads must be positive')
    bounds = args.bounds or (BITGARAM_BOUNDS if args.source == 'vworld' and not args.tile_index else None)
    if bounds:
        xmin,ymin,xmax,ymax = bounds
        if xmin > xmax or ymin > ymax:
            parser.error('invalid bounds')
        coords = [(x,y) for x in range(xmin,xmax+1) for y in range(ymin,ymax+1)]
        missing = [f'{x}_{y}.jpeg' for x,y in coords if not (args.tiles/f'{x}_{y}.jpeg').is_file()]
        if missing and args.source == 'local':
            raise FileNotFoundError(f'Input tiles missing: {len(missing)}; first {missing[0]}')
    else:
        tile_index = args.tile_index
        if tile_index:
            with tile_index.open(encoding='utf-8-sig',newline='') as stream:
                coords = sorted({(int(row['x']),int(row['y'])) for row in csv.DictReader(stream)})
        elif args.source == 'local':
            coords = sorted(tuple(map(int,p.stem.split('_'))) for p in args.tiles.glob('*.jpeg'))
        if not coords: raise FileNotFoundError('No source tiles found')
    if any(x < 0 or y < 0 or x >= 2**19 or y >= 2**19 for x,y in coords):
        raise ValueError('Invalid z19 tile coordinate')
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env')
    except ImportError:
        pass
    vworld_key = os.getenv('VWORLD_KEY','').strip()
    vworld_domain = os.getenv('VWORLD_DOMAIN','').strip()
    if args.source == 'vworld' and not vworld_key:
        raise ValueError('VWORLD_KEY is required for --source vworld')
    checkpoint_hash = model_hash(args.checkpoint)
    config_hash = model_hash(args.config)
    manifest_path = args.output/'manifest.json'
    manifest = {'version':1,'zoom':19,'model':model_name,'model_sha256':checkpoint_hash,
                'config_sha256':config_hash,'min_confidence':0.05,'source':'VWorld z19',
                'imagery_observed_at':None,'tiles':{}}
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(current.get(k) != manifest[k] for k in ('version','zoom','model_sha256','config_sha256','min_confidence')):
            raise ValueError('Existing output was produced by different model or settings; choose a new --output')
        manifest = current
    pending = [(x, y) for x, y in coords
               if f'{x}_{y}' not in manifest['tiles'] or not (args.output/'tiles'/f'{x}_{y}.geojson').is_file()]
    if not pending:
        update_catalog(output_root)
        print('All requested tiles were already processed.', flush=True)
        return
    import torch
    torch.set_num_threads(args.threads)
    model = RTMDetPanelModel(str(args.config), str(args.checkpoint), device=args.device, threshold=0.05)
    catalog_updated = False
    for index, (x,y) in enumerate(coords, 1):
        key = f'{x}_{y}'
        target = args.output/'tiles'/f'{key}.geojson'
        if key in manifest['tiles'] and target.is_file():
            print(f'[{index}/{len(coords)}] {key} already processed', flush=True)
            continue
        if args.source == 'vworld':
            pixels = fetch_tile(x, y, key=vworld_key, domain=vworld_domain)
        else:
            source=args.tiles/f'{key}.jpeg'
            with Image.open(source) as image:
                if image.size != (256,256): raise ValueError(f'Unexpected tile dimensions: {key}')
                pixels = np.asarray(image.convert('RGB'))
        now = datetime.now(timezone.utc)
        prediction = model.predict(pixels)
        output = export_tile(x,y,prediction)
        atomic_json(target, output)
        manifest['tiles'][key] = {'file':f'tiles/{key}.geojson','count':len(output['features']),'processed_at':now.isoformat()}
        manifest['updated_at'] = now.isoformat()
        atomic_json(manifest_path, manifest)
        if not catalog_updated:
            update_catalog(output_root)
            catalog_updated = True
        print(f'[{index}/{len(coords)}] {key}: {len(output["features"])} detections', flush=True)
    return args.device if args.hard_exit_after_success else None


if __name__ == '__main__':
    exit_device = main()
    if exit_device is not None:
        hard_exit_after_success(exit_device)
