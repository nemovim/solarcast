"""Fetch complete VWorld building footprints for one z19 tile."""
import requests
from pyproj import Transformer
from modules.imagery.tiles import tile_transform


def fetch_buildings(x, y, *, key, domain=''):
    affine = tile_transform(x, y)
    project = Transformer.from_crs(3857, 4326, always_xy=True)
    west, north = project.transform(*(affine * (0, 0)))
    east, south = project.transform(*(affine * (256, 256)))
    features = []
    for page in range(1, 11):
        params = {'key': key, 'service': 'data', 'version': '2.0', 'request': 'GetFeature',
                  'data': 'LT_C_SPBD', 'format': 'json', 'geometry': 'true', 'attribute': 'true',
                  'crs': 'EPSG:4326', 'geomFilter': f'BOX({west},{south},{east},{north})',
                  'size': 1000, 'page': page}
        if domain:
            params['domain'] = domain
        try:
            response = requests.get('https://api.vworld.kr/req/data', params=params, timeout=(10, 30))
            response.raise_for_status()
            result = response.json()['response']
        except (requests.RequestException, ValueError, KeyError):
            raise RuntimeError(f'VWorld building request failed for {x}_{y}') from None
        if result.get('status') == 'NOT_FOUND':
            return features
        if result.get('status') != 'OK':
            raise RuntimeError(f'VWorld building request failed for {x}_{y}')
        items = result.get('result', {}).get('featureCollection', {}).get('features')
        if not isinstance(items, list):
            raise RuntimeError(f'Invalid building response for {x}_{y}')
        features.extend(items)
        total = result.get('page', {}).get('total')
        try:
            done = page >= int(total) if total is not None else len(items) < 1000
        except (TypeError, ValueError):
            raise RuntimeError(f'Invalid building page count for {x}_{y}') from None
        if done:
            return features
    raise RuntimeError(f'Too many buildings for tile {x}_{y}; tile was not marked complete')
