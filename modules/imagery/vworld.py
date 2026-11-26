"""Retrieve VWorld z19 imagery in memory for offline inference."""
import io
import time
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image


def fetch_tile(x, y, *, key, domain=''):
    if not key:
        raise ValueError('VWORLD_KEY is required')
    url = f'https://api.vworld.kr/req/wmts/1.0.0/{quote(key, safe="")}/Satellite/19/{y}/{x}.jpeg'
    headers = {'Referer': domain} if domain else {}
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=(10, 30))
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(attempt + 1)
                continue
            response.raise_for_status()
            with Image.open(io.BytesIO(response.content)) as image:
                if image.size != (256, 256):
                    raise ValueError(f'Unexpected tile dimensions: {x}_{y}')
                return np.asarray(image.convert('RGB'))
        except (requests.Timeout, requests.ConnectionError):
            if attempt < 2:
                time.sleep(attempt + 1)
                continue
            raise RuntimeError(f'VWorld tile request failed for {x}_{y}') from None
        except requests.RequestException:
            raise RuntimeError(f'VWorld tile request failed for {x}_{y}') from None
    raise RuntimeError(f'VWorld tile request failed for {x}_{y}')
