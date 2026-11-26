"""Check that validation images do not occur in the training dataset."""
import hashlib
import json
from PIL import Image


def check_evaluation_leakage(training, validation):
    def fingerprints(dataset, split):
        payload = json.loads((dataset / split / 'annotations.json').read_text(encoding='utf-8'))
        result = set()
        for item in payload['images']:
            with Image.open(dataset / split / 'images' / item['file_name']) as image:
                rgb = image.convert('RGB')
                result.add(hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest())
        return result
    if fingerprints(training, 'train') & fingerprints(validation, 'val'):
        raise ValueError('Validation images occur in the training dataset')
