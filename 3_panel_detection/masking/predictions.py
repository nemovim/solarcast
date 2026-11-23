"""Read-only AI overlay; never used as annotation or export input."""
import csv
from pathlib import Path

from core import REPO

DEFAULT_PREDICTIONS = REPO / 'data/solar_panel/conf_0p1'


class Predictions:
    def __init__(self, root=DEFAULT_PREDICTIONS):
        self.root = Path(root).resolve()
        summary = self.root / 'summary.csv'
        self.rows = {}
        if summary.is_file():
            with summary.open(encoding='utf-8-sig', newline='') as stream:
                self.rows = {row['key']: row for row in csv.DictReader(stream)}

    def describe(self, item):
        row = self.rows.get(item['id'])
        result = {'available': False, 'source': self.root.name, 'instances': 0}
        if not row:
            return {**result, 'reason': '이 이미지의 AI 예측 기록이 없습니다.'}
        if (int(row['width']), int(row['height'])) != (item['width'], item['height']):
            return {**result, 'reason': 'AI 예측과 원본의 크기가 다릅니다.'}
        path = self.path(item['id'])
        if not path.is_file():
            return {**result, 'reason': 'AI 예측 마스크 파일이 없습니다.'}
        return {**result, 'available': True, 'instances': int(row['instances']),
                'url': f"/predictions/{item['id']}"}

    def path(self, key):
        path = (self.root / self.rows[key]['mask_path']).resolve()
        path.relative_to(self.root)
        return path
