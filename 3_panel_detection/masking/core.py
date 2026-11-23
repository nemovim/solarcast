"""Fresh four-part annotation format. Python 3.10+, standard library only."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from threading import RLock
from datetime import datetime, timezone
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ASSIGNMENTS = ROOT / "assignments.json"
IMAGES = REPO / "data/bitgaram/raw_tiles/z19"
WORKSPACE = REPO / "data/bitgaram/annotations_v2"
STATUSES = {"pending", "in_progress", "done", "empty", "deferred"}
COMPLETE = {"done", "empty"}
SCHEMA = 2


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, c):
    return (abs(cross(a, b, c)) < 1e-8
            and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


def intersects(a, b, c, d):
    ca, cb, cc, cd = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return (ca * cb < 0 and cc * cd < 0) or any((
        on_segment(a, b, c), on_segment(a, b, d),
        on_segment(c, d, a), on_segment(c, d, b)))


def validate_polygons(polygons, width, height):
    if not isinstance(polygons, list) or len(polygons) > 500:
        raise ValueError("영역 목록이 올바르지 않습니다 (최대 500개).")
    result = []
    for polygon in polygons:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 100:
            raise ValueError("영역마다 꼭짓점이 3~100개 필요합니다.")
        points = []
        for point in polygon:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
                raise ValueError("꼭짓점 좌표가 올바르지 않습니다.")
            x, y = point
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError("꼭짓점이 이미지 밖에 있습니다.")
            points.append([round(x, 4), round(y, 4)])
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError("같은 위치의 꼭짓점이 중복되어 있습니다.")
        area = abs(sum(p[0] * q[1] - q[0] * p[1]
                       for p, q in zip(points, points[1:] + points[:1]))) / 2
        if area < 0.5:
            raise ValueError("영역이 너무 작거나 꼭짓점 순서가 잘못되었습니다.")
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if intersects(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                    raise ValueError("영역의 변이 교차합니다. 경계를 따라 순서대로 찍어 주세요.")
        result.append(points)
    return result


def mask_png(width, height, polygons):
    """Rasterize pixel centers, union all polygons; output 8-bit 0/255 grayscale PNG."""
    pixels = [bytearray(width) for _ in range(height)]
    for points in polygons:
        edges = list(zip(points, points[1:] + points[:1]))
        for y in range(max(0, math.floor(min(p[1] for p in points))),
                       min(height, math.ceil(max(p[1] for p in points)))):
            scan_y = y + 0.5
            hits = sorted(a[0] + (scan_y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                          for a, b in edges if (a[1] <= scan_y < b[1]) or (b[1] <= scan_y < a[1]))
            for left, right in zip(hits[::2], hits[1::2]):
                start, end = max(0, math.ceil(left - 0.5)), min(width, math.ceil(right - 0.5))
                if end > start:
                    pixels[y][start:end] = b"\xff" * (end - start)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row for row in pixels))) + chunk(b"IEND", b""))


def png_pixels(data, width, height):
    """Read this format's grayscale PNG pixels independently of zlib version/compression."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("PNG 형식이 올바르지 않습니다.")
    position, compressed, saw_header, saw_end = 8, b"", False, False
    while position + 12 <= len(data):
        size = struct.unpack(">I", data[position:position + 4])[0]
        kind, payload = data[position + 4:position + 8], data[position + 8:position + 8 + size]
        if position + 12 + size > len(data):
            raise ValueError("잘린 PNG 파일입니다.")
        crc = struct.unpack(">I", data[position + 8 + size:position + 12 + size])[0]
        if zlib.crc32(kind + payload) != crc:
            raise ValueError("PNG 체크섬이 일치하지 않습니다.")
        if kind == b"IHDR":
            if payload != struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0):
                raise ValueError("마스크 크기/색상 형식이 올바르지 않습니다.")
            saw_header = True
        elif kind == b"IDAT":
            compressed += payload
        elif kind == b"IEND":
            saw_end = True
        position += size + 12
    if not saw_header or not saw_end or position != len(data):
        raise ValueError("PNG 구조가 올바르지 않습니다.")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, (width + 1) * height + 1)
    except zlib.error as error:
        raise ValueError("PNG 압축 데이터가 손상되었습니다.") from error
    if len(raw) != (width + 1) * height or not decoder.eof or decoder.unused_data:
        raise ValueError("마스크 픽셀 수가 올바르지 않습니다.")
    if any(raw[y * (width + 1)] != 0 for y in range(height)):
        raise ValueError("이 도구에서 내보낸 PNG 형식이 아닙니다.")
    return raw


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, assignments=ASSIGNMENTS, images=IMAGES, workspace=WORKSPACE):
        self.assignment_bytes = Path(assignments).read_bytes()
        self.assignment = json.loads(self.assignment_bytes)
        canonical = json.dumps(self.assignment, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assignment_hash = hashlib.sha256(canonical).hexdigest()
        self.items = self.assignment["images"]
        self.by_id = {item["id"]: item for item in self.items}
        if len(self.by_id) != len(self.items):
            raise ValueError("배정 목록에 중복 이미지가 있습니다.")
        self.images, self.workspace = Path(images), Path(workspace)
        self.lock = RLock()
        for item in self.items:
            if (Path(item["file"]).name != item["file"] or item["id"] != Path(item["file"]).stem
                    or item["part"] not in (1, 2, 3, 4)):
                raise ValueError("배정 목록이 올바르지 않습니다.")
        marker = self.workspace / "assignment.json"
        if marker.exists() and json.loads(marker.read_text("utf-8"))["sha256"] != self.assignment_hash:
            raise ValueError("이 저장 폴더는 다른 배정 목록을 사용합니다. 별도 --workspace를 지정하세요.")
        # Startup and read-only inspection never create work records.

    def image_path(self, key):
        return self.images / self.by_id[key]["file"]

    def record_path(self, key):
        item = self.by_id[key]
        return self.workspace / f"part-{item['part']}" / f"{key}.json"

    def blank(self, key):
        item = self.by_id[key]
        return {"schema": SCHEMA, "assignment_sha256": self.assignment_hash,
                "image_id": key, "part": item["part"], "width": item["width"], "height": item["height"],
                "status": "pending", "polygons": [], "note": "", "revision": 0, "updated_at": None}

    def read(self, key):
        with self.lock:
            path = self.record_path(key)
            return json.loads(path.read_text("utf-8")) if path.exists() else self.blank(key)

    def validate(self, key, data):
        item = self.by_id[key]
        if not isinstance(data, dict) or data.get("part") != item["part"]:
            raise ValueError("이미지의 담당 파트가 일치하지 않습니다.")
        if data.get("assignment_sha256") != self.assignment_hash:
            raise ValueError("배정 목록이 변경되었습니다. 서버와 브라우저를 다시 열어 주세요.")
        status = data.get("status")
        if status not in STATUSES:
            raise ValueError("알 수 없는 작업 상태입니다.")
        polygons = validate_polygons(data.get("polygons"), item["width"], item["height"])
        if status == "done" and not polygons:
            raise ValueError("영역이 없습니다. 패널이 없다면 '패널 없음'을 선택하세요.")
        if status == "empty" and polygons:
            raise ValueError("영역이 있는 이미지를 패널 없음으로 저장할 수 없습니다.")
        note = data.get("note", "")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("메모가 너무 길거나 올바르지 않습니다.")
        return {**self.blank(key), "status": status, "polygons": polygons, "note": note}

    def save(self, key, data):
        with self.lock:
            record = self.validate(key, data)
            old = self.read(key)
            if type(data.get("revision")) is not int or data["revision"] != old["revision"]:
                raise Conflict("다른 창에서 수정한 기록이 있습니다. 초안을 내려받고 새로고침해 주세요.")
            record.update(revision=old["revision"] + 1, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(self.workspace / "assignment.json", {"sha256": self.assignment_hash, "schema": SCHEMA})
            atomic_json(self.record_path(key), record)
            return record

    def listing(self, part):
        with self.lock:
            return [{**item, "status": self.read(item["id"])["status"]}
                    for item in self.items if item["part"] == part]

    def export(self, part):
        with self.lock:
            records = [self.read(item["id"]) for item in self.items if item["part"] == part]
        if not records:
            raise ValueError("파트는 1~4입니다.")
        manifest = {"schema": SCHEMA, "assignment_sha256": self.assignment_hash, "part": part,
                    "total": len(records), "complete": sum(r["status"] in COMPLETE for r in records),
                    "images": [{"id": r["image_id"], "status": r["status"]} for r in records]}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json_bytes(manifest))
            for record in records:
                key = record["image_id"]
                archive.writestr(f"annotations/{key}.json", json_bytes(record))
                if record["status"] in COMPLETE:
                    archive.writestr(f"masks/{key}.png", mask_png(record["width"], record["height"], record["polygons"]))
        return buffer.getvalue()
