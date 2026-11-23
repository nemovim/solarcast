"""Run: python masking/server.py --open"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import urlparse
import webbrowser

from core import ASSIGNMENTS, IMAGES, WORKSPACE, ROOT, Conflict, Store, json_bytes
from predictions import Predictions


def make_handler(store, predictions=None):
    predictions = predictions if predictions is not None else Predictions()
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type="application/json; charset=utf-8", filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def check_origin(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            allowed_hosts = {expected, f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed_hosts:
                raise ValueError("로컬 주소로 접속해 주세요.")
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed_hosts}:
                raise ValueError("다른 사이트에서 보낸 요청은 허용하지 않습니다.")

        def do_GET(self):
            try:
                self.check_origin()
                path = urlparse(self.path).path
                if path == "/api/config":
                    result = {"schema": 2, "assignment_sha256": store.assignment_hash,
                              "total": len(store.items), "parts": [
                                  {"part": part, "total": sum(i["part"] == part for i in store.items)}
                                  for part in range(1, 5)]}
                    return self.reply(200, json_bytes(result))
                if path.startswith("/api/parts/"):
                    part = int(path.rsplit("/", 1)[1])
                    if part not in (1, 2, 3, 4):
                        raise ValueError("파트는 1~4입니다.")
                    return self.reply(200, json_bytes(store.listing(part)))
                if path.startswith("/api/predictions/"):
                    key = path.rsplit("/", 1)[1]
                    return self.reply(200, json_bytes(predictions.describe(store.by_id[key])))
                if path.startswith("/predictions/"):
                    key = path.rsplit("/", 1)[1]
                    metadata = predictions.describe(store.by_id[key])
                    if not metadata['available']:
                        return self.reply(404, json_bytes({'error': metadata['reason']}))
                    return self.reply(200, predictions.path(key).read_bytes(), 'image/png')
                if path.startswith("/api/records/"):
                    return self.reply(200, json_bytes(store.read(path.rsplit("/", 1)[1])))
                if path.startswith("/api/export/"):
                    part = int(path.rsplit("/", 1)[1])
                    return self.reply(200, store.export(part), "application/zip", f"part-{part}.zip")
                if path.startswith("/images/"):
                    file = store.image_path(path.rsplit("/", 1)[1])
                    return self.reply(200, file.read_bytes(), "image/jpeg")
                files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css", "/geometry.js": "geometry.js"}
                if path not in files:
                    return self.reply(404, json_bytes({"error": "찾을 수 없는 주소입니다."}))
                name = files[path]
                content_type = "text/javascript" if name.endswith(".js") else mimetypes.guess_type(name)[0]
                return self.reply(200, (ROOT / "web" / name).read_bytes(), content_type + "; charset=utf-8")
            except (ValueError, KeyError) as error:
                self.reply(400, json_bytes({"error": str(error)}))
            except OSError as error:
                self.reply(500, json_bytes({"error": f"파일을 읽지 못했습니다: {error}"}))

        def do_POST(self):
            try:
                self.check_origin()
                path = urlparse(self.path).path
                if not path.startswith("/api/records/"):
                    return self.reply(404, json_bytes({"error": "찾을 수 없는 주소입니다."}))
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("요청 크기가 올바르지 않습니다.")
                data = json.loads(self.rfile.read(length))
                record = store.save(path.rsplit("/", 1)[1], data)
                self.reply(200, json_bytes(record))
            except Conflict as error:
                self.reply(409, json_bytes({"error": str(error)}))
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, json_bytes({"error": str(error)}))
            except OSError as error:
                self.reply(500, json_bytes({"error": f"저장 실패. 창을 닫지 말고 다시 시도해 주세요: {error}"}))

    return Handler


def main():
    parser = argparse.ArgumentParser(description="빛가람동 4인 마스킹 도구 (로컬 전용)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    store = Store(ASSIGNMENTS, args.images, args.workspace)
    missing = [item["file"] for item in store.items if not store.image_path(item["id"]).is_file()]
    if missing:
        sys.exit(f"원본 이미지 {len(missing)}장이 없습니다. --images 경로를 확인하세요. 예: {missing[:3]}")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    except OSError as error:
        sys.exit(f"서버를 시작하지 못했습니다: {error}\n다른 포트를 쓰려면 --port 8766을 지정하세요.")
    print(f"Local: http://127.0.0.1:{server.server_port}\nImages: {len(store.items)}\nSave: {store.workspace}", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
