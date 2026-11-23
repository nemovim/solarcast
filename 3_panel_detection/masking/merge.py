"""Validate and combine four exported part ZIPs without overwriting existing work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

from core import ASSIGNMENTS, COMPLETE, REPO, SCHEMA, Store, atomic_json, mask_png, png_pixels


def collect(paths, store, require_complete=False):
    records, parts = {}, set()
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"ZIP 안에 중복 파일이 있습니다: {path}")
            if any(info.file_size > 2_000_000 for info in archive.infolist()) or sum(i.file_size for i in archive.infolist()) > 200_000_000:
                raise ValueError(f"ZIP 내용이 허용 크기를 초과합니다: {path}")
            manifest = json.loads(archive.read("manifest.json"))
            part = manifest.get("part")
            if type(part) is not int or part not in (1, 2, 3, 4) or part in parts:
                raise ValueError(f"잘못되었거나 중복된 파트입니다: {part}")
            if manifest.get("schema") != SCHEMA or manifest.get("assignment_sha256") != store.assignment_hash:
                raise ValueError(f"배정 목록/저장 형식이 다릅니다: {path}")
            parts.add(part)
            expected = {item["id"] for item in store.items if item["part"] == part}
            entries = manifest.get("images", [])
            if {entry["id"] for entry in entries} != expected or len(entries) != len(expected) or manifest.get("total") != len(expected):
                raise ValueError(f"파트 {part}의 이미지 목록에 누락 또는 중복이 있습니다.")
            status_by_id = {entry["id"]: entry["status"] for entry in entries}
            expected_names = {"manifest.json"}
            complete_count = 0
            for key in sorted(expected):
                name = f"annotations/{key}.json"
                expected_names.add(name)
                record = json.loads(archive.read(name))
                item = store.by_id[key]
                for field, value in (("schema", SCHEMA), ("assignment_sha256", store.assignment_hash),
                                     ("image_id", key), ("part", part), ("width", item["width"]), ("height", item["height"])):
                    if record.get(field) != value:
                        raise ValueError(f"{key}: {field} 값이 일치하지 않습니다.")
                validated = store.validate(key, record)
                if record.get("status") != status_by_id[key]:
                    raise ValueError(f"{key}: 상태 목록과 작업 기록이 다릅니다.")
                if type(record.get("revision")) is not int or record["revision"] < 0:
                    raise ValueError(f"{key}: 잘못된 저장 버전입니다.")
                if validated["status"] in COMPLETE:
                    complete_count += 1
                    mask_name = f"masks/{key}.png"
                    expected_names.add(mask_name)
                    expected_png = mask_png(item["width"], item["height"], validated["polygons"])
                    if png_pixels(archive.read(mask_name), item["width"], item["height"]) != png_pixels(expected_png, item["width"], item["height"]):
                        raise ValueError(f"{key}: 마스크 크기/내용이 좌표와 일치하지 않습니다.")
                records[key] = record
            if manifest.get("complete") != complete_count:
                raise ValueError(f"파트 {part}: 완료 개수가 일치하지 않습니다.")
            if set(names) != expected_names:
                raise ValueError(f"파트 {part}: 예상하지 않은 파일 또는 누락 파일이 있습니다.")
    if parts != {1, 2, 3, 4}:
        raise ValueError(f"네 파트가 모두 필요합니다. 누락 파트: {sorted({1, 2, 3, 4} - parts)}")
    unfinished = [key for key, record in records.items() if record["status"] not in COMPLETE]
    if require_complete and unfinished:
        raise ValueError(f"미완료 이미지 {len(unfinished)}장이 있습니다. 예: {unfinished[:5]}")
    return records, {"schema": SCHEMA, "total": len(records), "complete": len(records) - len(unfinished),
                     "unfinished": unfinished, "assignment_sha256": store.assignment_hash}


def merge(paths, output, assignments=ASSIGNMENTS, require_complete=False):
    output = Path(output)
    if output.exists():
        raise ValueError("출력 폴더가 이미 있습니다. 기존 결과 보호를 위해 새 폴더를 지정하세요.")
    store = Store(assignments=assignments, workspace=output)
    records, report = collect(paths, store, require_complete)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    for key, record in records.items():
        atomic_json(output / "annotations" / f"{key}.json", record)
        if record["status"] in COMPLETE:
            destination = output / "masks" / f"{key}.png"
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(mask_png(record["width"], record["height"], record["polygons"]))
    return report


def main():
    parser = argparse.ArgumentParser(description="네 파트 ZIP 검증·수합")
    parser.add_argument("archives", nargs=4, type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "data/bitgaram/annotations_merged")
    parser.add_argument("--require-complete", action="store_true", help="미완료 이미지가 있으면 수합하지 않음")
    args = parser.parse_args()
    try:
        report = merge(args.archives, args.output, require_complete=args.require_complete)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        sys.exit(f"수합 실패: {error}")
    print(f"수합 완료: {report['total']}장 / 완료 {report['complete']}장 / 미완료 {len(report['unfinished'])}장\n{args.output.resolve()}")


if __name__ == "__main__":
    main()
