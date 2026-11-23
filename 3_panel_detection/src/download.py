"""Download one of the model files used by the project."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
FILES = {
    "deeplabv3": (
        (
            "https://owncloud.fraunhofer.de/index.php/s/Ph9TC6BTxPi5oZZ/download",
            MODEL_DIR / "deeplabv3.onnx",
        ),
    ),
    "yolo": (
        (
            "https://huggingface.co/agademer/yolo-remote-sensing-photovoltaic/resolve/main/"
            "yolo-remote-sensing-photovoltaic-v26l-solar-farms-and-cities-"
            "v20260403-segment-300_epochs.pt",
            MODEL_DIR / "yolo.pt",
        ),
    ),
    "nlr": (
        (
            "https://raw.githubusercontent.com/NatLabRockies/Panel-Segmentation/"
            "master/panel_segmentation/models/panel_detection_config.py",
            MODEL_DIR / "nlr_config.py",
        ),
        (
            "https://media.githubusercontent.com/media/NatLabRockies/Panel-Segmentation/"
            "master/panel_segmentation/models/panel_detection_model.pth",
            MODEL_DIR / "nlr.pth",
        ),
    ),
}


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.is_file() and destination.stat().st_size > 0 and not force:
        print(f"Exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "solar-panel/1.0"})
    print(f"Downloading {url}\n        -> {destination}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r{received / total:6.1%}", end="", flush=True)
        print()
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(FILES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for url, destination in FILES[args.model]:
        download(url, destination, args.force)
    print(f"{args.model} model ready")


if __name__ == "__main__":
    main()
