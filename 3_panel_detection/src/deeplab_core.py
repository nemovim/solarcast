"""Reusable inference and geospatial post-processing utilities.

The module intentionally keeps model inference separate from tiling and export.
That makes the non-ML parts testable without downloading a 235 MB checkpoint.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class Segmenter(Protocol):
    """Small interface implemented by real and smoke-test predictors."""

    def predict_probability(self, image: Image.Image) -> np.ndarray:
        """Return a float32 probability map with shape ``(height, width)``."""


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class ONNXPanelSegmenter:
    """ONNX Runtime wrapper for the public multi-resolution DeepLabV3 model."""

    def __init__(self, model_path: str | Path, use_gpu: bool = True):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - user-facing dependency error
            raise RuntimeError("onnxruntime is required; run through run_deeplabv3.sh") from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if use_gpu and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        # Managed containers often expose the host's full CPU topology while
        # restricting the process to a smaller cpuset. ORT's default thread
        # affinity then targets unavailable cores and logs pthread errors.
        # An explicit positive thread count disables that automatic affinity.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=providers,
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def predict_probability(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        original_size = rgb.size

        shape = self.input.shape
        fixed_height = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else None
        fixed_width = shape[3] if len(shape) == 4 and isinstance(shape[3], int) else None
        if fixed_height and fixed_width and rgb.size != (fixed_width, fixed_height):
            prepared = rgb.resize((fixed_width, fixed_height), Image.Resampling.BILINEAR)
        else:
            prepared = rgb

        tensor = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        raw = np.asarray(self.session.run([self.output.name], {self.input.name: tensor})[0])
        while raw.ndim > 2:
            raw = raw[0]
        probability = raw.astype(np.float32)
        if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
            probability = _sigmoid(probability)

        if probability.shape != (original_size[1], original_size[0]):
            probability = np.asarray(
                Image.fromarray(probability, mode="F").resize(original_size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        return np.clip(probability, 0.0, 1.0)


@dataclass
class ColorRuleSegmenter:
    """Deterministic offline predictor used only to smoke-test the pipeline."""

    blue_margin: float = 1.0

    def predict_probability(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        red, green, blue = array[..., 0], array[..., 1], array[..., 2]
        score = (blue - np.maximum(red, green) - self.blue_margin) / 30.0
        return _sigmoid(score).astype(np.float32)


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def predict_tiled(
    image: Image.Image,
    segmenter: Segmenter,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Predict a large image by averaging overlapping square-tile predictions."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    probability_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for top in _starts(height, tile_size, overlap):
        for left in _starts(width, tile_size, overlap):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tile = rgb.crop((left, top, right, bottom))
            probability = np.asarray(segmenter.predict_probability(tile), dtype=np.float32)
            expected = (bottom - top, right - left)
            if probability.shape != expected:
                raise ValueError(f"Predictor returned {probability.shape}; expected {expected}")
            probability_sum[top:bottom, left:right] += probability
            count[top:bottom, left:right] += 1.0

    if np.any(count == 0):
        raise RuntimeError("Tiling left uncovered pixels")
    return probability_sum / count


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D array")
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[np.ndarray] = []
    current = 0

    for row, col in zip(*np.nonzero(binary)):
        if labels[row, col] != 0:
            continue
        current += 1
        queue = deque([(int(row), int(col))])
        labels[row, col] = current
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = current
                    queue.append((ny, nx))
        components.append(np.asarray(pixels, dtype=np.int32))
    return labels, components


def binary_mask(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_pixels: int = 20,
    closing_radius: int = 1,
) -> np.ndarray:
    """Threshold, optionally close small gaps, and remove tiny components."""
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("probability must be a 2D array")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1")

    image = Image.fromarray((probability >= threshold).astype(np.uint8) * 255, mode="L")
    if closing_radius > 0:
        kernel = closing_radius * 2 + 1
        image = image.filter(ImageFilter.MaxFilter(kernel)).filter(ImageFilter.MinFilter(kernel))
    candidate = np.asarray(image) > 0
    labels, components = _label_components(candidate)
    keep = np.zeros_like(candidate)
    for index, pixels in enumerate(components, start=1):
        if len(pixels) >= min_pixels:
            keep[labels == index] = True
    return keep


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay detected pixels in red on an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask shape must match image height and width")
    color = np.zeros_like(rgb)
    color[..., 0] = 255.0
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color[mask]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def evaluate_binary_mask(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Calculate robust binary segmentation metrics."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target masks must have the same shape")
    tp = int(np.logical_and(predicted, target).sum())
    fp = int(np.logical_and(predicted, ~target).sum())
    fn = int(np.logical_and(~predicted, target).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"precision": precision, "recall": recall, "iou": iou, "dice": dice}


def _pixel_rectangle_geometry(pixels: np.ndarray) -> dict:
    rows, cols = pixels[:, 0], pixels[:, 1]
    left, right = int(cols.min()), int(cols.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return {
        "type": "Polygon",
        "coordinates": [[[left, top], [right, top], [right, bottom], [left, bottom], [left, top]]],
    }


def components_to_geojson(
    mask: np.ndarray,
    probability: np.ndarray | None = None,
    *,
    transform=None,
    crs: str | None = None,
    output_crs: str = "EPSG:4326",
    source_name: str | None = None,
) -> dict:
    """Convert connected panel regions into a GeoJSON FeatureCollection.

    With ``transform=None``, coordinates remain pixels. With a Rasterio affine
    transform, true map coordinates are emitted and optionally reprojected.
    """
    binary = np.asarray(mask, dtype=bool)
    if probability is not None and np.asarray(probability).shape != binary.shape:
        raise ValueError("probability shape must match mask")
    labels, components = _label_components(binary)
    features = []

    rasterio_features = None
    transform_geom = None
    if transform is not None:
        try:
            from rasterio.features import shapes as raster_shapes
            from rasterio.warp import transform_geom as raster_transform_geom
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rasterio is required for georeferenced export") from exc
        rasterio_features = raster_shapes
        transform_geom = raster_transform_geom

    for index, pixels in enumerate(components, start=1):
        component_mask = labels == index
        if transform is None:
            geometry = _pixel_rectangle_geometry(pixels)
            coordinate_space = "pixel"
            area_map_units2 = None
        else:
            generated = list(
                rasterio_features(
                    component_mask.astype(np.uint8),
                    mask=component_mask,
                    transform=transform,
                )
            )
            geometry = generated[0][0]
            if crs and output_crs and crs.upper() != output_crs.upper():
                geometry = transform_geom(crs, output_crs, geometry, precision=7)
            coordinate_space = output_crs if crs else "map"
            area_map_units2 = abs(transform.a * transform.e - transform.b * transform.d) * len(pixels)

        confidence = float(np.asarray(probability)[component_mask].mean()) if probability is not None else None
        properties = {
            "panel_id": f"PV_{index:04d}",
            "pixel_count": int(len(pixels)),
            "mean_confidence": round(confidence, 6) if confidence is not None else None,
            "coordinate_space": coordinate_space,
            "source": source_name,
        }
        if area_map_units2 is not None:
            properties["area_map_units2"] = round(float(area_map_units2), 3)
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})

    return {"type": "FeatureCollection", "features": features}


def create_demo_image(size: tuple[int, int] = (768, 512)) -> tuple[Image.Image, np.ndarray]:
    """Create a deterministic roof-like image and its synthetic panel mask."""
    width, height = size
    image = Image.new("RGB", size, (188, 184, 174))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, width - 40, height - 40), fill=(205, 204, 196), outline=(90, 90, 90), width=4)
    draw.rectangle((width - 180, 70, width - 90, 145), fill=(150, 150, 148), outline=(80, 80, 80))
    draw.ellipse((70, height - 150, 150, height - 70), fill=(160, 160, 158), outline=(80, 80, 80))
    mask = np.zeros((height, width), dtype=bool)
    for top in (135, 205, 275):
        for left in (185, 280, 375, 470):
            box = (left, top, left + 78, top + 48)
            draw.rectangle(box, fill=(25, 55, 115), outline=(215, 225, 240), width=2)
            mask[top : top + 49, left : left + 79] = True
            for x in range(left + 19, left + 78, 20):
                draw.line((x, top, x, top + 48), fill=(100, 140, 190), width=1)
    return image, mask


def save_geojson(data: dict, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination
