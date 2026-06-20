from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from .config import QualityConfig

try:  # pragma: no cover - depends on optional install.
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(slots=True)
class ImageMetrics:
    decode_ok: bool
    actual_width: int | None = None
    actual_height: int | None = None
    actual_ratio: float | None = None
    file_size_bytes: int = 0
    luma_std: float | None = None
    rgb_std_mean: float | None = None
    entropy: float | None = None
    laplacian_var_512: float | None = None
    edge_density_512: float | None = None
    failure_reason: str | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def score_image_bytes(data: bytes, cfg: QualityConfig, *, min_side: int, max_side: int, min_ratio: float, max_ratio: float) -> tuple[ImageMetrics, bool]:
    metrics = ImageMetrics(decode_ok=False, file_size_bytes=len(data))
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        metrics.failure_reason = f"decode_failed:{exc.__class__.__name__}"
        return metrics, True

    width, height = image.size
    metrics.decode_ok = True
    metrics.actual_width = width
    metrics.actual_height = height
    metrics.actual_ratio = width / height if height else None
    if width < min_side or height < min_side:
        metrics.failure_reason = "actual_dimensions_too_small"
        return metrics, True
    if max(width, height) > max_side:
        metrics.failure_reason = "actual_dimensions_too_large"
        return metrics, True
    if metrics.actual_ratio is None or not (min_ratio <= metrics.actual_ratio <= max_ratio):
        metrics.failure_reason = "actual_ratio_out_of_range"
        return metrics, True

    small = image.copy()
    small.thumbnail((512, 512), Image.Resampling.LANCZOS)
    rgb = np.asarray(small, dtype=np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    metrics.luma_std = float(np.std(luma))
    metrics.rgb_std_mean = float(np.mean(np.std(rgb, axis=(0, 1))))
    metrics.entropy = _entropy(luma)
    metrics.laplacian_var_512 = _laplacian_var(luma)
    metrics.edge_density_512 = _edge_density(small, luma)

    if metrics.luma_std < cfg.monotone_luma_std or metrics.entropy < cfg.monotone_entropy:
        metrics.failure_reason = "near_monotone"
        return metrics, True
    if metrics.laplacian_var_512 < cfg.blurry_laplacian_var and metrics.edge_density_512 < cfg.blurry_edge_density:
        metrics.failure_reason = "very_blurry"
        return metrics, True
    return metrics, False


def _entropy(luma: np.ndarray) -> float:
    hist, _ = np.histogram(luma, bins=256, range=(0, 255), density=False)
    probs = hist.astype(np.float64)
    probs = probs[probs > 0] / max(1, probs.sum())
    return float(-np.sum(probs * np.log2(probs)))


def _laplacian_var(luma: np.ndarray) -> float:
    if cv2 is not None:
        return float(cv2.Laplacian(luma.astype(np.float32), cv2.CV_32F).var())
    padded = np.pad(luma, 1, mode="edge")
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def _edge_density(image: Image.Image, luma: np.ndarray) -> float:
    if cv2 is not None:
        edges = cv2.Canny(luma.astype(np.uint8), 80, 160)
        return float(np.mean(edges > 0))
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.mean(np.asarray(edges) > 32))
