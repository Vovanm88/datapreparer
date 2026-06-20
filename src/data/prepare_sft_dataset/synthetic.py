from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


CORE_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
}


def generate_synthetic_bad(root: Path, *, noise_count: int, seed: int) -> list[dict[str, object]]:
    out_dir = root / "images" / "bad" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name, rgb in CORE_COLORS.items():
        path = out_dir / f"solid_{name}.png"
        if not path.exists():
            Image.new("RGB", (1024, 1024), rgb).save(path)
        rows.append(_row(root, path, name))

    rng = np.random.default_rng(seed)
    for idx in range(noise_count):
        if idx % 2 == 0:
            arr = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
            caption = "white noise"
            stem = "white_noise"
        else:
            base = rng.integers(96, 160, size=(1024, 1024, 3), dtype=np.uint8)
            arr = base
            caption = "low contrast noise"
            stem = "low_contrast_noise"
        path = out_dir / f"{stem}_{idx:03d}.png"
        if not path.exists():
            Image.fromarray(arr, "RGB").save(path)
        rows.append(_row(root, path, caption))
    return rows


def png_bytes(color: tuple[int, int, int], size: int = 1024) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _row(root: Path, path: Path, caption: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "rel_path": path.relative_to(root).as_posix(),
        "blip2_caption": caption,
        "quality_label": "synthetic_bad",
        "is_bad": True,
        "is_synthetic": True,
        "decode_ok": True,
        "actual_width": 1024,
        "actual_height": 1024,
        "actual_ratio": 1.0,
        "file_size_bytes": stat.st_size,
        "luma_std": 0.0 if caption in CORE_COLORS else None,
        "rgb_std_mean": 0.0 if caption in CORE_COLORS else None,
        "entropy": 0.0 if caption in CORE_COLORS else None,
        "laplacian_var_512": 0.0 if caption in CORE_COLORS else None,
        "edge_density_512": 0.0 if caption in CORE_COLORS else None,
        "failure_reason": "synthetic_bad",
        "source_parquet_path": None,
        "source_row_index": None,
        "download_attempts": 0,
        "processed_at": None,
    }
