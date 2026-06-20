from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

from data.prepare_sft_dataset.config import QualityConfig
from data.prepare_sft_dataset.metadata import is_candidate, strip_caption
from data.prepare_sft_dataset.quality import score_image_bytes
from data.prepare_sft_dataset.shards import is_near_square_bucket, parse_aspect_ratio_bucket
from data.prepare_sft_dataset.synthetic import generate_synthetic_bad, png_bytes


def test_aspect_ratio_bucket_parsing() -> None:
    assert float(parse_aspect_ratio_bucket("aspect_ratio_bucket=15-16")) == 15 / 16
    assert is_near_square_bucket("aspect_ratio_bucket=1-1", 0.9, 1.1)
    assert not is_near_square_bucket("aspect_ratio_bucket=2-1", 0.9, 1.1)


def test_candidate_filter_and_caption_drop() -> None:
    row = {
        "width": 1600,
        "height": 1590,
        "blip2_caption": "a photo",
        "caption": "trash",
        "url": "https://example.com/a.jpg",
    }
    assert is_candidate(row, min_side=1024, max_side=3072, min_ratio=0.9, max_ratio=1.1)
    assert "caption" not in strip_caption(row)


def test_candidate_filter_rejects_oversize() -> None:
    row = {"width": 4096, "height": 4096, "blip2_caption": "x", "url": "https://example.com/a.jpg"}
    assert not is_candidate(row, min_side=1024, max_side=3072, min_ratio=0.9, max_ratio=1.1)


def test_solid_color_scores_bad() -> None:
    data = png_bytes((0, 0, 255))
    metrics, is_bad = score_image_bytes(
        data,
        QualityConfig(),
        min_side=1024,
        max_side=3072,
        min_ratio=0.9,
        max_ratio=1.1,
    )
    assert metrics.decode_ok
    assert is_bad
    assert metrics.failure_reason == "near_monotone"


def test_synthetic_generation(tmp_path: Path) -> None:
    rows = generate_synthetic_bad(tmp_path, noise_count=2, seed=1)
    assert len(rows) == 11
    assert all(row["is_synthetic"] for row in rows)
    assert all((tmp_path / str(row["rel_path"])).exists() for row in rows)
