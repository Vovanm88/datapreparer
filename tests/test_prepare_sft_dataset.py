from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl

from data.prepare_sft_dataset.config import BuilderConfig, OutputConfig, QualityConfig
from data.prepare_sft_dataset.metadata import get_image_bytes, is_candidate, stable_sample_id, strip_caption
from data.prepare_sft_dataset.pipeline import _process_row
from data.prepare_sft_dataset.quality import score_image_bytes
from data.prepare_sft_dataset.shards import SourceShard, is_near_square_bucket, parse_aspect_ratio_bucket
from data.prepare_sft_dataset.synthetic import generate_synthetic_bad, png_bytes


class _FailingDownloader:
    async def fetch(self, url: str):  # pragma: no cover - should never be called.
        raise AssertionError("downloader should not be used when jpg.bytes exists")


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


def test_candidate_accepts_hf_image_bytes_without_url() -> None:
    row = {
        "width": 1600,
        "height": 1590,
        "blip2_caption": "a photo",
        "jpg": {"bytes": b"abc", "path": "image.jpg"},
    }
    assert is_candidate(row, min_side=1024, max_side=3072, min_ratio=0.9, max_ratio=1.1)
    assert get_image_bytes(row) == b"abc"
    assert stable_sample_id(row).startswith("jpg_bytes:")


def test_candidate_accepts_binary_jpg_column_without_url() -> None:
    row = {
        "width": 1600,
        "height": 1590,
        "blip2_caption": "a photo",
        "jpg": b"abc",
    }
    assert is_candidate(row, min_side=1024, max_side=3072, min_ratio=0.9, max_ratio=1.1)
    assert get_image_bytes(row) == b"abc"


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


def test_process_row_prefers_hf_image_bytes(tmp_path: Path) -> None:
    row = {
        "width": 1024,
        "height": 1024,
        "blip2_caption": "blue",
        "jpg": png_bytes((0, 0, 255)),
        "ext": "png",
        "sha256": "sample",
    }
    cfg = BuilderConfig(output=OutputConfig(root=tmp_path))
    shard = SourceShard("x.parquet", "hf://x.parquet", "0", "1024-2048", "aspect_ratio_bucket=1-1")

    result = asyncio.run(_process_row(cfg, row, None, "sha256:sample", shard, 0, _FailingDownloader()))

    assert result is not None
    assert result["ok"]
    assert result["image_source"] == "hf_parquet_jpg_bytes"
    assert (tmp_path / result["row"]["rel_path"]).exists()


def test_local_probe_parquet_jpg_is_binary_if_present() -> None:
    probe_root = Path("tmp/commoncatalog_probe")
    parquet_files = list(probe_root.rglob("*.parquet")) if probe_root.exists() else []
    if not parquet_files:
        return
    row = pl.read_parquet(parquet_files[0], n_rows=1).to_dicts()[0]
    image_bytes = get_image_bytes(row)
    assert image_bytes is not None
    assert image_bytes[:2] == b"\xff\xd8"
