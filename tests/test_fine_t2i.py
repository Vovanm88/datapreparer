from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from data.prepare_sft_dataset.fine_t2i_config import (
    FineDatasetConfig,
    FineDownloadConfig,
    FineFolderConfig,
    FineOutputConfig,
    FineSamplingConfig,
    FineT2IConfig,
)
from data.prepare_sft_dataset.fine_t2i_quota import estimate_folder_quotas
from data.prepare_sft_dataset.fine_t2i_tar import choose_caption, iter_webdataset_samples
from data.prepare_sft_dataset.fine_t2i_validate import validate_fine_t2i
from data.prepare_sft_dataset.fine_t2i_writer import FineWebDatasetWriter


def test_fine_quota_ratios_by_sample_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, target_bytes=1000)
    quotas = estimate_folder_quotas(cfg)
    total = sum(quotas.values())
    assert quotas["curated"] == round(total * 0.5)
    assert quotas["synthetic_original_prompt_square_resolution"] == round(total * 0.3)
    assert quotas["synthetic_enhanced_prompt_square_resolution"] == total - quotas["curated"] - quotas["synthetic_original_prompt_square_resolution"]


def test_choose_caption_priority_and_fallback() -> None:
    caption, source, _ = choose_caption(b'{"enhanced_prompt": "better", "prompt": "base"}', b"txt")
    assert caption == "base"
    assert source == "json.prompt"
    caption, source, _ = choose_caption(b"{}", b"from txt")
    assert caption == "from txt"
    assert source == "txt"


def test_iter_webdataset_samples_groups_triplets() -> None:
    tar_bytes = _fixture_tar({"a": {"json": {"prompt": "hello"}, "txt": "fallback", "jpg": b"jpgbytes"}})
    samples = list(iter_webdataset_samples(io.BytesIO(tar_bytes), shuffle_buffer=2, seed="x"))
    assert len(samples) == 1
    assert samples[0].key == "a"
    assert samples[0].image_bytes == b"jpgbytes"


def test_fine_writer_and_validator(tmp_path: Path) -> None:
    writer = FineWebDatasetWriter(tmp_path, shard_bytes=1, start_part=0, start_manifest_part=0)
    full = writer.add_sample(
        "sample0",
        "jpg",
        b"image",
        b"caption",
        b'{"prompt":"caption"}',
        {
            "image_ext": "jpg",
            "caption": "caption",
            "caption_source": "json.prompt",
            "source_folder": "curated",
            "source_tar": "curated/train-000000.tar",
            "source_key": "a",
            "image_bytes": 5,
            "json_bytes": 20,
            "txt_bytes": 7,
            "written_at": "now",
        },
        "source-id",
    )
    assert full
    writer.finalize_current()
    assert validate_fine_t2i(tmp_path) == []


def _cfg(tmp_path: Path, target_bytes: int) -> FineT2IConfig:
    return FineT2IConfig(
        dataset=FineDatasetConfig(
            folders={
                "curated": FineFolderConfig(0.5, 10, 1000),
                "synthetic_original_prompt_square_resolution": FineFolderConfig(0.3, 10, 1000),
                "synthetic_enhanced_prompt_square_resolution": FineFolderConfig(0.2, 10, 1000),
            }
        ),
        output=FineOutputConfig(root=tmp_path, target_bytes=target_bytes, tar_shard_bytes=100),
        sampling=FineSamplingConfig(rng_seed=1, shuffle_buffer=2),
        download=FineDownloadConfig(),
    )


def _fixture_tar(samples: dict[str, dict[str, object]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for key, parts in samples.items():
            _add(tar, f"{key}.jpg", parts["jpg"])
            _add(tar, f"{key}.txt", str(parts["txt"]).encode())
            _add(tar, f"{key}.json", json.dumps(parts["json"]).encode())
    return buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
