from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


GIB = 1024**3
MIB = 1024**2


@dataclass(slots=True)
class DatasetConfig:
    repo_id: str = "common-canvas/commoncatalog-cc-by"
    split: str = "train"
    source_uri_prefix: str = "hf://datasets/common-canvas/commoncatalog-cc-by"
    hf_token_env: str = "HF_TOKEN"
    top_dirs: list[str] = field(default_factory=lambda: [str(i) for i in range(10)])
    least_dim_ranges: list[str] = field(default_factory=lambda: ["1024-2048", "2048-4096", "4096+"])
    min_side: int = 1024
    max_side: int = 3072
    min_ratio: float = 0.9
    max_ratio: float = 1.1


@dataclass(slots=True)
class OutputConfig:
    root: Path = Path("./out/commoncatalog_sft")
    target_image_bytes: int = 200 * GIB
    metadata_rows_per_part: int = 2048
    metadata_bytes_per_part: int = 2 * GIB


@dataclass(slots=True)
class DownloadConfig:
    concurrency: int = 64
    queue_size: int = 512
    timeout_seconds: float = 30.0
    retries: int = 3
    max_file_bytes: int = 20 * MIB
    user_agent: str = "ImGenMagaCommonCatalogBuilder/0.1"


@dataclass(slots=True)
class QualityConfig:
    monotone_luma_std: float = 8.0
    monotone_entropy: float = 2.0
    blurry_laplacian_var: float = 30.0
    blurry_edge_density: float = 0.01


@dataclass(slots=True)
class BadPoolConfig:
    target_ratio: float = 0.075
    hard_cap_ratio: float = 0.10
    rng_seed: int = 20260620
    synthetic_noise_count: int = 6


@dataclass(slots=True)
class BuilderConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    bad_pool: BadPoolConfig = field(default_factory=BadPoolConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    base = cls()
    if not values:
        return base
    valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {key: value for key, value in values.items() if key in valid}
    if cls is OutputConfig and "root" in kwargs:
        kwargs["root"] = Path(kwargs["root"])
    return cls(**{**asdict(base), **kwargs})


def load_config(path: str | Path) -> BuilderConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return BuilderConfig(
        dataset=_merge_dataclass(DatasetConfig, raw.get("dataset")),
        output=_merge_dataclass(OutputConfig, raw.get("output")),
        download=_merge_dataclass(DownloadConfig, raw.get("download")),
        quality=_merge_dataclass(QualityConfig, raw.get("quality")),
        bad_pool=_merge_dataclass(BadPoolConfig, raw.get("bad_pool")),
    )
