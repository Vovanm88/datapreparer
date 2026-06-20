from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


GIB = 1024**3


@dataclass(slots=True)
class FineFolderConfig:
    ratio: float
    published_samples: int
    published_bytes: int


@dataclass(slots=True)
class FineDatasetConfig:
    repo_id: str = "ma-xu/fine-t2i"
    hf_token_env: str = "HF_TOKEN"
    folders: dict[str, FineFolderConfig] = field(default_factory=dict)


@dataclass(slots=True)
class FineOutputConfig:
    root: Path = Path("./out/fine_t2i_sample")
    target_bytes: int = 100 * GIB
    tar_shard_bytes: int = GIB


@dataclass(slots=True)
class FineSamplingConfig:
    rng_seed: int = 20260620
    shuffle_buffer: int = 2048


@dataclass(slots=True)
class FineDownloadConfig:
    timeout_seconds: float = 60.0
    user_agent: str = "ImGenMagaFineT2ISampler/0.1"


@dataclass(slots=True)
class FineT2IConfig:
    dataset: FineDatasetConfig
    output: FineOutputConfig = field(default_factory=FineOutputConfig)
    sampling: FineSamplingConfig = field(default_factory=FineSamplingConfig)
    download: FineDownloadConfig = field(default_factory=FineDownloadConfig)


def load_fine_t2i_config(path: str | Path) -> FineT2IConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    dataset_raw = raw.get("dataset") or {}
    folders = {
        name: FineFolderConfig(**value)
        for name, value in (dataset_raw.get("folders") or {}).items()
    }
    dataset = FineDatasetConfig(
        repo_id=dataset_raw.get("repo_id", "ma-xu/fine-t2i"),
        hf_token_env=dataset_raw.get("hf_token_env", "HF_TOKEN"),
        folders=folders,
    )
    output = _merge(FineOutputConfig, raw.get("output"))
    if isinstance(output.root, str):
        output.root = Path(output.root)
    return FineT2IConfig(
        dataset=dataset,
        output=output,
        sampling=_merge(FineSamplingConfig, raw.get("sampling")),
        download=_merge(FineDownloadConfig, raw.get("download")),
    )


def _merge(cls: type, values: dict[str, Any] | None):
    base = cls()
    if not values:
        return base
    return cls(**{**asdict(base), **values})
