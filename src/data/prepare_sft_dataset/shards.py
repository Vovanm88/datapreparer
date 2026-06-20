from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath

from huggingface_hub import HfApi

from .auth import resolve_hf_token
from .config import DatasetConfig


@dataclass(frozen=True, slots=True)
class SourceShard:
    repo_path: str
    uri: str
    top_dir: str
    least_dim_range: str
    aspect_ratio_bucket: str


def parse_aspect_ratio_bucket(name: str) -> Fraction:
    prefix = "aspect_ratio_bucket="
    if name.startswith(prefix):
        name = name[len(prefix) :]
    left, right = name.split("-", 1)
    return Fraction(int(left), int(right))


def is_near_square_bucket(name: str, min_ratio: float, max_ratio: float) -> bool:
    ratio = float(parse_aspect_ratio_bucket(name))
    return min_ratio <= ratio <= max_ratio


def parquet_uri(prefix: str, repo_path: str) -> str:
    return f"{prefix.rstrip('/')}/{repo_path}"


def discover_commoncatalog_shards(cfg: DatasetConfig, limit: int | None = None) -> list[SourceShard]:
    api = HfApi(token=resolve_hf_token(cfg.hf_token_env))
    shards: list[SourceShard] = []
    for top_dir in cfg.top_dirs:
        for least_range in cfg.least_dim_ranges:
            base = f"{top_dir}/least_dim_range={least_range}"
            try:
                aspect_entries = list(api.list_repo_tree(cfg.repo_id, path_in_repo=base, repo_type="dataset"))
            except Exception:
                continue
            aspect_dirs = sorted(
                entry.path for entry in aspect_entries if getattr(entry, "path", "").split("/")[-1].startswith("aspect_ratio_bucket=")
            )
            for aspect_path in aspect_dirs:
                aspect_bucket = PurePosixPath(aspect_path).name
                if not is_near_square_bucket(aspect_bucket, cfg.min_ratio, cfg.max_ratio):
                    continue
                try:
                    file_entries = list(api.list_repo_tree(cfg.repo_id, path_in_repo=aspect_path, repo_type="dataset"))
                except Exception:
                    continue
                for entry in sorted(file_entries, key=lambda item: item.path):
                    repo_path = getattr(entry, "path", "")
                    if not repo_path.endswith(".parquet"):
                        continue
                    shards.append(
                        SourceShard(
                            repo_path=repo_path,
                            uri=parquet_uri(cfg.source_uri_prefix, repo_path),
                            top_dir=top_dir,
                            least_dim_range=least_range,
                            aspect_ratio_bucket=aspect_bucket,
                        )
                    )
                    if limit is not None and len(shards) >= limit:
                        return shards
    return shards
