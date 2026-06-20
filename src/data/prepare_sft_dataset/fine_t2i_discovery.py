from __future__ import annotations

import random
from dataclasses import dataclass

from huggingface_hub import HfApi

from .auth import resolve_hf_token
from .fine_t2i_config import FineT2IConfig


@dataclass(frozen=True, slots=True)
class FineTarShard:
    folder: str
    repo_path: str
    url: str
    size_bytes: int | None


def discover_fine_t2i_shards(cfg: FineT2IConfig) -> dict[str, list[FineTarShard]]:
    api = HfApi(token=resolve_hf_token(cfg.dataset.hf_token_env))
    out: dict[str, list[FineTarShard]] = {}
    for folder in cfg.dataset.folders:
        entries = api.list_repo_tree(cfg.dataset.repo_id, path_in_repo=folder, repo_type="dataset")
        shards: list[FineTarShard] = []
        for entry in entries:
            path = getattr(entry, "path", "")
            if not path.endswith(".tar"):
                continue
            shards.append(
                FineTarShard(
                    folder=folder,
                    repo_path=path,
                    url=f"https://huggingface.co/datasets/{cfg.dataset.repo_id}/resolve/main/{path}",
                    size_bytes=getattr(entry, "size", None),
                )
            )
        rng = random.Random(f"{cfg.sampling.rng_seed}:{folder}:shards")
        rng.shuffle(shards)
        out[folder] = shards
    return out
