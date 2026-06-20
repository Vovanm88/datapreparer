from __future__ import annotations

from .fine_t2i_config import FineT2IConfig


def estimate_folder_quotas(cfg: FineT2IConfig) -> dict[str, int]:
    weighted_avg = 0.0
    for folder in cfg.dataset.folders.values():
        weighted_avg += folder.ratio * (folder.published_bytes / folder.published_samples)
    total_samples = max(1, round(cfg.output.target_bytes / weighted_avg))
    quotas: dict[str, int] = {}
    remaining = total_samples
    items = list(cfg.dataset.folders.items())
    for name, folder in items[:-1]:
        count = round(total_samples * folder.ratio)
        quotas[name] = count
        remaining -= count
    quotas[items[-1][0]] = max(0, remaining)
    return quotas
