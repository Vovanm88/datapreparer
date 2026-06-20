from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx

from .auth import resolve_hf_token
from .fine_t2i_config import FineT2IConfig
from .fine_t2i_discovery import FineTarShard, discover_fine_t2i_shards
from .fine_t2i_quota import estimate_folder_quotas
from .fine_t2i_state import FineStateStore
from .fine_t2i_tar import FineSample, choose_caption, iter_webdataset_samples
from .fine_t2i_writer import FineWebDatasetWriter, write_fine_summary
from .state import remove_dangling_temps


class _IteratorReader(io.RawIOBase):
    def __init__(self, chunks: Iterable[bytes]):
        self._chunks = iter(chunks)
        self._buffer = bytearray()
        self._closed = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._closed:
            return 0
        while not self._buffer:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                self._closed = True
                return 0
        n = min(len(b), len(self._buffer))
        b[:n] = self._buffer[:n]
        del self._buffer[:n]
        return n


def inspect_fine_t2i(cfg: FineT2IConfig) -> dict[str, Any]:
    shards = discover_fine_t2i_shards(cfg)
    quotas = estimate_folder_quotas(cfg)
    return {
        "repo_id": cfg.dataset.repo_id,
        "target_bytes": cfg.output.target_bytes,
        "quotas": quotas,
        "folders": {
            folder: {
                "ratio": cfg.dataset.folders[folder].ratio,
                "shards": len(items),
                "known_tar_bytes": sum(item.size_bytes or 0 for item in items),
                "first_shards": [item.repo_path for item in items[:5]],
            }
            for folder, items in shards.items()
        },
    }


def sample_fine_t2i(cfg: FineT2IConfig, *, dry_run: bool = False) -> dict[str, Any]:
    root = cfg.output.root
    root.mkdir(parents=True, exist_ok=True)
    removed_temps = remove_dangling_temps(root)
    shards = discover_fine_t2i_shards(cfg)
    quotas = estimate_folder_quotas(cfg)
    if dry_run:
        summary = {**inspect_fine_t2i(cfg), "dry_run": True, "removed_temps": removed_temps}
        write_fine_summary(root, summary)
        return summary

    store = FineStateStore(root)
    state = store.load()
    writer = FineWebDatasetWriter(
        root,
        shard_bytes=cfg.output.tar_shard_bytes,
        start_part=state.output_part_index,
        start_manifest_part=state.manifest_part_index,
    )
    headers = {"User-Agent": cfg.download.user_agent}
    token = resolve_hf_token(cfg.dataset.hf_token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=httpx.Timeout(cfg.download.timeout_seconds), follow_redirects=True, headers=headers) as client:
        for folder in cfg.dataset.folders:
            accepted = state.accepted_counts.get(folder, 0)
            quota = quotas[folder]
            if accepted >= quota:
                continue
            folder_shards = shards.get(folder, [])
            start = state.folder_positions.get(folder, 0)
            for shard_index, shard in enumerate(folder_shards[start:], start=start):
                accepted = state.accepted_counts.get(folder, 0)
                if accepted >= quota:
                    break
                _process_source_tar(cfg, client, shard, quota, state, store, writer)
                state.folder_positions[folder] = shard_index + 1
                state.output_part_index = writer.part_index
                state.manifest_part_index = writer.manifest_part_index
                store.save(state)
    _, source_ids = writer.finalize_current()
    if source_ids:
        store.mark_many_seen(source_ids)
        state.output_part_index = writer.part_index
        state.manifest_part_index = writer.manifest_part_index
        store.save(state)

    summary = {
        "dry_run": False,
        "target_bytes": cfg.output.target_bytes,
        "written_bytes": state.written_bytes,
        "quotas": quotas,
        "accepted_counts": state.accepted_counts,
        "output_part_index": state.output_part_index,
        "manifest_part_index": state.manifest_part_index,
    }
    write_fine_summary(root, summary)
    return summary


def _process_source_tar(
    cfg: FineT2IConfig,
    client: httpx.Client,
    shard: FineTarShard,
    quota: int,
    state,
    store: FineStateStore,
    writer: FineWebDatasetWriter,
) -> None:
    folder = shard.folder
    with client.stream("GET", shard.url) as response:
        response.raise_for_status()
        reader = _IteratorReader(response.iter_bytes())
        samples = iter_webdataset_samples(
            reader,
            shuffle_buffer=cfg.sampling.shuffle_buffer,
            seed=f"{cfg.sampling.rng_seed}:{shard.repo_path}:samples",
        )
        for sample in samples:
            if state.accepted_counts.get(folder, 0) >= quota:
                break
            source_id = f"{folder}/{shard.repo_path}/{sample.key}"
            if store.seen(source_id):
                continue
            accepted = _write_sample(cfg, shard, sample, source_id, writer)
            if not accepted:
                continue
            state.accepted_counts[folder] = state.accepted_counts.get(folder, 0) + 1
            state.written_bytes += accepted
            if writer.current_bytes >= cfg.output.tar_shard_bytes:
                _, source_ids = writer.finalize_current()
                store.mark_many_seen(source_ids)
                state.output_part_index = writer.part_index
                state.manifest_part_index = writer.manifest_part_index
                store.save(state)


def _write_sample(cfg: FineT2IConfig, shard: FineTarShard, sample: FineSample, source_id: str, writer: FineWebDatasetWriter) -> int | None:
    if sample.image_bytes is None or sample.txt_bytes is None or sample.json_bytes is None or sample.image_ext is None:
        return None
    caption, caption_source, meta = choose_caption(sample.json_bytes, sample.txt_bytes)
    source_txt = sample.txt_bytes.decode("utf-8", errors="replace")
    out_key = _output_key(shard.folder, source_id)
    augmented = {
        **meta,
        "source_folder": shard.folder,
        "source_tar": shard.repo_path,
        "source_key": sample.key,
        "source_txt": source_txt,
        "caption_source": caption_source,
        "sample_ratio_group": shard.folder,
    }
    out_txt = caption.encode("utf-8")
    out_json = json.dumps(augmented, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest = {
        "image_ext": sample.image_ext,
        "caption": caption,
        "caption_source": caption_source,
        "source_folder": shard.folder,
        "source_tar": shard.repo_path,
        "source_key": sample.key,
        "image_bytes": len(sample.image_bytes),
        "json_bytes": len(out_json),
        "txt_bytes": len(out_txt),
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    writer.add_sample(out_key, sample.image_ext, sample.image_bytes, out_txt, out_json, manifest, source_id)
    return len(sample.image_bytes) + len(out_txt) + len(out_json)


def _output_key(folder: str, source_id: str) -> str:
    safe = source_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe
