from __future__ import annotations

import asyncio
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .bad_pool import BadPoolManager
from .config import BuilderConfig
from .downloader import ImageDownloader, atomic_write, extension_from_url
from .metadata import get_image_url, is_candidate, stable_sample_id, strip_caption
from .quality import score_image_bytes
from .shards import SourceShard, discover_commoncatalog_shards
from .state import StateStore, remove_dangling_temps
from .synthetic import generate_synthetic_bad
from .writer import MetadataWriter, write_summary


async def run_builder(cfg: BuilderConfig, *, dry_run: bool = False, limit_shards: int | None = None) -> dict[str, Any]:
    root = cfg.output.root
    root.mkdir(parents=True, exist_ok=True)
    removed_temps = remove_dangling_temps(root)
    store = StateStore(root)
    state = store.load()
    state.rng_seed = cfg.bad_pool.rng_seed
    state.concurrency = cfg.download.concurrency
    writer = MetadataWriter(
        root,
        rows_per_part=cfg.output.metadata_rows_per_part,
        bytes_per_part=cfg.output.metadata_bytes_per_part,
        start_part=state.output_part_index,
    )
    bad_pool = BadPoolManager(root, hard_cap_ratio=cfg.bad_pool.hard_cap_ratio)

    if state.synthetic_count == 0 and not dry_run:
        for row in generate_synthetic_bad(root, noise_count=cfg.bad_pool.synthetic_noise_count, seed=cfg.bad_pool.rng_seed):
            writer.add(row)
            state.synthetic_count += 1
            state.bad_count += 1
            state.saved_image_bytes += int(row.get("file_size_bytes") or 0)
        state.output_part_index = writer.part_index
        store.save(state)

    shards = discover_commoncatalog_shards(cfg.dataset, limit=limit_shards)
    if dry_run:
        summary = {
            "dry_run": True,
            "candidate_shards": len(shards),
            "first_shards": [shard.repo_path for shard in shards[:20]],
            "concurrency": cfg.download.concurrency,
            "removed_temps": removed_temps,
        }
        write_summary(root, summary)
        return summary

    downloader = ImageDownloader(
        concurrency=cfg.download.concurrency,
        timeout_seconds=cfg.download.timeout_seconds,
        retries=cfg.download.retries,
        max_file_bytes=cfg.download.max_file_bytes,
        user_agent=cfg.download.user_agent,
    )
    try:
        for shard_index, shard in enumerate(shards[state.current_shard_index :], start=state.current_shard_index):
            if state.saved_image_bytes >= cfg.output.target_image_bytes:
                break
            await _process_shard(cfg, shard, shard_index, state, store, writer, downloader, bad_pool)
            state.current_shard_index = shard_index + 1
            state.current_row_offset = 0
            state.output_part_index = writer.part_index
            store.save(state)
    finally:
        await downloader.close()
        flushed = writer.flush()
        if flushed is not None:
            state.output_part_index = writer.part_index
            store.save(state)

    summary = {
        "dry_run": False,
        "saved_image_bytes": state.saved_image_bytes,
        "good_count": state.good_count,
        "bad_count": state.bad_count,
        "synthetic_count": state.synthetic_count,
        "output_part_index": state.output_part_index,
        "current_shard_index": state.current_shard_index,
    }
    write_summary(root, summary)
    return summary


async def _process_shard(
    cfg: BuilderConfig,
    shard: SourceShard,
    shard_index: int,
    state,
    store: StateStore,
    writer: MetadataWriter,
    downloader: ImageDownloader,
    bad_pool: BadPoolManager,
) -> None:
    frame = pl.read_parquet(shard.uri)
    rows = frame.iter_rows(named=True)
    pending: set[asyncio.Task[dict[str, Any] | None]] = set()
    for row_index, row in enumerate(rows):
        if shard_index == state.current_shard_index and row_index < state.current_row_offset:
            continue
        if state.saved_image_bytes >= cfg.output.target_image_bytes:
            break
        if not is_candidate(
            row,
            min_side=cfg.dataset.min_side,
            max_side=cfg.dataset.max_side,
            min_ratio=cfg.dataset.min_ratio,
            max_ratio=cfg.dataset.max_ratio,
        ):
            continue
        url = get_image_url(row)
        assert url is not None
        sample_id = stable_sample_id(row, url)
        if store.seen(sample_id):
            continue
        pending.add(
            asyncio.create_task(_process_row(cfg, row, url, sample_id, shard, row_index, downloader))
        )
        if len(pending) >= cfg.download.concurrency:
            await _drain_one(pending, state, store, writer, bad_pool)
            state.current_row_offset = row_index + 1
            state.output_part_index = writer.part_index
            store.save(state)
    while pending:
        await _drain_one(pending, state, store, writer, bad_pool)
        state.output_part_index = writer.part_index
        store.save(state)


async def _drain_one(
    pending: set[asyncio.Task[dict[str, Any] | None]],
    state,
    store: StateStore,
    writer: MetadataWriter,
    bad_pool: BadPoolManager,
) -> None:
    done, pending_rest = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    pending.clear()
    pending.update(pending_rest)
    for task in done:
        result = task.result()
        if result is None:
            continue
        row = result["row"]
        sample_id = result["sample_id"]
        if not bad_pool.accept(row, good_count=state.good_count, bad_count=state.bad_count):
            store.mark_seen(sample_id)
            continue
        writer.add(row)
        store.mark_seen(sample_id)
        state.saved_image_bytes += int(row.get("file_size_bytes") or 0)
        if row.get("is_bad"):
            state.bad_count += 1
        else:
            state.good_count += 1


async def _process_row(
    cfg: BuilderConfig,
    row: dict[str, Any],
    url: str,
    sample_id: str,
    shard: SourceShard,
    row_index: int,
    downloader: ImageDownloader,
) -> dict[str, Any] | None:
    result = await downloader.fetch(url)
    now = datetime.now(timezone.utc).isoformat()
    base = strip_caption(row)
    base["source_parquet_path"] = shard.repo_path
    base["source_row_index"] = row_index
    base["processed_at"] = now
    base["download_attempts"] = result.attempts
    base["is_synthetic"] = False

    if not result.ok:
        return None

    metrics, is_bad = score_image_bytes(
        result.data,
        cfg.quality,
        min_side=cfg.dataset.min_side,
        max_side=cfg.dataset.max_side,
        min_ratio=cfg.dataset.min_ratio,
        max_ratio=cfg.dataset.max_ratio,
    )
    digest = hashlib.sha256(result.data).hexdigest()
    ext = extension_from_url(url, result.content_type)
    label = "bad" if is_bad else "good"
    path = cfg.output.root / "images" / label / f"{digest[:2]}" / f"{digest}{ext}"
    atomic_write(path, result.data)

    out = {
        "rel_path": path.relative_to(cfg.output.root).as_posix(),
        "blip2_caption": row.get("blip2_caption"),
        "quality_label": label,
        "is_bad": is_bad,
        **base,
        **metrics.to_row(),
    }
    return {"sample_id": sample_id, "row": out}
