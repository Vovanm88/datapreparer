from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import polars as pl

from .auth import hf_storage_options, resolve_hf_token
from .bad_pool import BadPoolManager
from .config import BuilderConfig
from .downloader import ImageDownloader, atomic_write, extension_from_url
from .metadata import get_image_bytes, get_image_url, is_candidate, stable_sample_id, strip_caption
from .progress import ProgressReporter
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
    progress = ProgressReporter(root, interval_seconds=cfg.download.progress_interval_seconds)

    if state.synthetic_count == 0 and not dry_run:
        for row in generate_synthetic_bad(root, noise_count=cfg.bad_pool.synthetic_noise_count, seed=cfg.bad_pool.rng_seed):
            writer.add(row)
            state.synthetic_count += 1
            state.bad_count += 1
            state.saved_image_bytes += int(row.get("file_size_bytes") or 0)
        state.output_part_index = writer.part_index
        store.save(state)
    progress.maybe_emit(state, force=True)

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

    try:
        await asyncio.to_thread(_process_shards_parallel, cfg, shards, state, store, writer, bad_pool, progress)
    finally:
        flushed = writer.flush()
        if flushed is not None:
            progress.counters.metadata_flushes += 1
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
    progress.maybe_emit(state, force=True)
    return summary


def _process_shards_parallel(
    cfg: BuilderConfig,
    shards: list[SourceShard],
    state,
    store: StateStore,
    writer: MetadataWriter,
    bad_pool: BadPoolManager,
    progress: ProgressReporter,
) -> None:
    start = state.current_shard_index
    workers = max(1, cfg.dataset.shard_workers)
    for batch_start in range(start, len(shards), workers):
        if state.saved_image_bytes >= cfg.output.target_image_bytes:
            break
        batch = list(enumerate(shards[batch_start : batch_start + workers], start=batch_start))
        events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, cfg.download.queue_size))
        stop_event = threading.Event()
        with ThreadPoolExecutor(max_workers=len(batch), thread_name_prefix="commoncatalog-shard") as executor:
            for shard_index, shard in batch:
                executor.submit(_shard_worker, cfg, shard, shard_index, store, events, stop_event)
            done = 0
            while done < len(batch):
                event = events.get()
                kind = event.get("type")
                if kind == "progress":
                    progress.counters.rows_scanned += int(event.get("rows_scanned", 0))
                    progress.counters.candidates += int(event.get("candidates", 0))
                    progress.counters.seen_skipped += int(event.get("seen_skipped", 0))
                    progress.set_shard_position(
                        shard=str(event["shard"]),
                        shard_index=int(event["shard_index"]),
                        row_offset=int(event["row_offset"]),
                    )
                    progress.in_flight = len(progress.active_shards)
                    progress.maybe_emit(state)
                elif kind == "result":
                    _commit_result(event, state, store, writer, bad_pool, progress)
                    state.output_part_index = writer.part_index
                    store.save(state)
                    progress.maybe_emit(state)
                    if state.saved_image_bytes >= cfg.output.target_image_bytes:
                        stop_event.set()
                elif kind == "failed":
                    progress.counters.download_failed += 1
                    progress.failure_reasons[str(event.get("failure_reason") or "failed")] += 1
                    progress.maybe_emit(state)
                elif kind == "shard_done":
                    done += 1
                    shard_index = int(event["shard_index"])
                    progress.finish_shard(shard_index)
                    if event.get("error"):
                        progress.counters.download_failed += 1
                        progress.failure_reasons[f"shard_error:{event['error']}"] += 1
                    progress.in_flight = len(progress.active_shards)
                    progress.maybe_emit(state, force=True)
        state.current_shard_index = batch_start + len(batch)
        state.current_row_offset = 0
        state.output_part_index = writer.part_index
        store.save(state)
        progress.maybe_emit(state, force=True)


def _shard_worker(
    cfg: BuilderConfig,
    shard: SourceShard,
    shard_index: int,
    store: StateStore,
    events: queue.Queue[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    try:
        token = resolve_hf_token(cfg.dataset.hf_token_env)
        frame = pl.read_parquet(shard.uri, storage_options=hf_storage_options(token))
        rows_scanned = 0
        candidates = 0
        seen_skipped = 0
        for row_index, row in enumerate(frame.iter_rows(named=True)):
            if stop_event.is_set():
                break
            rows_scanned += 1
            if not is_candidate(
                row,
                min_side=cfg.dataset.min_side,
                max_side=cfg.dataset.max_side,
                min_ratio=cfg.dataset.min_ratio,
                max_ratio=cfg.dataset.max_ratio,
            ):
                if rows_scanned % 256 == 0:
                    events.put(_progress_event(shard, shard_index, row_index + 1, rows_scanned, candidates, seen_skipped))
                    rows_scanned = candidates = seen_skipped = 0
                continue
            candidates += 1
            url = get_image_url(row)
            sample_id = stable_sample_id(row, url)
            if store.seen(sample_id):
                seen_skipped += 1
                continue
            result = _process_row_from_hf_bytes(cfg, row, url, sample_id, shard, row_index)
            if result.get("ok"):
                events.put({"type": "result", **result})
            else:
                events.put({"type": "failed", **result})
            if rows_scanned >= 256:
                events.put(_progress_event(shard, shard_index, row_index + 1, rows_scanned, candidates, seen_skipped))
                rows_scanned = candidates = seen_skipped = 0
        if rows_scanned or candidates or seen_skipped:
            events.put(_progress_event(shard, shard_index, row_index + 1 if "row_index" in locals() else 0, rows_scanned, candidates, seen_skipped))
        events.put({"type": "shard_done", "shard_index": shard_index, "shard": shard.repo_path})
    except Exception as exc:
        events.put({"type": "shard_done", "shard_index": shard_index, "shard": shard.repo_path, "error": exc.__class__.__name__})


def _progress_event(
    shard: SourceShard,
    shard_index: int,
    row_offset: int,
    rows_scanned: int,
    candidates: int,
    seen_skipped: int,
) -> dict[str, Any]:
    return {
        "type": "progress",
        "shard": shard.repo_path,
        "shard_index": shard_index,
        "row_offset": row_offset,
        "rows_scanned": rows_scanned,
        "candidates": candidates,
        "seen_skipped": seen_skipped,
    }


def _commit_result(
    event: dict[str, Any],
    state,
    store: StateStore,
    writer: MetadataWriter,
    bad_pool: BadPoolManager,
    progress: ProgressReporter,
) -> None:
    row = event["row"]
    sample_id = event["sample_id"]
    progress.counters.download_ok += 1
    progress.counters.hf_image_bytes_ok += 1
    if not bad_pool.accept(row, good_count=state.good_count, bad_count=state.bad_count):
        store.mark_seen(sample_id)
        progress.counters.bad_rejected += 1
        return
    flushed = writer.add(row)
    progress.counters.metadata_flushes += len([path for path in flushed if path is not None])
    store.mark_seen(sample_id)
    state.saved_image_bytes += int(row.get("file_size_bytes") or 0)
    if row.get("is_bad"):
        state.bad_count += 1
        progress.counters.bad_written += 1
    else:
        state.good_count += 1
        progress.counters.good_written += 1


async def _process_shard(
    cfg: BuilderConfig,
    shard: SourceShard,
    shard_index: int,
    state,
    store: StateStore,
    writer: MetadataWriter,
    downloader: ImageDownloader,
    bad_pool: BadPoolManager,
    progress: ProgressReporter,
) -> None:
    token = resolve_hf_token(cfg.dataset.hf_token_env)
    progress.set_position(shard=shard.repo_path, shard_index=shard_index, row_offset=state.current_row_offset, in_flight=0)
    progress.maybe_emit(state, force=True)
    frame = pl.read_parquet(shard.uri, storage_options=hf_storage_options(token))
    rows = frame.iter_rows(named=True)
    pending: set[asyncio.Task[dict[str, Any] | None]] = set()
    for row_index, row in enumerate(rows):
        progress.counters.rows_scanned += 1
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
        progress.counters.candidates += 1
        url = get_image_url(row)
        sample_id = stable_sample_id(row, url)
        if store.seen(sample_id):
            progress.counters.seen_skipped += 1
            continue
        pending.add(
            asyncio.create_task(_process_row(cfg, row, url, sample_id, shard, row_index, downloader))
        )
        progress.set_position(shard=shard.repo_path, shard_index=shard_index, row_offset=row_index + 1, in_flight=len(pending))
        progress.maybe_emit(state)
        if len(pending) >= cfg.download.concurrency:
            await _drain_one(pending, state, store, writer, bad_pool, progress)
            state.current_row_offset = row_index + 1
            state.output_part_index = writer.part_index
            store.save(state)
            progress.set_position(shard=shard.repo_path, shard_index=shard_index, row_offset=state.current_row_offset, in_flight=len(pending))
            progress.maybe_emit(state)
    while pending:
        await _drain_one(pending, state, store, writer, bad_pool, progress)
        state.output_part_index = writer.part_index
        store.save(state)
        progress.set_position(shard=shard.repo_path, shard_index=shard_index, row_offset=state.current_row_offset, in_flight=len(pending))
        progress.maybe_emit(state)


async def _drain_one(
    pending: set[asyncio.Task[dict[str, Any] | None]],
    state,
    store: StateStore,
    writer: MetadataWriter,
    bad_pool: BadPoolManager,
    progress: ProgressReporter,
) -> None:
    done, pending_rest = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    pending.clear()
    pending.update(pending_rest)
    for task in done:
        result = task.result()
        if result is None:
            progress.counters.download_failed += 1
            progress.failure_reasons["unknown"] += 1
            continue
        if not result.get("ok", True):
            progress.counters.download_failed += 1
            progress.failure_reasons[str(result.get("failure_reason") or "download_failed")] += 1
            continue
        row = result["row"]
        sample_id = result["sample_id"]
        progress.counters.download_ok += 1
        if result.get("image_source") == "hf_parquet_jpg_bytes":
            progress.counters.hf_image_bytes_ok += 1
        elif result.get("image_source") == "fallback_url":
            progress.counters.fallback_url_downloads += 1
        if not bad_pool.accept(row, good_count=state.good_count, bad_count=state.bad_count):
            store.mark_seen(sample_id)
            progress.counters.bad_rejected += 1
            continue
        flushed = writer.add(row)
        progress.counters.metadata_flushes += len([path for path in flushed if path is not None])
        store.mark_seen(sample_id)
        state.saved_image_bytes += int(row.get("file_size_bytes") or 0)
        if row.get("is_bad"):
            state.bad_count += 1
            progress.counters.bad_written += 1
        else:
            state.good_count += 1
            progress.counters.good_written += 1


def _process_row_from_hf_bytes(
    cfg: BuilderConfig,
    row: dict[str, Any],
    url: str | None,
    sample_id: str,
    shard: SourceShard,
    row_index: int,
) -> dict[str, Any]:
    image_bytes = get_image_bytes(row)
    if image_bytes is None:
        return {
            "ok": False,
            "sample_id": sample_id,
            "failure_reason": "missing_hf_image_bytes",
        }
    now = datetime.now(timezone.utc).isoformat()
    base = strip_caption(row)
    base["source_parquet_path"] = shard.repo_path
    base["source_row_index"] = row_index
    base["processed_at"] = now
    base["download_attempts"] = 0
    base["image_source"] = "hf_parquet_jpg_bytes"
    base["is_synthetic"] = False

    metrics, is_bad = score_image_bytes(
        image_bytes,
        cfg.quality,
        min_side=cfg.dataset.min_side,
        max_side=cfg.dataset.max_side,
        min_ratio=cfg.dataset.min_ratio,
        max_ratio=cfg.dataset.max_ratio,
    )
    digest = hashlib.sha256(image_bytes).hexdigest()
    ext = extension_from_row_or_url(row, url, None)
    label = "bad" if is_bad else "good"
    path = cfg.output.root / "images" / label / f"{digest[:2]}" / f"{digest}{ext}"
    atomic_write(path, image_bytes)

    out = {
        "rel_path": path.relative_to(cfg.output.root).as_posix(),
        "blip2_caption": row.get("blip2_caption"),
        "quality_label": label,
        "is_bad": is_bad,
        **base,
        **metrics.to_row(),
    }
    return {"ok": True, "sample_id": sample_id, "row": out, "image_source": "hf_parquet_jpg_bytes"}


async def _process_row(
    cfg: BuilderConfig,
    row: dict[str, Any],
    url: str,
    sample_id: str,
    shard: SourceShard,
    row_index: int,
    downloader: ImageDownloader,
) -> dict[str, Any] | None:
    image_bytes = get_image_bytes(row)
    content_type = None
    attempts = 0
    source = "hf_parquet_jpg_bytes"
    if image_bytes is None:
        if url is None:
            return {
                "ok": False,
                "sample_id": sample_id,
                "failure_reason": "missing_image_bytes_and_url",
                "download_attempts": 0,
            }
        result = await downloader.fetch(url)
        attempts = result.attempts
        content_type = result.content_type
        source = "fallback_url"
        if not result.ok:
            return {
                "ok": False,
                "sample_id": sample_id,
                "failure_reason": result.failure_reason or "download_failed",
                "download_attempts": result.attempts,
            }
        image_bytes = result.data
    now = datetime.now(timezone.utc).isoformat()
    base = strip_caption(row)
    base["source_parquet_path"] = shard.repo_path
    base["source_row_index"] = row_index
    base["processed_at"] = now
    base["download_attempts"] = attempts
    base["image_source"] = source
    base["is_synthetic"] = False

    metrics, is_bad = score_image_bytes(
        image_bytes,
        cfg.quality,
        min_side=cfg.dataset.min_side,
        max_side=cfg.dataset.max_side,
        min_ratio=cfg.dataset.min_ratio,
        max_ratio=cfg.dataset.max_ratio,
    )
    digest = hashlib.sha256(image_bytes).hexdigest()
    ext = extension_from_row_or_url(row, url, content_type)
    label = "bad" if is_bad else "good"
    path = cfg.output.root / "images" / label / f"{digest[:2]}" / f"{digest}{ext}"
    atomic_write(path, image_bytes)

    out = {
        "rel_path": path.relative_to(cfg.output.root).as_posix(),
        "blip2_caption": row.get("blip2_caption"),
        "quality_label": label,
        "is_bad": is_bad,
        **base,
        **metrics.to_row(),
    }
    return {"ok": True, "sample_id": sample_id, "row": out, "image_source": source}


def extension_from_row_or_url(row: dict[str, Any], url: str | None, content_type: str | None) -> str:
    ext = row.get("ext")
    if isinstance(ext, str):
        suffix = "." + ext.lower().lstrip(".")
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return ".jpg" if suffix == ".jpeg" else suffix
    jpg = row.get("jpg")
    if isinstance(jpg, dict):
        path = jpg.get("path")
        if isinstance(path, str):
            suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
            if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                return ".jpg" if suffix == ".jpeg" else suffix
    return extension_from_url(url or "", content_type)
