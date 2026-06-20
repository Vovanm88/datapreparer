from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProgressCounters:
    rows_scanned: int = 0
    candidates: int = 0
    seen_skipped: int = 0
    download_ok: int = 0
    download_failed: int = 0
    good_written: int = 0
    bad_written: int = 0
    bad_rejected: int = 0
    metadata_flushes: int = 0


class ProgressReporter:
    def __init__(self, root: Path, *, interval_seconds: float):
        self.root = root
        self.reports_dir = root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.last_bytes = 0
        self.counters = ProgressCounters()
        self.current_shard: str | None = None
        self.current_shard_index: int | None = None
        self.current_row_offset: int | None = None
        self.in_flight = 0

    def set_position(self, *, shard: str, shard_index: int, row_offset: int, in_flight: int) -> None:
        self.current_shard = shard
        self.current_shard_index = shard_index
        self.current_row_offset = row_offset
        self.in_flight = in_flight

    def maybe_emit(self, state: Any, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_emit_at < self.interval_seconds:
            return
        elapsed = max(0.001, now - self.started_at)
        saved = int(getattr(state, "saved_image_bytes", 0))
        delta_bytes = saved - self.last_bytes
        mb_per_min_since_last = 0.0
        if self.last_emit_at:
            delta_seconds = max(0.001, now - self.last_emit_at)
            mb_per_min_since_last = (delta_bytes / 1024 / 1024) / (delta_seconds / 60)
        payload = {
            "updated_at_unix": time.time(),
            "elapsed_seconds": elapsed,
            "current_shard": self.current_shard,
            "current_shard_index": self.current_shard_index,
            "current_row_offset": self.current_row_offset,
            "in_flight_downloads": self.in_flight,
            "saved_image_bytes": saved,
            "saved_gib": saved / 1024**3,
            "mb_per_min_since_last": mb_per_min_since_last,
            "good_count": getattr(state, "good_count", 0),
            "bad_count": getattr(state, "bad_count", 0),
            "synthetic_count": getattr(state, "synthetic_count", 0),
            "counters": asdict(self.counters),
        }
        self._write(payload)
        self._print(payload)
        self.last_emit_at = now
        self.last_bytes = saved

    def _write(self, payload: dict[str, Any]) -> None:
        tmp = self.reports_dir / "progress.json.tmp"
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp.replace(self.reports_dir / "progress.json")

    def _print(self, payload: dict[str, Any]) -> None:
        counters = payload["counters"]
        print(
            "[commoncatalog-progress] "
            f"shard={payload['current_shard_index']} row={payload['current_row_offset']} "
            f"inflight={payload['in_flight_downloads']} saved={payload['saved_gib']:.2f}GiB "
            f"speed={payload['mb_per_min_since_last']:.1f}MB/min "
            f"ok={counters['download_ok']} fail={counters['download_failed']} "
            f"good={payload['good_count']} bad={payload['bad_count']} "
            f"bad_rejected={counters['bad_rejected']}",
            flush=True,
        )
