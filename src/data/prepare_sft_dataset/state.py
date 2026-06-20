from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class RunState:
    current_shard_index: int = 0
    current_row_offset: int = 0
    output_part_index: int = 0
    saved_image_bytes: int = 0
    good_count: int = 0
    bad_count: int = 0
    synthetic_count: int = 0
    rng_seed: int = 20260620
    concurrency: int = 64


class StateStore:
    def __init__(self, root: Path):
        self.root = root
        self.state_dir = root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "state.json"
        self.sqlite_path = self.state_dir / "seen_ids.sqlite"
        self._init_db()

    def load(self) -> RunState:
        if not self.path.exists():
            return RunState()
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return RunState(**{**RunState().__dict__, **raw})

    def save(self, state: RunState) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def seen(self, sample_id: str) -> bool:
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("select 1 from seen where sample_id = ?", (sample_id,)).fetchone()
        return row is not None

    def mark_seen(self, sample_id: str) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("insert or ignore into seen(sample_id) values (?)", (sample_id,))

    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("create table if not exists seen(sample_id text primary key)")


def remove_dangling_temps(root: Path) -> int:
    count = 0
    if not root.exists():
        return count
    for tmp in root.rglob("*.tmp"):
        tmp.unlink(missing_ok=True)
        count += 1
    return count
