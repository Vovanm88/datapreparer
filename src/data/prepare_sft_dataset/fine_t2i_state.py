from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class FineState:
    folder_positions: dict[str, int] = field(default_factory=dict)
    accepted_counts: dict[str, int] = field(default_factory=dict)
    output_part_index: int = 0
    manifest_part_index: int = 0
    written_bytes: int = 0


class FineStateStore:
    def __init__(self, root: Path):
        self.root = root
        self.state_dir = root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "state.json"
        self.sqlite_path = self.state_dir / "seen.sqlite"
        self._init_db()

    def load(self) -> FineState:
        if not self.path.exists():
            return FineState()
        with self.path.open("r", encoding="utf-8") as fh:
            return FineState(**json.load(fh))

    def save(self, state: FineState) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(state), fh, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def seen(self, source_id: str) -> bool:
        with sqlite3.connect(self.sqlite_path) as conn:
            return conn.execute("select 1 from seen where source_id = ?", (source_id,)).fetchone() is not None

    def mark_many_seen(self, source_ids: list[str]) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany("insert or ignore into seen(source_id) values (?)", [(x,) for x in source_ids])

    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("create table if not exists seen(source_id text primary key)")
