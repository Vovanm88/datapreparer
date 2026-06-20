from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class MetadataWriter:
    def __init__(self, root: Path, *, rows_per_part: int, bytes_per_part: int, start_part: int = 0):
        self.root = root
        self.metadata_dir = root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_part = rows_per_part
        self.bytes_per_part = bytes_per_part
        self.part_index = start_part
        self.buffer: list[dict[str, Any]] = []
        self.buffer_image_bytes = 0

    def add(self, row: dict[str, Any]) -> list[Path]:
        self.buffer.append(row)
        self.buffer_image_bytes += int(row.get("file_size_bytes") or 0)
        if len(self.buffer) >= self.rows_per_part or self.buffer_image_bytes >= self.bytes_per_part:
            return [self.flush()]
        return []

    def flush(self) -> Path | None:
        if not self.buffer:
            return None
        path = self.metadata_dir / f"part-{self.part_index:06d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        normalized = [_json_safe(row) for row in self.buffer]
        table = pa.Table.from_pylist(normalized)
        pq.write_table(table, tmp)
        tmp.replace(path)
        self.buffer.clear()
        self.buffer_image_bytes = 0
        self.part_index += 1
        return path


def write_summary(root: Path, summary: dict[str, Any]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tmp = reports / "summary.json.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(summary), fh, indent=2, sort_keys=True)
    tmp.replace(reports / "summary.json")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
