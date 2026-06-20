from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {"rel_path", "blip2_caption", "quality_label", "is_bad", "is_synthetic"}


def validate_output(root: Path) -> list[str]:
    errors: list[str] = []
    metadata_dir = root / "metadata"
    if not metadata_dir.exists():
        return [f"missing metadata directory: {metadata_dir}"]
    parts = sorted(metadata_dir.glob("part-*.parquet"))
    if not parts:
        errors.append("no metadata parquet parts found")
        return errors
    for part in parts:
        table = pq.read_table(part)
        columns = set(table.column_names)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            errors.append(f"{part.name}: missing columns {sorted(missing)}")
            continue
        if "caption" in columns:
            errors.append(f"{part.name}: forbidden source column caption is present")
        rel_paths = table.column("rel_path").to_pylist()
        for rel_path in rel_paths:
            if not rel_path:
                errors.append(f"{part.name}: empty rel_path")
                continue
            if Path(rel_path).is_absolute():
                errors.append(f"{part.name}: absolute rel_path {rel_path}")
                continue
            if not (root / rel_path).exists():
                errors.append(f"{part.name}: missing image file {rel_path}")
    return errors
