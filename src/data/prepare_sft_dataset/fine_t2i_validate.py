from __future__ import annotations

import tarfile
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_MANIFEST_COLUMNS = {
    "tar_rel_path",
    "sample_key",
    "image_ext",
    "caption",
    "caption_source",
    "source_folder",
    "source_tar",
    "source_key",
    "image_bytes",
    "json_bytes",
    "txt_bytes",
    "written_at",
}


def validate_fine_t2i(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_parts = sorted((root / "metadata").glob("manifest-*.parquet"))
    if not manifest_parts:
        return ["no manifest parquet parts found"]
    expected: dict[str, set[str]] = {}
    for part in manifest_parts:
        table = pq.read_table(part)
        missing = REQUIRED_MANIFEST_COLUMNS - set(table.column_names)
        if missing:
            errors.append(f"{part.name}: missing columns {sorted(missing)}")
            continue
        for row in table.select(["tar_rel_path", "sample_key", "image_ext"]).to_pylist():
            expected.setdefault(row["tar_rel_path"], set()).update(
                {f"{row['sample_key']}.{row['image_ext']}", f"{row['sample_key']}.txt", f"{row['sample_key']}.json"}
            )
    for rel_tar, names in expected.items():
        tar_path = root / rel_tar
        if not tar_path.exists():
            errors.append(f"missing tar {rel_tar}")
            continue
        with tarfile.open(tar_path, mode="r") as tar:
            actual = set(tar.getnames())
        missing_names = names - actual
        if missing_names:
            errors.append(f"{rel_tar}: missing {len(missing_names)} sample members")
    return errors
