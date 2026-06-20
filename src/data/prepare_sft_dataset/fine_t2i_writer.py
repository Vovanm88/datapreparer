from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .fine_t2i_tar import add_bytes_to_tar


class FineWebDatasetWriter:
    def __init__(self, root: Path, *, shard_bytes: int, start_part: int, start_manifest_part: int):
        self.root = root
        self.webdataset_dir = root / "webdataset"
        self.metadata_dir = root / "metadata"
        self.webdataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.part_index = start_part
        self.manifest_part_index = start_manifest_part
        self.current_path: Path | None = None
        self.current_tmp: Path | None = None
        self.tar: tarfile.TarFile | None = None
        self.current_bytes = 0
        self.manifest_rows: list[dict[str, Any]] = []
        self.pending_source_ids: list[str] = []

    def add_sample(self, sample_key: str, image_ext: str, image_bytes: bytes, txt_bytes: bytes, json_bytes: bytes, manifest: dict[str, Any], source_id: str) -> bool:
        self._ensure_open()
        assert self.tar is not None and self.current_path is not None
        add_bytes_to_tar(self.tar, f"{sample_key}.{image_ext}", image_bytes)
        add_bytes_to_tar(self.tar, f"{sample_key}.txt", txt_bytes)
        add_bytes_to_tar(self.tar, f"{sample_key}.json", json_bytes)
        row = {**manifest, "tar_rel_path": self.current_path.relative_to(self.root).as_posix(), "sample_key": sample_key}
        self.manifest_rows.append(row)
        self.pending_source_ids.append(source_id)
        self.current_bytes += len(image_bytes) + len(txt_bytes) + len(json_bytes)
        return self.current_bytes >= self.shard_bytes

    def finalize_current(self) -> tuple[Path | None, list[str]]:
        if self.tar is None:
            return None, []
        assert self.current_tmp is not None and self.current_path is not None
        self.tar.close()
        self.current_tmp.replace(self.current_path)
        manifest_path = self._flush_manifest()
        source_ids = self.pending_source_ids
        self.pending_source_ids = []
        self.tar = None
        self.current_tmp = None
        self.current_path = None
        self.current_bytes = 0
        self.part_index += 1
        return manifest_path, source_ids

    def _ensure_open(self) -> None:
        if self.tar is not None:
            return
        self.current_path = self.webdataset_dir / f"train-{self.part_index:06d}.tar"
        self.current_tmp = self.current_path.with_suffix(".tar.tmp")
        self.tar = tarfile.open(self.current_tmp, mode="w")

    def _flush_manifest(self) -> Path | None:
        if not self.manifest_rows:
            return None
        path = self.metadata_dir / f"manifest-{self.manifest_part_index:06d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self.manifest_rows)
        pq.write_table(table, tmp)
        tmp.replace(path)
        self.manifest_rows.clear()
        self.manifest_part_index += 1
        return path


def write_fine_summary(root: Path, summary: dict[str, Any]) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tmp = reports / "summary.json.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    tmp.replace(reports / "summary.json")
