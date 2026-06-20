from __future__ import annotations

import io
import json
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import BinaryIO, Iterable


IMAGE_EXTS = {"jpg", "jpeg", "png", "webp"}
PROMPT_KEYS = ("prompt", "enhanced_prompt", "original_prompt", "caption", "text")


@dataclass(slots=True)
class FineSample:
    key: str
    image_ext: str | None = None
    image_bytes: bytes | None = None
    txt_bytes: bytes | None = None
    json_bytes: bytes | None = None


def iter_webdataset_samples(fileobj: BinaryIO, *, shuffle_buffer: int, seed: str) -> Iterable[FineSample]:
    rng = random.Random(seed)
    grouped: dict[str, FineSample] = {}
    buffer: list[FineSample] = []
    with tarfile.open(fileobj=fileobj, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            key, ext = split_member_name(member.name)
            sample = grouped.setdefault(key, FineSample(key=key))
            ext_lower = ext.lower()
            if ext_lower in IMAGE_EXTS:
                sample.image_ext = "jpg" if ext_lower == "jpeg" else ext_lower
                sample.image_bytes = data
            elif ext_lower == "txt":
                sample.txt_bytes = data
            elif ext_lower == "json":
                sample.json_bytes = data
            if is_complete(sample):
                grouped.pop(key, None)
                buffer.append(sample)
                if len(buffer) >= shuffle_buffer:
                    idx = rng.randrange(len(buffer))
                    yield buffer.pop(idx)
    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)


def split_member_name(name: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    ext = path.suffix.lstrip(".")
    return path.with_suffix("").as_posix(), ext


def is_complete(sample: FineSample) -> bool:
    return sample.image_bytes is not None and sample.txt_bytes is not None and sample.json_bytes is not None


def choose_caption(json_bytes: bytes, txt_bytes: bytes) -> tuple[str, str, dict[str, object]]:
    txt = txt_bytes.decode("utf-8", errors="replace").strip()
    try:
        meta = json.loads(json_bytes.decode("utf-8", errors="replace"))
        if not isinstance(meta, dict):
            meta = {"_raw_json": meta}
    except json.JSONDecodeError:
        meta = {"_raw_json": json_bytes.decode("utf-8", errors="replace")}
    for key in PROMPT_KEYS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"json.{key}", meta
    return txt, "txt", meta


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))
