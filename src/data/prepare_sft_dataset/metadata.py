from __future__ import annotations

import math
import hashlib
from typing import Any


DROP_SOURCE_COLUMNS = {"caption"}


def get_image_bytes(row: dict[str, Any]) -> bytes | None:
    jpg = row.get("jpg")
    if isinstance(jpg, dict):
        value = jpg.get("bytes")
        if isinstance(value, bytes | bytearray | memoryview):
            return bytes(value)
    return None


def get_image_url(row: dict[str, Any]) -> str | None:
    jpg = row.get("jpg")
    if isinstance(jpg, dict):
        for key in ("path", "url", "src"):
            value = jpg.get(key)
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                return value
    for key in ("url", "downloadurl"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def stable_sample_id(row: dict[str, Any], url: str | None = None) -> str:
    for key in ("sha256", "photoid", "key"):
        value = row.get(key)
        if value is not None and str(value):
            return f"{key}:{value}"
    image_bytes = get_image_bytes(row)
    if image_bytes is not None:
        return f"jpg_bytes:{hashlib.sha256(image_bytes).hexdigest()}"
    if url is None:
        return "unknown"
    return f"url:{url}"


def is_candidate(row: dict[str, Any], *, min_side: int, max_side: int, min_ratio: float, max_ratio: float) -> bool:
    caption = row.get("blip2_caption")
    if not isinstance(caption, str) or not caption.strip():
        return False
    width = _as_int(row.get("width"))
    height = _as_int(row.get("height"))
    if width is None or height is None:
        return False
    if width < min_side or height < min_side:
        return False
    if max(width, height) > max_side:
        return False
    ratio = width / height
    if not (min_ratio <= ratio <= max_ratio):
        return False
    return get_image_bytes(row) is not None or get_image_url(row) is not None


def strip_caption(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in row.items() if key not in DROP_SOURCE_COLUMNS}


def _as_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return None
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value
