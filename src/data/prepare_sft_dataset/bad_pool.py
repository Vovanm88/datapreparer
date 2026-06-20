from __future__ import annotations

from pathlib import Path


class BadPoolManager:
    """Keeps the written bad sample ratio under a hard cap.

    Synthetic bad rows are added before normal processing and are treated as
    pinned. For streamed rows we avoid writing extra non-synthetic bad examples
    once the hard cap would be exceeded; the downloaded file is removed before
    metadata is committed.
    """

    def __init__(self, root: Path, *, hard_cap_ratio: float):
        self.root = root
        self.hard_cap_ratio = hard_cap_ratio

    def accept(self, row: dict[str, object], *, good_count: int, bad_count: int) -> bool:
        if not row.get("is_bad") or row.get("is_synthetic"):
            return True
        next_bad = bad_count + 1
        next_total = good_count + next_bad
        if next_total == 0:
            return True
        if (next_bad / next_total) <= self.hard_cap_ratio:
            return True
        rel_path = row.get("rel_path")
        if isinstance(rel_path, str):
            (self.root / rel_path).unlink(missing_ok=True)
        return False
