from __future__ import annotations

from pathlib import Path

from data.prepare_sft_dataset.bad_pool import BadPoolManager


def test_bad_pool_rejects_extra_bad_and_deletes_file(tmp_path: Path) -> None:
    image = tmp_path / "images" / "bad" / "x.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    row = {"is_bad": True, "is_synthetic": False, "rel_path": "images/bad/x.jpg"}
    pool = BadPoolManager(tmp_path, hard_cap_ratio=0.10)

    assert not pool.accept(row, good_count=9, bad_count=1)
    assert not image.exists()


def test_bad_pool_keeps_synthetic_bad(tmp_path: Path) -> None:
    row = {"is_bad": True, "is_synthetic": True, "rel_path": "images/bad/synthetic/x.png"}
    pool = BadPoolManager(tmp_path, hard_cap_ratio=0.10)

    assert pool.accept(row, good_count=9, bad_count=1)
