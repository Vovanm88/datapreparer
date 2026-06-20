from __future__ import annotations

import json

from data.prepare_sft_dataset.progress import ProgressReporter
from data.prepare_sft_dataset.state import RunState, StateStore


def test_progress_reporter_writes_json(tmp_path) -> None:
    reporter = ProgressReporter(tmp_path, interval_seconds=999)
    reporter.counters.download_ok = 3
    reporter.failure_reasons["http_404"] = 2
    reporter.set_position(shard="x.parquet", shard_index=2, row_offset=10, in_flight=4)
    state = RunState(saved_image_bytes=1024, good_count=1)

    reporter.maybe_emit(state, force=True)

    payload = json.loads((tmp_path / "reports" / "progress.json").read_text())
    assert payload["current_shard"] == "x.parquet"
    assert payload["in_flight_downloads"] == 4
    assert payload["counters"]["download_ok"] == 3
    assert payload["failure_reasons"]["http_404"] == 2


def test_state_load_supports_slots_dataclass(tmp_path) -> None:
    store = StateStore(tmp_path)
    store.save(RunState(saved_image_bytes=123, good_count=7))

    loaded = store.load()

    assert loaded.saved_image_bytes == 123
    assert loaded.good_count == 7
