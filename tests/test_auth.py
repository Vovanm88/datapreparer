from __future__ import annotations

from data.prepare_sft_dataset.auth import hf_storage_options, resolve_hf_token


def test_resolve_hf_token_prefers_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert resolve_hf_token() == "hf_test"


def test_hf_storage_options_only_when_token() -> None:
    assert hf_storage_options("hf_test") == {"token": "hf_test"}
    assert hf_storage_options(None) == {}
