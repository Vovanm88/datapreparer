from __future__ import annotations

import asyncio

from data.prepare_sft_dataset.downloader import ImageDownloader


def test_parse_retry_after_seconds() -> None:
    downloader = ImageDownloader(
        concurrency=1,
        timeout_seconds=1,
        retries=1,
        max_file_bytes=1,
        user_agent="test",
        rate_limit_max_sleep_seconds=300,
    )
    try:
        assert downloader._parse_retry_after("120") == 120
        assert downloader._parse_retry_after("9999") == 300
    finally:
        asyncio.run(downloader.close())


def test_parse_retry_after_invalid() -> None:
    downloader = ImageDownloader(concurrency=1, timeout_seconds=1, retries=1, max_file_bytes=1, user_agent="test")
    try:
        assert downloader._parse_retry_after("nope") is None
    finally:
        asyncio.run(downloader.close())
