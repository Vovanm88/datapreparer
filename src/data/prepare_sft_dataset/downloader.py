from __future__ import annotations

import asyncio
import email.utils
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


@dataclass(slots=True)
class DownloadResult:
    ok: bool
    data: bytes = b""
    attempts: int = 0
    failure_reason: str | None = None
    content_type: str | None = None


class ImageDownloader:
    def __init__(
        self,
        *,
        concurrency: int,
        timeout_seconds: float,
        retries: int,
        max_file_bytes: int,
        user_agent: str,
        rate_limit_base_sleep_seconds: float = 30.0,
        rate_limit_max_sleep_seconds: float = 300.0,
    ):
        self.concurrency = concurrency
        self.retries = retries
        self.max_file_bytes = max_file_bytes
        self.rate_limit_base_sleep_seconds = rate_limit_base_sleep_seconds
        self.rate_limit_max_sleep_seconds = rate_limit_max_sleep_seconds
        self._rate_limited_until = 0.0
        self._rate_limit_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch(self, url: str) -> DownloadResult:
        async with self._sem:
            attempts = 0
            last_reason = None
            for _ in range(self.retries):
                attempts += 1
                await self._wait_for_rate_limit()
                try:
                    head = await self._client.head(url)
                    if head.status_code == 429:
                        last_reason = "http_429"
                        await self._set_rate_limit(head.headers.get("Retry-After"), attempts)
                        continue
                    if head.status_code < 400:
                        length = head.headers.get("Content-Length")
                        if length and int(length) > self.max_file_bytes:
                            return DownloadResult(False, attempts=attempts, failure_reason="content_length_too_large")
                except Exception as exc:
                    last_reason = f"head_failed:{exc.__class__.__name__}"

                try:
                    chunks: list[bytes] = []
                    total = 0
                    async with self._client.stream("GET", url) as response:
                        if response.status_code == 429:
                            last_reason = "http_429"
                            await self._set_rate_limit(response.headers.get("Retry-After"), attempts)
                            continue
                        if response.status_code >= 400:
                            last_reason = f"http_{response.status_code}"
                            continue
                        content_type = response.headers.get("Content-Type")
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > self.max_file_bytes:
                                return DownloadResult(False, attempts=attempts, failure_reason="body_too_large")
                            chunks.append(chunk)
                    return DownloadResult(True, data=b"".join(chunks), attempts=attempts, content_type=content_type)
                except Exception as exc:
                    last_reason = f"get_failed:{exc.__class__.__name__}"
                    await asyncio.sleep(min(2.0, 0.2 * attempts))
            return DownloadResult(False, attempts=attempts, failure_reason=last_reason or "download_failed")

    async def _wait_for_rate_limit(self) -> None:
        delay = self._rate_limited_until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _set_rate_limit(self, retry_after: str | None, attempts: int) -> None:
        delay = self._parse_retry_after(retry_after)
        if delay is None:
            delay = min(
                self.rate_limit_max_sleep_seconds,
                self.rate_limit_base_sleep_seconds * max(1, attempts),
            )
        until = time.monotonic() + delay
        async with self._rate_limit_lock:
            self._rate_limited_until = max(self._rate_limited_until, until)

    def _parse_retry_after(self, retry_after: str | None) -> float | None:
        if not retry_after:
            return None
        try:
            return max(0.0, min(float(retry_after), self.rate_limit_max_sleep_seconds))
        except ValueError:
            pass
        try:
            dt = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, min((dt - datetime.now(timezone.utc)).total_seconds(), self.rate_limit_max_sleep_seconds))


def extension_from_url(url: str, content_type: str | None) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if content_type:
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
    return ".jpg"


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
