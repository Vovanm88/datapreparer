from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    def __init__(self, *, concurrency: int, timeout_seconds: float, retries: int, max_file_bytes: int, user_agent: str):
        self.concurrency = concurrency
        self.retries = retries
        self.max_file_bytes = max_file_bytes
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
                try:
                    head = await self._client.head(url)
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
