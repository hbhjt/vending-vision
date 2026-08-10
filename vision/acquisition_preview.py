"""Attempt-scoped display-only MJPEG preview capabilities.

The store owns no camera reads.  Acquisition writes JPEG display snapshots from
the Vision-owned source frame, while HTTP consumers can only read the current
attempt token.  Closing is immediate and tokens are never reused.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewSnapshot:
    attempt_id: str
    token: str
    jpeg: bytes


class AcquisitionPreviewStore:
    """One active, tokenized, attempt-owned preview with bounded readers."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._snapshot: PreviewSnapshot | None = None

    async def open(self, attempt_id: str, jpeg: bytes) -> str:
        if not jpeg:
            raise ValueError("preview requires JPEG bytes")
        token = secrets.token_urlsafe(32)
        async with self._changed:
            self._snapshot = PreviewSnapshot(attempt_id, token, bytes(jpeg))
            self._changed.notify_all()
        return token

    async def update(self, attempt_id: str, token: str, jpeg: bytes) -> bool:
        if not jpeg:
            return False
        async with self._changed:
            current = self._snapshot
            if current is None or current.attempt_id != attempt_id or current.token != token:
                return False
            self._snapshot = PreviewSnapshot(attempt_id, token, bytes(jpeg))
            self._changed.notify_all()
            return True

    async def get(self, token: str) -> PreviewSnapshot | None:
        async with self._lock:
            current = self._snapshot
            if current is None or not secrets.compare_digest(current.token, token):
                return None
            return current

    async def wait_for_change(self, token: str, previous: bytes) -> PreviewSnapshot | None:
        async with self._changed:
            await self._changed.wait_for(
                lambda: self._snapshot is None
                or self._snapshot.token != token
                or self._snapshot.jpeg != previous
            )
            current = self._snapshot
            if current is None or current.token != token:
                return None
            return current

    async def close(self, attempt_id: str | None = None) -> None:
        async with self._changed:
            if self._snapshot is not None and (
                attempt_id is None or self._snapshot.attempt_id == attempt_id
            ):
                self._snapshot = None
                self._changed.notify_all()
