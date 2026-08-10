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


@dataclass(frozen=True)
class PreviewLease:
    snapshot: PreviewSnapshot
    lease_id: str


class AcquisitionPreviewStore:
    """One active, tokenized, attempt-owned preview with bounded readers."""

    def __init__(self, *, max_readers: int = 2):
        if max_readers < 1:
            raise ValueError("preview requires at least one reader")
        self._lock = asyncio.Lock()
        self._changed = asyncio.Condition(self._lock)
        self._snapshot: PreviewSnapshot | None = None
        self._max_readers = max_readers
        self._readers: set[str] = set()
        self._reader_tasks: dict[str, asyncio.Task] = {}
        self._stubborn_closed = False

    async def open(self, attempt_id: str, jpeg: bytes) -> str:
        if not jpeg:
            raise ValueError("preview requires JPEG bytes")
        token = secrets.token_urlsafe(32)
        async with self._changed:
            if self._stubborn_closed or self._readers:
                raise RuntimeError("acquisition_preview_closed_stubborn_readers")
            self._readers.clear()
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

    async def acquire(self, token: str) -> PreviewLease | None:
        """Atomically reserve one streaming reader, never a queued reader."""
        async with self._changed:
            current = self._snapshot
            if current is None or not secrets.compare_digest(current.token, token):
                return None
            if len(self._readers) >= self._max_readers:
                raise RuntimeError("acquisition_preview_reader_limit")
            lease_id = secrets.token_urlsafe(16)
            self._readers.add(lease_id)
            return PreviewLease(current, lease_id)

    async def release(self, lease_id: str) -> None:
        async with self._changed:
            self._readers.discard(lease_id)
            self._reader_tasks.pop(lease_id, None)
            if not self._readers:
                self._stubborn_closed = False
            self._changed.notify_all()

    async def register_task(self, lease_id: str, task: asyncio.Task | None = None) -> None:
        task = task or asyncio.current_task()
        if task is None:
            return
        async with self._changed:
            if lease_id in self._readers:
                self._reader_tasks[lease_id] = task

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

    async def close(self, attempt_id: str | None = None, *, timeout: float = 1.0) -> None:
        async with self._changed:
            if self._snapshot is not None and (
                attempt_id is None or self._snapshot.attempt_id == attempt_id
            ):
                self._snapshot = None
                self._changed.notify_all()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.001)
        natural_deadline = min(deadline, loop.time() + min(0.05, max(timeout, 0.001) / 2))
        while loop.time() < natural_deadline:
            async with self._changed:
                if not self._readers:
                    return
            await asyncio.sleep(0.002)
        async with self._changed:
            reader_tasks = [
                task for task in self._reader_tasks.values() if not task.done()
            ]
        for task in reader_tasks:
            task.cancel()
        while loop.time() < deadline:
            async with self._changed:
                if not self._readers:
                    return
            await asyncio.sleep(0.002)
        async with self._changed:
            if self._readers:
                self._stubborn_closed = True
                self._changed.notify_all()
                raise RuntimeError("acquisition_preview_stubborn_readers")

    async def reader_count(self) -> int:
        async with self._lock:
            return len(self._readers)
