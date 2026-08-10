import asyncio
import multiprocessing
import time
from multiprocessing.connection import Connection

import pytest

from vision.attempt_worker import _run_worker


def _blocked_native_boundary(connection: Connection) -> None:
    """A stand-in for a native call which never observes Python cancellation."""
    try:
        time.sleep(5)
        connection.send(("ok", "late"))
    finally:
        connection.close()


def _echo_payload_size(connection: Connection, payload: bytes) -> None:
    try:
        connection.send(("ok", len(payload)))
    finally:
        connection.close()


def test_attempt_worker_deadline_terminates_non_cooperative_child_process():
    """A deadline returns promptly and leaves no attempt child behind."""
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(_run_worker(_blocked_native_boundary, (), timeout=0.05))

    assert time.monotonic() - started < 1.0
    assert multiprocessing.active_children() == []


def test_attempt_worker_large_payload_deadline_does_not_block_event_loop_or_leave_child():
    """Spawn start/IPC for max legal payload remains inside the absolute deadline."""
    async def scenario():
        payload = b"x" * (4096 * 4096 * 4)
        gaps: list[float] = []
        stop = asyncio.Event()

        async def ticker():
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(0.001)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        ticker_task = asyncio.create_task(ticker())
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await _run_worker(_echo_payload_size, (payload,), timeout=0.01)
        elapsed = time.monotonic() - started
        stop.set()
        await ticker_task

        assert elapsed < 1.0
        assert max(gaps, default=0) < 0.1
        assert multiprocessing.active_children() == []

    asyncio.run(scenario())
