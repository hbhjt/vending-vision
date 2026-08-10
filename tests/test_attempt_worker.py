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


def test_attempt_worker_deadline_terminates_non_cooperative_child_process():
    """A deadline returns promptly and leaves no attempt child behind."""
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(_run_worker(_blocked_native_boundary, (), timeout=0.05))

    assert time.monotonic() - started < 1.0
    assert multiprocessing.active_children() == []
