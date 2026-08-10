import asyncio
import multiprocessing
import os
import threading
import time

import numpy as np
import pytest

from vision.directshow_broker import (
    DirectShowCameraBroker,
    directshow_broker_entry,
)


def _blocking_broker_target(connection, _config):
    try:
        command, _payload = connection.recv()
        if command == "read":
            while True:
                time.sleep(1)
    finally:
        connection.close()


def _happy_broker_target(connection, _config):
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command == "read":
                connection.send(
                    (
                        "ok",
                        {
                            "pid": os.getpid(),
                            "image": np.zeros((12, 16, 3), dtype=np.uint8),
                        },
                    )
                )
    finally:
        connection.close()


def _block_first_broker_target(connection, config):
    counter = config["requestCounter"]
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command == "read":
                with counter.get_lock():
                    counter.value += 1
                    request_number = counter.value
                if request_number == 1:
                    while True:
                        time.sleep(1)
                connection.send(
                    (
                        "ok",
                        {
                            "pid": os.getpid(),
                            "image": np.zeros((12, 16, 3), dtype=np.uint8),
                        },
                    )
                )
    finally:
        connection.close()


def _broker_config():
    return {
        "role": "profile_fast_try_on",
        "index": 0,
        "backend": "dshow",
        "stableId": "test-front",
        "keep_open": True,
    }


def test_directshow_broker_target_is_spawn_importable_without_app_boundary():
    assert directshow_broker_entry.__module__ == "vision.directshow_broker"


def test_directshow_broker_deadline_kills_blocked_child_and_next_request_restarts():
    context = multiprocessing.get_context("spawn")
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        target=_blocking_broker_target,
    )

    with pytest.raises(TimeoutError):
        broker.read(warmup_frames=1, timeout=0.05)
    assert broker.assert_dead()

    restarted = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        target=_happy_broker_target,
    )
    try:
        image = restarted.read(warmup_frames=1, timeout=1.0)
        assert image.shape == (12, 16, 3)
        assert restarted.pid is not None
    finally:
        restarted.release()
    assert restarted.assert_dead()


def test_async_blocked_request_keeps_loop_responsive_and_cancel_joins_before_restart():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = DirectShowCameraBroker(
        "front",
        {**_broker_config(), "requestCounter": counter},
        context=context,
        target=_block_first_broker_target,
    )

    async def scenario():
        read_task = asyncio.create_task(broker.read_async(warmup_frames=1, timeout=15.0))
        deadline = asyncio.get_running_loop().time() + 1.0
        while counter.value < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert broker.pid is not None
        assert counter.value == 1
        ticks = 0
        tick_deadline = asyncio.get_running_loop().time() + 0.05
        while asyncio.get_running_loop().time() < tick_deadline:
            ticks += 1
            await asyncio.sleep(0.002)

        read_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(read_task, timeout=1.0)

        assert ticks >= 10
        assert broker.assert_dead()
        assert broker.active_request_count == 0
        image = await broker.read_async(warmup_frames=1, timeout=1.0)
        assert image.shape == (12, 16, 3)

    try:
        asyncio.run(scenario())
    finally:
        broker.release()
    assert broker.assert_dead()


def test_stubborn_process_is_retained_and_broker_fails_closed_without_restart():
    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            return False

        def close(self):
            return None

    class StubbornProcess:
        pid = 9191
        exitcode = None

        def start(self):
            return None

        def is_alive(self):
            return True

        def kill(self):
            raise OSError("kill denied")

        def terminate(self):
            raise PermissionError("terminate denied")

        def join(self, timeout=None):
            return None

    class Context:
        def __init__(self):
            self.processes = []

        def Pipe(self, duplex=True):
            return Connection(), Connection()

        def Process(self, **_kwargs):
            process = StubbornProcess()
            self.processes.append(process)
            return process

    context = Context()
    broker = DirectShowCameraBroker("front", _broker_config(), context=context)
    broker._start_locked()
    stubborn = context.processes[0]

    assert broker.release() is False
    assert broker.assert_dead() is False
    assert broker._process is stubborn
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.read(warmup_frames=1, timeout=0.01)
    assert context.processes == [stubborn]


def test_live_process_kill_oserror_is_reported_and_retained_fail_closed():
    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            return False

        def close(self):
            return None

    class KillDeniedProcess:
        pid = 9292
        exitcode = None

        def start(self):
            return None

        def is_alive(self):
            return True

        def kill(self):
            raise OSError("kill denied")

        def terminate(self):
            raise PermissionError("terminate denied")

        def join(self, timeout=None):
            raise OSError("join denied")

    class Context:
        def __init__(self):
            self.processes = []

        def Pipe(self, duplex=True):
            return Connection(), Connection()

        def Process(self, **_kwargs):
            process = KillDeniedProcess()
            self.processes.append(process)
            return process

    context = Context()
    broker = DirectShowCameraBroker("front", _broker_config(), context=context)
    broker._start_locked()
    denied = context.processes[0]

    assert broker.release() is False
    assert broker.assert_dead() is False
    assert broker._process is denied
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.read(warmup_frames=1, timeout=0.01)
    assert context.processes == [denied]


def test_abort_async_control_error_returns_failure_by_deadline_without_thread_leak():
    class LiveProcess:
        pid = 9393
        exitcode = None

        def is_alive(self):
            return True

        def kill(self):
            return None

        def terminate(self):
            return None

        def join(self, timeout=None):
            return None

    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.05
    )
    live = LiveProcess()
    broker._process = live

    async def scenario():
        result = await asyncio.wait_for(
            broker.abort_async(reason="replacement"), timeout=0.1
        )
        await asyncio.sleep(0)
        return result

    assert asyncio.run(scenario()) is False
    assert broker.assert_dead() is False
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.read(warmup_frames=1, timeout=0.01)
    assert not any(
        thread.name == "directshow-front-abort" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_abort_async_does_not_call_stubborn_blocking_join_before_dead():
    class LiveProcess:
        pid = 9494
        exitcode = None

        def __init__(self):
            self.join_calls = []

        def is_alive(self):
            return True

        def kill(self):
            return None

        def terminate(self):
            return None

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            if timeout and timeout > 0:
                while True:
                    time.sleep(1)

    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.05
    )
    live = LiveProcess()
    broker._process = live

    async def scenario():
        started = time.monotonic()
        result = await asyncio.wait_for(
            broker.abort_async(reason="replacement"), timeout=0.2
        )
        return result, time.monotonic() - started

    result, elapsed = asyncio.run(scenario())

    assert result is False
    assert elapsed < 3.5
    assert live.join_calls == []
    assert broker.assert_dead() is False
    assert not any(
        thread.name == "directshow-front-abort" and thread.is_alive()
        for thread in threading.enumerate()
    )
