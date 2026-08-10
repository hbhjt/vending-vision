import asyncio
import multiprocessing
import os
import threading
import time

import numpy as np
import pytest

import vision.directshow_broker as directshow_broker
from vision.directshow_broker import (
    DirectShowCameraUnavailable,
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


def test_async_read_timeout_kills_real_blocked_child_and_keeps_loop_responsive():
    context = multiprocessing.get_context("spawn")
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        target=_blocking_broker_target,
        stop_timeout_seconds=2.0,
    )

    async def scenario():
        started = time.monotonic()
        read_task = asyncio.create_task(
            broker.read_async(warmup_frames=1, timeout=0.05)
        )
        ticks = 0
        while not read_task.done():
            ticks += 1
            await asyncio.sleep(0.002)
        with pytest.raises(TimeoutError, match="read deadline exceeded"):
            await read_task
        return time.monotonic() - started, ticks

    try:
        elapsed, ticks = asyncio.run(scenario())
    finally:
        broker.release()

    assert elapsed < 2.5
    assert ticks >= 5
    assert broker.assert_dead()
    assert broker.active_request_count == 0


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


@pytest.mark.parametrize("liveness_error", [PermissionError, OSError, ValueError])
def test_unknown_process_liveness_is_retained_and_broker_fails_closed(
    liveness_error,
):
    class Connection:
        def close(self):
            return None

    class UnknownProcess:
        pid = 9211
        exitcode = None

        def __init__(self):
            self.kill_attempted = False

        def start(self):
            return None

        def is_alive(self):
            raise liveness_error("liveness unavailable")

        def kill(self):
            self.kill_attempted = True

        def terminate(self):
            self.kill_attempted = True

    class Context:
        def __init__(self):
            self.processes = []

        def Pipe(self, duplex=True):
            return Connection(), Connection()

        def Process(self, **_kwargs):
            process = UnknownProcess()
            self.processes.append(process)
            return process

    context = Context()
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        stop_timeout_seconds=0.01,
    )
    broker._start_locked()
    unknown = context.processes[0]

    assert broker.release() is False
    assert unknown.kill_attempted
    assert broker.assert_dead() is False
    assert broker._process is unknown
    with pytest.raises(RuntimeError, match="unavailable"):
        broker.read(warmup_frames=1, timeout=0.01)
    assert context.processes == [unknown]


def test_async_read_timeout_reports_fatal_unavailable_when_physical_stop_fails():
    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            return False

        def close(self):
            return None

    class StubbornProcess:
        pid = 9242
        exitcode = None
        kill_attempted = True

        def start(self):
            return None

        def is_alive(self):
            return True

        def kill(self):
            return None

        def terminate(self):
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
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        stop_timeout_seconds=0.05,
    )

    async def scenario():
        with pytest.raises(
            DirectShowCameraUnavailable,
            match="directshow broker is unavailable: request_abort_failed",
        ):
            await asyncio.wait_for(
                broker.read_async(warmup_frames=1, timeout=0.01),
                timeout=0.2,
            )
        assert broker.active_request_count == 0

    asyncio.run(scenario())

    stubborn = context.processes[0]
    assert broker._process is stubborn
    assert broker.assert_dead() is False
    with pytest.raises(DirectShowCameraUnavailable, match="unavailable"):
        broker.read(warmup_frames=1, timeout=0.01)
    assert context.processes == [stubborn]


def test_async_read_timeout_waits_for_delayed_physical_death_and_thread_exit():
    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            return False

        def close(self):
            return None

    class DelayedDeathProcess:
        pid = 9252
        exitcode = None
        kill_attempted = True

        def __init__(self):
            self.dead_at = None

        def start(self):
            return None

        def is_alive(self):
            return self.dead_at is None or time.monotonic() < self.dead_at

        def kill(self):
            self.dead_at = time.monotonic() + 0.3

        def terminate(self):
            self.kill()

    class Context:
        def __init__(self):
            self.processes = []

        def Pipe(self, duplex=True):
            return Connection(), Connection()

        def Process(self, **_kwargs):
            process = DelayedDeathProcess()
            self.processes.append(process)
            return process

    context = Context()
    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=context,
        stop_timeout_seconds=0.6,
    )

    async def scenario():
        started = time.monotonic()
        read_task = asyncio.create_task(
            broker.read_async(warmup_frames=1, timeout=0.01)
        )
        ticks = 0
        while not read_task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        with pytest.raises(TimeoutError, match="read deadline exceeded"):
            await read_task
        return time.monotonic() - started, ticks

    elapsed, ticks = asyncio.run(scenario())

    assert elapsed >= 0.25
    assert elapsed < 0.8
    assert ticks >= 30
    assert broker.assert_dead()
    assert broker.active_request_count == 0


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


def test_abort_async_waits_for_process_sentinel_without_blocking_event_loop(
    monkeypatch,
):
    class SentinelProcess:
        pid = 9595
        sentinel = 77

        def __init__(self):
            self.exitcode = None
            self.join_calls = []

        def is_alive(self):
            return self.exitcode is None

        def kill(self):
            return None

        def terminate(self):
            return None

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            while True:
                time.sleep(1)

    process = SentinelProcess()

    def wait_for_exit(sentinels, timeout):
        assert sentinels == [process.sentinel]
        assert timeout > 0
        time.sleep(0.08)
        process.exitcode = 9
        return [process.sentinel]

    monkeypatch.setattr(directshow_broker, "wait_for_sentinels", wait_for_exit)
    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.2
    )
    broker._process = process

    async def scenario():
        stop_task = asyncio.create_task(broker.abort_async(reason="replacement"))
        ticks = 0
        while not stop_task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        return await stop_task, ticks

    stopped, ticks = asyncio.run(scenario())

    assert stopped is True
    assert ticks >= 8
    assert process.join_calls == []
    assert broker.assert_dead()


def test_release_confirms_liveness_after_sentinel_becomes_ready(monkeypatch):
    class SentinelRaceProcess:
        pid = 9646
        sentinel = 79
        exitcode = None

        def __init__(self):
            self.dead_at = None
            self.join_calls = []

        def is_alive(self):
            if self.dead_at is None or time.monotonic() < self.dead_at:
                return True
            self.exitcode = 9
            return False

        def kill(self):
            self.dead_at = time.monotonic() + 0.03

        def terminate(self):
            self.kill()

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            while True:
                time.sleep(1)

    process = SentinelRaceProcess()
    monkeypatch.setattr(
        directshow_broker,
        "wait_for_sentinels",
        lambda _sentinels, timeout: [process.sentinel],
    )
    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.1
    )
    broker._process = process

    assert broker.release() is True
    assert process.join_calls == []
    assert broker.assert_dead()


def test_abort_async_finishes_sentinel_wait_after_cancellation(monkeypatch):
    class SentinelProcess:
        pid = 9696
        sentinel = 88

        def __init__(self):
            self.exitcode = None

        def is_alive(self):
            return self.exitcode is None

        def kill(self):
            return None

        def terminate(self):
            return None

    process = SentinelProcess()
    wait_entered = threading.Event()
    wait_finished = threading.Event()

    def wait_for_exit(_sentinels, timeout):
        assert timeout > 0
        wait_entered.set()
        time.sleep(0.08)
        process.exitcode = 9
        wait_finished.set()
        return [process.sentinel]

    monkeypatch.setattr(directshow_broker, "wait_for_sentinels", wait_for_exit)
    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.2
    )
    broker._process = process

    async def scenario():
        stop_task = asyncio.create_task(broker.abort_async(reason="replacement"))
        while not wait_entered.is_set() and not stop_task.done():
            await asyncio.sleep(0.002)
        assert wait_entered.is_set(), f"stop completed early: {stop_task.result()}"
        stop_task.cancel()
        return await stop_task

    assert asyncio.run(scenario()) is True
    assert wait_finished.is_set()
    assert broker.assert_dead()


def test_abort_async_invalid_sentinel_falls_back_to_async_liveness_poll():
    class InvalidSentinelProcess:
        pid = 9797
        exitcode = None

        def __init__(self):
            self.dead_at = None

        @property
        def sentinel(self):
            raise ValueError("invalid sentinel")

        def is_alive(self):
            if self.dead_at is None or time.monotonic() < self.dead_at:
                return True
            self.exitcode = 9
            return False

        def kill(self):
            self.dead_at = time.monotonic() + 0.05

        def terminate(self):
            self.kill()

    process = InvalidSentinelProcess()
    broker = DirectShowCameraBroker(
        "front", _broker_config(), stop_timeout_seconds=0.2
    )
    broker._process = process

    async def scenario():
        stop_task = asyncio.create_task(broker.abort_async(reason="replacement"))
        ticks = 0
        while not stop_task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        return await stop_task, ticks

    stopped, ticks = asyncio.run(scenario())

    assert stopped is True
    assert ticks >= 5
    assert broker.assert_dead()


def test_async_request_cancel_waits_for_error_cleanup_blocked_on_stop_lock():
    request_polled = threading.Event()

    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            request_polled.set()
            return False

        def close(self):
            return None

    class Process:
        pid = 9898
        sentinel = None
        exitcode = None

        def start(self):
            return None

        def is_alive(self):
            return self.exitcode is None

        def kill(self):
            self.exitcode = 9

        def terminate(self):
            self.exitcode = 9

    class Context:
        def Pipe(self, duplex=True):
            return Connection(), Connection()

        def Process(self, **_kwargs):
            return Process()

    broker = DirectShowCameraBroker(
        "front",
        _broker_config(),
        context=Context(),
        stop_timeout_seconds=0.2,
    )

    async def scenario():
        await broker._async_stop_lock.acquire()
        read_task = asyncio.create_task(
            broker.read_async(warmup_frames=1, timeout=0.01)
        )
        while not request_polled.is_set():
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.02)
        read_task.cancel()
        await asyncio.sleep(0)
        read_task.cancel()
        await asyncio.sleep(0)
        canceled_before_cleanup = read_task.done()
        broker._async_stop_lock.release()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(read_task, timeout=0.5)
        slot_reacquired = broker._request_slot.acquire(blocking=False)
        if slot_reacquired:
            broker._request_slot.release()
        return canceled_before_cleanup, slot_reacquired

    canceled_before_cleanup, slot_reacquired = asyncio.run(scenario())

    assert canceled_before_cleanup is False
    assert slot_reacquired is True
    assert broker.active_request_count == 0
    assert broker._request_threads == set()
    assert broker.assert_dead()
