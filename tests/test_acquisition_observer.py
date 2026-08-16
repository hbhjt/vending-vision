import asyncio
import json
import multiprocessing
import os
import signal
import threading
import time

import pytest
import numpy as np

from vision.acquisition_observer import (
    AcquisitionObservation,
    AcquisitionObservationWorker,
    _read_shared_frame,
)
from vision.shared_ipc_slot import _HEADER


def _blocking_observer_target(connection):
    try:
        connection.send(("ready", None))
        while True:
            command, _payload = connection.recv()
            if command == "observe":
                while True:
                    time.sleep(1)
    finally:
        connection.close()


def _counting_observer_target(connection, starts):
    with starts.get_lock():
        starts.value += 1
    connection.send(("ready", None))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "observe":
                connection.send(("ok", AcquisitionObservation(b"jpeg", "single", True)))
            elif command == "shutdown":
                connection.send(("ok", None))
                return
    finally:
        connection.close()


def _corrupt_ready_observer_target(connection):
    message = {
        "kind": "ready",
        "payload": {"pid": os.getpid()},
        "processGeneration": 999,
        "requestGeneration": 0,
        "responseBytes": 0,
    }
    encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
    _request_json, response_json, _request_bytes, _response_bytes = connection._offsets()
    with connection._lock:
        request_json_len, _old_response, request_bytes_len, _old_bytes = _HEADER.unpack_from(
            connection._shm.buf, 0
        )
        _HEADER.pack_into(
            connection._shm.buf,
            0,
            request_json_len,
            len(encoded),
            request_bytes_len,
            0,
        )
        connection._shm.buf[response_json : response_json + len(encoded)] = encoded
        connection._response_event.set()
    while True:
        time.sleep(1)


def test_single_slot_observation_keeps_the_event_loop_responsive_until_cancel_cleanup():
    """A blocked production-boundary observation cannot starve ping/cancel work."""
    async def scenario():
        worker = AcquisitionObservationWorker(
            context=multiprocessing.get_context("spawn"), target=_blocking_observer_target
        )
        request = None
        try:
            request = asyncio.create_task(
                worker.observe(np.zeros((8, 8, 3), dtype=np.uint8))
            )
            await asyncio.sleep(0.05)

            # Prove the caller-visible property directly: another coroutine
            # can be scheduled while the native observation remains blocked.
            loop_responsive = asyncio.Event()

            async def ping():
                await asyncio.sleep(0)
                loop_responsive.set()

            asyncio.create_task(ping())
            await asyncio.wait_for(loop_responsive.wait(), timeout=1.0)
            assert not request.done()

            second = asyncio.create_task(worker.observe(object()))
            with pytest.raises(RuntimeError, match="busy"):
                await second
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(request, timeout=1)
            assert worker.assert_dead
        finally:
            if request is not None and not request.done():
                request.cancel()
                try:
                    await asyncio.wait_for(request, timeout=1)
                except (asyncio.CancelledError, TimeoutError):
                    pass
            await worker.shutdown()

    asyncio.run(scenario())


def test_normal_acquisition_observer_reuses_one_prewarmed_child_for_multiple_observations():
    """Production YOLO/Pose startup is one child lifetime, not one import per frame."""
    async def scenario():
        context = multiprocessing.get_context("spawn")
        starts = context.Value("i", 0)
        worker = AcquisitionObservationWorker(
            context=context,
            target=_counting_observer_target,
            target_args=(starts,),
        )
        first = await worker.observe(np.zeros((8, 8, 3), dtype=np.uint8))
        first_pid = worker.pid
        second = await worker.observe(np.ones((8, 8, 3), dtype=np.uint8))
        assert first == AcquisitionObservation(b"jpeg", "single", True)
        assert second == first
        assert starts.value == 1
        assert worker.pid == first_pid
        assert worker.active_request_count == 0
        await worker.shutdown()
        assert worker.assert_dead

    asyncio.run(scenario())


def test_acquisition_spawn_failure_unlinks_slot_and_closes_process_handle():
    real_context = multiprocessing.get_context("spawn")

    class StartFailureProcess:
        def __init__(self):
            self.closed = False

        def start(self):
            raise OSError("spawn denied")

        def close(self):
            self.closed = True

    class StartFailureContext:
        def __init__(self):
            self.process = StartFailureProcess()

        def Event(self):
            return real_context.Event()

        def Lock(self):
            return real_context.Lock()

        def Process(self, **_kwargs):
            return self.process

    before = set(__import__("os").listdir("/dev/shm")) if __import__("os").path.isdir("/dev/shm") else set()
    context = StartFailureContext()
    worker = AcquisitionObservationWorker(context=context)

    with pytest.raises(RuntimeError, match="acquisition_observer_start_failed"):
        worker._start()

    after = set(__import__("os").listdir("/dev/shm")) if __import__("os").path.isdir("/dev/shm") else set()
    assert context.process.closed is True
    assert {name for name in after - before if "vem_acq" in name} == set()
    assert worker.pid is None
    assert worker.ready is False
    with pytest.raises(RuntimeError, match="acquisition_observer_start_failed"):
        worker._start()


def test_live_acquisition_prewarm_corrupt_ready_stops_child_and_fails_closed():
    async def scenario():
        worker = AcquisitionObservationWorker(
            context=multiprocessing.get_context("spawn"),
            target=_corrupt_ready_observer_target,
        )
        with pytest.raises(RuntimeError, match="prewarm"):
            await worker.start()
        assert worker.ready is False
        assert worker.pid is None
        assert worker.active_request_count == 0
        assert worker._slot is None
        assert worker.fatal_error is not None
        assert "prewarm" in worker.fatal_error
        assert {
            child.pid for child in multiprocessing.active_children()
        } == set()
        with pytest.raises(RuntimeError, match="unavailable"):
            await worker.start()

    asyncio.run(scenario())


def test_production_observation_request_does_not_call_parent_connection_send(monkeypatch):
    """A normal request must complete even if parent-side Pipe send is unusable."""
    from multiprocessing.connection import Connection

    async def scenario():
        context = multiprocessing.get_context("spawn")
        starts = context.Value("i", 0)
        worker = AcquisitionObservationWorker(
            context=context,
            target=_counting_observer_target,
            target_args=(starts,),
        )
        await worker.start()
        original_send = Connection.send
        parent_send_called = threading.Event()

        def blocked_parent_send(self, _payload):
            parent_send_called.set()
            raise AssertionError("parent request channel must not use Connection.send")

        monkeypatch.setattr(Connection, "send", blocked_parent_send)
        try:
            result = await asyncio.wait_for(
                worker.observe(np.zeros((8, 8, 3), dtype=np.uint8), timeout=2.0),
                timeout=3.0,
            )
        finally:
            monkeypatch.setattr(Connection, "send", original_send)
            await worker.shutdown()

        assert result == AcquisitionObservation(b"jpeg", "single", True)
        assert not parent_send_called.is_set()
        assert worker.active_request_count == 0

    asyncio.run(scenario())


class _PermissionDeniedProcess:
    pid = 999999

    def __init__(self):
        self.kill_attempted = False
        self.terminate_attempted = False
        self.join_calls = 0

    def is_alive(self):
        return True

    def kill(self):
        self.kill_attempted = True
        raise PermissionError("access denied")

    def terminate(self):
        self.terminate_attempted = True
        raise OSError("terminate denied")

    def join(self, timeout=None):
        self.join_calls += 1


class _AliveProcess:
    pid = 100001

    def __init__(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def kill(self):
        self._alive = False
        return None

    def terminate(self):
        self._alive = False
        return None

    def join(self, timeout=None):
        return None


class _DeadProcess:
    pid = 100002

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _BlockingSendConnection:
    def __init__(self):
        self.send_entered = threading.Event()
        self.release_send = threading.Event()
        self.closed = False
        self.large_payload_send_count = 0

    def send(self, payload):
        self.send_entered.set()
        if _payload_contains_large_frame(payload):
            self.large_payload_send_count += 1
            assert self.release_send.wait(timeout=2.0)

    def poll(self, _timeout=0):
        return False

    def recv(self):
        raise AssertionError("recv should not be reached by the blocking send probe")

    def close(self):
        self.closed = True


def _payload_contains_large_frame(payload):
    try:
        import numpy as np

        if isinstance(payload, np.ndarray):
            return payload.nbytes > 1024
    except Exception:
        pass
    if isinstance(payload, bytes):
        return len(payload) > 1024
    if isinstance(payload, dict):
        return any(_payload_contains_large_frame(value) for value in payload.values())
    if isinstance(payload, (tuple, list)):
        return any(_payload_contains_large_frame(value) for value in payload)
    return False


def test_observation_abort_is_bounded_when_parent_connection_send_blocks(monkeypatch):
    """Cancellation must not wait on any parent request Connection.send path."""
    from multiprocessing.connection import Connection

    async def scenario():
        import numpy as np

        worker = AcquisitionObservationWorker(
            context=multiprocessing.get_context("spawn"), target=_blocking_observer_target
        )
        await worker.start()

        def blocked_parent_send(self, payload):
            raise AssertionError("parent request channel must not use Connection.send")

        monkeypatch.setattr(Connection, "send", blocked_parent_send)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        request = asyncio.create_task(worker.observe(frame))
        deadline = time.monotonic() + 1.0
        while worker.active_request_count == 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.002)
        assert worker.active_request_count == 1

        abort = asyncio.create_task(worker.abort_async(reason="replacement"))
        done, _ = await asyncio.wait({abort}, timeout=0.2)
        try:
            assert abort in done, "abort waited behind a request thread blocked in Connection.send"
            assert worker.active_request_count == 0
        finally:
            with pytest.raises(Exception):
                await asyncio.wait_for(request, timeout=1.0)
            if not abort.done():
                await asyncio.wait_for(abort, timeout=1.0)

    asyncio.run(scenario())


def test_observer_abort_async_runs_process_control_off_the_event_loop():
    class SlowKillProcess:
        pid = 7474
        exitcode = None

        def __init__(self):
            self._alive = True

        def is_alive(self):
            return self._alive

        def kill(self):
            time.sleep(0.08)
            self._alive = False

        def terminate(self):
            self._alive = False

        def join(self, timeout=None):
            return None

    async def scenario():
        worker = AcquisitionObservationWorker()
        worker._process = SlowKillProcess()
        worker._ready = True
        worker._generation = 1
        abort = asyncio.create_task(worker.abort_async(reason="replacement"))
        ticks = 0
        deadline = asyncio.get_running_loop().time() + 0.04
        while asyncio.get_running_loop().time() < deadline:
            ticks += 1
            await asyncio.sleep(0.001)
        assert ticks >= 5
        assert await asyncio.wait_for(abort, timeout=0.5) is True
        assert worker.assert_dead

    asyncio.run(scenario())


def test_repeated_cancel_waits_for_delayed_observer_death_without_starving_loop():
    class DelayedDeathProcess:
        pid = 7484
        exitcode = None
        sentinel = 7485

        def __init__(self):
            self.killed_at = None
            self.closed = False
            self.death_released = threading.Event()

        def is_alive(self):
            return self.killed_at is None or not self.death_released.is_set()

        def kill(self):
            if self.killed_at is None:
                self.killed_at = time.monotonic()

        def terminate(self):
            self.kill()

        def close(self):
            self.closed = True

    class Slot:
        def __init__(self):
            self.closed = False

        def submit(self, *_args, **_kwargs):
            return None

        def poll_response(self):
            return False

        def close(self, *, unlink):
            assert unlink is True
            self.closed = True

    async def scenario():
        worker = AcquisitionObservationWorker(stop_timeout_seconds=0.5)
        process = DelayedDeathProcess()
        slot = Slot()
        worker._process = process
        worker._slot = slot
        worker._ready = True
        worker._generation = 1

        async def already_started():
            return None

        worker.start = already_started
        request = asyncio.create_task(
            worker.observe(np.zeros((8, 8, 3), dtype=np.uint8), timeout=2.0)
        )
        deadline = asyncio.get_running_loop().time() + 0.2
        while worker.active_request_count == 0 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert worker.active_request_count == 1

        request.cancel()
        await asyncio.sleep(0)
        request.cancel()
        await asyncio.sleep(0)
        request.cancel()

        # Repeated cancellation must not bypass the physical-death barrier.
        # Keep the fake process alive until the event loop has observed that
        # stable state, instead of depending on Windows timer granularity.
        for _iteration in range(3):
            assert worker.active_request_count == 1
            assert worker._process is process
            assert not request.done()
            await asyncio.sleep(0)
        process.death_released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)

        assert process.closed is True
        assert slot.closed is True
        assert worker.assert_dead
        assert worker.active_request_count == 0
        assert worker.fatal_error is None

    asyncio.run(scenario())


def test_public_start_revalidates_stale_ready_and_preserves_fatal_handle():
    """The public start gate must use ready/process/fatal, not the raw _ready flag."""
    async def scenario():
        context = multiprocessing.get_context("spawn")
        starts = context.Value("i", 0)
        worker = AcquisitionObservationWorker(
            context=context,
            target=_counting_observer_target,
            target_args=(starts,),
        )
        worker._ready = True
        worker._process = _DeadProcess()

        await worker.start()
        assert worker.ready
        assert starts.value == 1
        await worker.shutdown()

        fatal = AcquisitionObservationWorker()
        live = _AliveProcess()
        fatal._ready = True
        fatal._process = live
        fatal._fatal_error = "permission_denied"
        with pytest.raises(RuntimeError, match="permission_denied"):
            await fatal.start()
        assert fatal._process is live

    asyncio.run(scenario())


def test_acquisition_observer_abort_permission_errors_fail_closed_without_traceback(capsys):
    """Kill/terminate failures retain the live handle and mark observer unavailable."""
    worker = AcquisitionObservationWorker()
    process = _PermissionDeniedProcess()
    worker._process = process

    assert worker.abort(reason="permission_denied") is False

    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert process.kill_attempted is True
    assert process.terminate_attempted is True
    assert worker.pid == process.pid
    with pytest.raises(RuntimeError, match="permission_denied"):
        worker._start()


def test_observer_abort_async_does_not_call_stubborn_blocking_join_before_dead():
    class LiveProcess:
        pid = 7575
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

    async def scenario():
        worker = AcquisitionObservationWorker(stop_timeout_seconds=0.05)
        live = LiveProcess()
        worker._process = live
        worker._ready = True
        started = time.monotonic()
        result = await asyncio.wait_for(
            worker.abort_async(reason="replacement"), timeout=0.2
        )
        return worker, live, result, time.monotonic() - started

    worker, live, result, elapsed = asyncio.run(scenario())

    assert result is False
    assert elapsed < 0.2
    assert live.join_calls == []
    assert worker.assert_dead is False
    assert worker.ready is False
    assert worker.fatal_error == "replacement"
    assert not any(
        thread.name == "acquisition-observer-abort" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_async_acquisition_start_ticks_and_cancel_stops_new_child_before_ready():
    real_context = multiprocessing.get_context("spawn")

    class SlowStartProcess:
        pid = 7676
        exitcode = None

        def __init__(self, **_kwargs):
            self._alive = False
            self.kill_attempted = False

        def start(self):
            time.sleep(0.2)
            self._alive = True

        def is_alive(self):
            return self._alive

        def kill(self):
            self.kill_attempted = True
            self._alive = False

        def terminate(self):
            self._alive = False

        def close(self):
            return None

    class Context:
        def __init__(self):
            self.process = SlowStartProcess()

        def Event(self):
            return real_context.Event()

        def Lock(self):
            return real_context.Lock()

        def Process(self, **_kwargs):
            return self.process

    async def scenario():
        context = Context()
        worker = AcquisitionObservationWorker(context=context)
        task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.05)
        ticks = 0
        deadline = asyncio.get_running_loop().time() + 0.05
        while asyncio.get_running_loop().time() < deadline:
            ticks += 1
            task.cancel()
            await asyncio.sleep(0.005)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        return context.process, worker, ticks

    process, worker, ticks = asyncio.run(scenario())

    assert ticks >= 5
    assert process.kill_attempted is True
    assert process.is_alive() is False
    # A canceled start leaves no live child but also no fatal error: the
    # observer must remain healable so the next attempt can re-spawn it.
    assert worker.pid is None
    assert worker.fatal_error is None
    assert worker.ready is True


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "kind": "shared_frame",
            "name": "arbitrary",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
        },
        {
            "kind": "shared_frame",
            "name": "vem_acq_valid_name_but_extra_key",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
            "extra": "rejected",
        },
        {
            "kind": "shared_frame",
            "name": "vem_acq_too_tall",
            "shape": [2048, 8, 3],
            "dtype": "uint8",
            "nbytes": 2048 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
        },
        {
            "kind": "shared_frame",
            "name": "vem_acq_bool_generation",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": True,
            "processGeneration": 1,
        },
        {
            "kind": "shared_frame",
            "name": "vem_acq_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
        },
        {
            "kind": "shared_frame",
            "name": "vem_acq_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": 1,
            "processGeneration": 999,
        },
    ],
)
def test_acquisition_rejects_strict_frame_metadata_before_arbitrary_shm_attach(
    monkeypatch, metadata
):
    attached = threading.Event()

    def forbidden_attach(*_args, **_kwargs):
        attached.set()
        raise AssertionError("invalid metadata must be rejected before shm attach")

    monkeypatch.setattr("vision.acquisition_observer.shared_memory.SharedMemory", forbidden_attach)

    with pytest.raises(ValueError):
        _read_shared_frame(metadata, generation=1, process_generation=1)
    assert not attached.is_set()


def test_acquisition_observer_accepts_vertical_720x1280_frame():
    """A physical vertical 720x1280 camera frame must pass the acquisition cap."""
    from vision.acquisition_observer import (
        MAX_FRAME_HEIGHT,
        _coerce_frame_for_shared_memory,
    )

    assert MAX_FRAME_HEIGHT == 1920
    frame = np.zeros((1280, 720, 3), dtype=np.uint8)
    coerced = _coerce_frame_for_shared_memory(frame)
    assert coerced.shape == (1280, 720, 3)
    assert coerced.flags.c_contiguous

    with pytest.raises(ValueError, match="exceeds cap"):
        _coerce_frame_for_shared_memory(np.zeros((1921, 720, 3), dtype=np.uint8))


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGKILL process-crash tracer")
def test_acquisition_observer_respawns_after_child_death_and_recovers_ready():
    """A killed observer child must be replaced by the next start() instead of a permanent degrade."""
    async def scenario():
        context = multiprocessing.get_context("spawn")
        starts = context.Value("i", 0)
        worker = AcquisitionObservationWorker(
            context=context,
            target=_counting_observer_target,
            target_args=(starts,),
        )
        first = await worker.observe(np.zeros((8, 8, 3), dtype=np.uint8))
        assert first == AcquisitionObservation(b"jpeg", "single", True)
        assert starts.value == 1
        first_pid = worker.pid
        assert first_pid is not None

        # Simulate a hard child crash: the observer must not degrade forever.
        os.kill(first_pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while worker.pid is not None and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert worker.pid is None
        assert worker.fatal_error is None
        assert worker.ready is True  # process is gone but no fatal error: healable

        # The next observation must respawn a fresh child and complete.
        second = await asyncio.wait_for(
            worker.observe(np.ones((8, 8, 3), dtype=np.uint8), timeout=35.0),
            timeout=40.0,
        )
        assert second == AcquisitionObservation(b"jpeg", "single", True)
        assert starts.value == 2
        assert worker.pid is not None
        assert worker.pid != first_pid
        await worker.shutdown()
        assert worker.assert_dead

    asyncio.run(scenario())
