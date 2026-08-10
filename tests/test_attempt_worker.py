import asyncio
import hashlib
import multiprocessing
import os
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.attempt_worker import AttemptWorkerError, FastRenderBroker, render_attempt_frame
from vision.render_worker_target import _render, render_worker_entry


_TEST_POSE_FIXTURE = {
    "testOnly": True,
    "landmarks": {
        11: (0.35, 0.32, 0.95),
        12: (0.65, 0.32, 0.95),
        23: (0.38, 0.68, 0.95),
        24: (0.62, 0.68, 0.95),
    },
}


def _test_fixture_render_worker_target(connection, test_pose_fixture):
    """Spawn-safe test worker; it is intentionally outside packaged Vision code."""
    from multiprocessing import shared_memory

    from vision.fast_tryon import (
        FastTryOnRuntime,
        GarmentFetchError,
        PoseUnavailableError,
        ValidatedGarmentSource,
    )

    class TestFixturePoseEstimator:
        def detect(self, _frame):
            points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
            for index, point in test_pose_fixture["landmarks"].items():
                x, y, visibility = point
                points[index] = SimpleNamespace(x=x, y=y, visibility=visibility)
            return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))

    runtime = FastTryOnRuntime(pose_estimator=TestFixturePoseEstimator())
    connection.send(("ready", {"pid": os.getpid(), "poseReady": True}))
    try:
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            try:
                frame_shared = payload["frameShared"]
                shm = shared_memory.SharedMemory(name=frame_shared["name"])
                try:
                    frame = np.ndarray(
                        tuple(frame_shared["shape"]),
                        dtype=np.uint8,
                        buffer=shm.buf,
                    ).copy()
                finally:
                    shm.close()
                garment = payload["garmentPng"]
                digest = "sha256:" + hashlib.sha256(garment).hexdigest()
                source = ValidatedGarmentSource(
                    png_bytes=garment,
                    digest=digest,
                    template=payload["template"],
                )
                connection.send(("ok", runtime.render(frame, source)))
            except BaseException as exc:
                kind = (
                    "garment_error"
                    if isinstance(exc, GarmentFetchError)
                    else "pose_error"
                    if isinstance(exc, PoseUnavailableError)
                    else "error"
                )
                connection.send((kind, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _block_first_render_target(connection, counter):
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number == 1:
                while True:
                    time.sleep(1)
            image = np.full((24, 18, 3), (20, 120, 220), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            connection.send(("ok", encoded.tobytes()))
    finally:
        connection.close()


def _crash_first_render_target(connection, counter):
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number == 1:
                os._exit(23)
            image = np.full((24, 18, 3), (20, 120, 220), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            connection.send(("ok", encoded.tobytes()))
    finally:
        connection.close()


def _pose_error_then_success_target(connection, counter):
    """Test-only deterministic worker response fixture: no pose model boundary."""
    connection.send(("ready", {"pid": os.getpid(), "poseReady": True}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number <= 3:
                connection.send(("pose_error", "PoseUnavailableError: C:\\internal\\model\\details"))
                continue
            image = np.full((24, 18, 3), (20, 120, 220), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            connection.send(("ok", encoded.tobytes()))
    finally:
        connection.close()


def _blocked_worker_encode_target(connection, entered, test_pose_fixture):
    import vision.fast_tryon as fast_tryon

    def blocked_encode(*_args, **_kwargs):
        entered.set()
        while True:
            time.sleep(1)

    fast_tryon.cv2.imencode = blocked_encode
    _test_fixture_render_worker_target(connection, test_pose_fixture)


def _slow_worker_encode_target(connection, entered, delay_seconds, test_pose_fixture):
    import vision.fast_tryon as fast_tryon

    real_encode = fast_tryon.cv2.imencode

    def slow_encode(*args, **kwargs):
        entered.set()
        time.sleep(delay_seconds)
        return real_encode(*args, **kwargs)

    fast_tryon.cv2.imencode = slow_encode
    _test_fixture_render_worker_target(connection, test_pose_fixture)


def test_production_render_target_rejects_test_arguments():
    with pytest.raises(ValueError, match="does not accept test arguments"):
        FastRenderBroker(
            target=render_worker_entry, target_args=(_TEST_POSE_FIXTURE,)
        )


@pytest.mark.parametrize("compression", ["compressible", "difficult"])
def test_prestarted_render_broker_rejects_real_max_images_without_blocking_or_leaking(
    compression,
):
    """A 10ms deadline is rejected before encoding/IPC for real legal images."""
    garment = np.zeros((4096, 4096, 4), dtype=np.uint8)
    garment[384:3712, 640:3456] = (20, 120, 220, 255)
    if compression == "difficult":
        rng = np.random.default_rng(20260810)
        garment[1400:2400, 1548:2548] = rng.integers(
            0, 256, size=(1000, 1000, 4), dtype=np.uint8
        )
    ok, encoded_garment = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded_garment.tobytes()
    assert len(garment_png) <= 8 * 1024 * 1024
    if compression == "compressible":
        assert len(garment_png) < 1024 * 1024
    else:
        assert len(garment_png) > 3 * 1024 * 1024
    frame = np.random.default_rng(4).integers(
        0, 256, size=(720, 1280, 3), dtype=np.uint8
    )

    async def scenario():
        broker = FastRenderBroker()
        await broker.start()
        assert broker.ready
        broker_child = broker.pid
        assert broker_child is not None
        baseline_children = {child.pid for child in multiprocessing.active_children()}
        baseline_threads = {thread.ident for thread in threading.enumerate()}
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
        await asyncio.sleep(0.005)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=0.01,
                broker=broker,
            )
        elapsed = time.monotonic() - started
        await asyncio.sleep(0.005)
        stop.set()
        await ticker_task

        assert elapsed < 0.1
        assert max(gaps, default=0) < 0.03
        assert broker.pid == broker_child
        assert {
            child.pid for child in multiprocessing.active_children()
        } == baseline_children
        assert {thread.ident for thread in threading.enumerate()} == baseline_threads
        await broker.shutdown()
        assert broker.pid is None

    asyncio.run(scenario())


def test_prestarted_render_broker_completes_one_real_encoded_job():
    garment = np.zeros((256, 192, 4), dtype=np.uint8)
    garment[8:248, 12:180] = (20, 120, 220, 220)
    ok, encoded_garment = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded_garment.tobytes()
    frame = np.full((720, 1280, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        # This is a broker/resource test.  The explicit worker-only fixture
        # avoids claiming a blank frame is a real production person.
        broker = FastRenderBroker(
            target=_test_fixture_render_worker_target,
            target_args=(_TEST_POSE_FIXTURE,),
        )
        await broker.start()
        pid = broker.pid
        result = await render_attempt_frame(
            frame,
            garment_png,
            digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
            template="tshirt_short_sleeve",
            timeout=5.0,
            broker=broker,
        )
        assert broker.pid == pid
        await broker.shutdown()
        return result

    decoded = cv2.imdecode(
        np.frombuffer(asyncio.run(scenario()), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape == frame.shape
    assert not np.array_equal(decoded, frame)


def test_production_render_request_does_not_call_parent_connection_send(monkeypatch):
    """Fast render request metadata must not depend on parent-side Pipe send."""
    from multiprocessing.connection import Connection

    garment = np.zeros((256, 192, 4), dtype=np.uint8)
    garment[8:248, 12:180] = (20, 120, 220, 220)
    ok, encoded_garment = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded_garment.tobytes()
    frame = np.full((96, 72, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            target=_test_fixture_render_worker_target,
            target_args=(_TEST_POSE_FIXTURE,),
        )
        await broker.start()
        original_send = Connection.send
        parent_send_called = threading.Event()

        def blocked_parent_send(self, _payload):
            parent_send_called.set()
            raise AssertionError("parent request channel must not use Connection.send")

        monkeypatch.setattr(Connection, "send", blocked_parent_send)
        try:
            result = await asyncio.wait_for(
                render_attempt_frame(
                    frame,
                    garment_png,
                    digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                    template="tshirt_short_sleeve",
                    timeout=5.0,
                    broker=broker,
                ),
                timeout=6.0,
            )
        finally:
            monkeypatch.setattr(Connection, "send", original_send)
            await broker.shutdown()
        return parent_send_called.is_set(), result, broker.active_request_count

    parent_send_called, result, active_requests = asyncio.run(scenario())
    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert not parent_send_called
    assert active_requests == 0


def test_oversized_frame_is_rejected_before_parent_copy():
    copied = threading.Event()

    class CopyTrap(np.ndarray):
        def tobytes(self, *args, **kwargs):
            copied.set()
            raise AssertionError("oversized frame must not be copied")

    frame = np.zeros((1081, 1920, 3), dtype=np.uint8).view(CopyTrap)

    with pytest.raises(ValueError, match="dimensions"):
        asyncio.run(
            render_attempt_frame(
                frame,
                b"valid-enough-for-preflight",
                digest="sha256:unused",
                template="tshirt_short_sleeve",
                timeout=5.0,
                broker=object(),
            )
        )

    assert not copied.is_set()


@pytest.mark.parametrize(
    "frame_shared",
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
            "name": "vem_render_valid_name_but_extra_key",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
            "extra": "rejected",
        },
        {
            "kind": "shared_frame",
            "name": "vem_render_too_tall",
            "shape": [1440, 8, 3],
            "dtype": "uint8",
            "nbytes": 1440 * 8 * 3,
            "generation": 1,
            "processGeneration": 1,
        },
        {
            "kind": "shared_frame",
            "name": "vem_render_bool_generation",
            "shape": [8, 8, 3],
            "dtype": "uint8",
            "nbytes": 8 * 8 * 3,
            "generation": True,
            "processGeneration": 1,
        },
    ],
)
def test_render_worker_rejects_strict_frame_metadata_before_arbitrary_shm_attach(
    monkeypatch, frame_shared
):
    attached = threading.Event()

    def forbidden_attach(*_args, **_kwargs):
        attached.set()
        raise AssertionError("invalid metadata must be rejected before shm attach")

    monkeypatch.setattr("vision.render_worker_target.shared_memory.SharedMemory", forbidden_attach)

    with pytest.raises(ValueError):
        _render(
            {
                "frameShared": frame_shared,
                "garmentPng": b"not-attached-before-digest",
                "garmentDigest": "sha256:unused",
                "template": "tshirt_short_sleeve",
            }
        )
    assert not attached.is_set()


def test_parent_cv2_encode_block_cannot_enter_the_prestarted_render_path(monkeypatch):
    garment = np.zeros((256, 192, 4), dtype=np.uint8)
    garment[8:248, 12:180] = (20, 120, 220, 220)
    ok, encoded_garment = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded_garment.tobytes()
    frame = np.full((720, 1280, 3), (235, 220, 205), dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()

    def blocked_parent_encode(*_args, **_kwargs):
        entered.set()
        release.wait()
        raise AssertionError("parent cv2.imencode must not run")

    async def scenario():
        broker = FastRenderBroker(
            target=_test_fixture_render_worker_target,
            target_args=(_TEST_POSE_FIXTURE,),
        )
        await broker.start()
        monkeypatch.setattr(cv2, "imencode", blocked_parent_encode)
        task = asyncio.create_task(
            render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=5.0,
                broker=broker,
            )
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while not task.done() and not entered.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        parent_encode_entered = entered.is_set()
        if parent_encode_entered:
            task.cancel()
            await asyncio.sleep(0.05)
            release.set()
        result = await asyncio.gather(task, return_exceptions=True)
        await broker.shutdown()
        return parent_encode_entered, result[0]

    parent_encode_entered, result = asyncio.run(scenario())

    assert not parent_encode_entered
    assert isinstance(result, bytes)
    assert not any(
        thread.name == "fast-render-encode" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_cancel_joins_blocked_native_encode_then_readies_one_replacement():
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_blocked_worker_encode_target,
            target_args=(entered, _TEST_POSE_FIXTURE),
        )
        await broker.start()
        first_pid = broker.pid
        task = asyncio.create_task(
            render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=10.0,
                broker=broker,
            )
        )
        assert await asyncio.to_thread(entered.wait, 2.0)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - started
        state = (
            first_pid,
            broker.ready,
            broker.pid,
            broker.active_request_count,
            {child.pid for child in multiprocessing.active_children()},
        )
        await broker.shutdown()
        return elapsed, state

    elapsed, (first_pid, ready, pid, active_requests, child_pids) = asyncio.run(
        scenario()
    )

    assert elapsed < 1.0
    assert ready
    assert pid is not None and pid != first_pid
    assert active_requests == 0
    assert child_pids == {pid}
    assert not any(
        thread.name == "fast-render-encode" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_worker_slow_encode_times_out_then_readies_one_replacement():
    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_slow_worker_encode_target,
            target_args=(entered, 0.2, _TEST_POSE_FIXTURE),
        )
        await broker.start()
        first_pid = broker.pid
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=0.1,
                broker=broker,
            )
        elapsed = time.monotonic() - started
        state = (
            entered.is_set(),
            first_pid,
            broker.ready,
            broker.pid,
            broker.active_request_count,
            {child.pid for child in multiprocessing.active_children()},
        )
        await broker.shutdown()
        return elapsed, state

    elapsed, (
        encode_entered,
        first_pid,
        ready,
        pid,
        active_requests,
        child_pids,
    ) = asyncio.run(scenario())

    assert encode_entered
    assert 0.08 <= elapsed < 1.0
    assert ready
    assert pid is not None and pid != first_pid
    assert active_requests == 0
    assert child_pids == {pid}


def test_cancelled_blocked_render_is_joined_before_controlled_recovery():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.zeros((64, 48, 4), dtype=np.uint8)
    garment[4:60, 4:44] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_block_first_render_target,
            target_args=(counter,),
        )
        await broker.start()
        first_pid = broker.pid
        task = asyncio.create_task(
            render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=10.0,
                broker=broker,
            )
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while counter.value < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert counter.value == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert broker.ready
        assert broker.pid is not None and broker.pid != first_pid
        assert broker.active_request_count == 0
        assert {child.pid for child in multiprocessing.active_children()} == {
            broker.pid
        }
        await broker.shutdown()

    asyncio.run(scenario())


def test_crashed_render_is_joined_and_one_replacement_is_prestarted():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_crash_first_render_target,
            target_args=(counter,),
        )
        await broker.start()
        crashed_pid = broker.pid
        with pytest.raises(AttemptWorkerError, match="render broker"):
            await render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=5.0,
                broker=broker,
            )
        assert broker.ready
        assert broker.pid is not None and broker.pid != crashed_pid
        assert broker.active_request_count == 0
        assert {child.pid for child in multiprocessing.active_children()} == {broker.pid}

        result = await render_attempt_frame(
            frame,
            garment_png,
            digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
            template="tshirt_short_sleeve",
            timeout=5.0,
            broker=broker,
        )
        assert cv2.imdecode(
            np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR
        ) is not None
        await broker.shutdown()

    asyncio.run(scenario())


def test_pose_failures_are_typed_attempt_outcomes_and_keep_the_worker_pid():
    """Blank/no-person/degenerate pose failures do not kill a healthy worker."""
    from vision.fast_tryon import PoseUnavailableError

    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)
    digest = "sha256:" + hashlib.sha256(garment_png).hexdigest()

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_pose_error_then_success_target,
            target_args=(counter,),
        )
        await broker.start()
        pid = broker.pid
        assert pid is not None
        for _ in range(3):
            with pytest.raises(PoseUnavailableError) as error:
                await render_attempt_frame(
                    frame, garment_png, digest=digest,
                    template="tshirt_short_sleeve", timeout=5.0, broker=broker,
                )
            assert str(error.value) == "pose_unavailable"
            assert broker.pid == pid
            assert broker.ready
        result = await render_attempt_frame(
            frame, garment_png, digest=digest,
            template="tshirt_short_sleeve", timeout=5.0, broker=broker,
        )
        assert broker.pid == pid
        assert counter.value == 4
        await broker.shutdown()
        return result

    assert cv2.imdecode(np.frombuffer(asyncio.run(scenario()), dtype=np.uint8), cv2.IMREAD_COLOR) is not None


def test_stubborn_kill_oserror_retains_handle_and_fails_closed_without_restart():
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

    async def scenario():
        broker = FastRenderBroker()
        broker._process = KillDeniedProcess()
        broker._ready = True
        assert broker.ready
        with pytest.raises(AttemptWorkerError, match="shutdown incomplete"):
            await broker.shutdown()
        assert not broker.ready
        assert broker.pid == 9292
        with pytest.raises(AttemptWorkerError, match="unavailable"):
            await broker.start()

    asyncio.run(scenario())


def test_concurrent_render_start_is_rejected_without_worker_or_queue_growth():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    digest = "sha256:" + hashlib.sha256(garment_png).hexdigest()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_block_first_render_target,
            target_args=(counter,),
        )
        await broker.start()
        first = asyncio.create_task(
            render_attempt_frame(
                frame,
                garment_png,
                digest=digest,
                template="tshirt_short_sleeve",
                timeout=10.0,
                broker=broker,
            )
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while counter.value < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert counter.value == 1

        with pytest.raises(AttemptWorkerError, match="active job"):
            await render_attempt_frame(
                frame,
                garment_png,
                digest=digest,
                template="tshirt_short_sleeve",
                timeout=5.0,
                broker=broker,
            )
        assert counter.value == 1
        assert broker.active_request_count == 1
        assert {child.pid for child in multiprocessing.active_children()} == {broker.pid}

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert broker.active_request_count == 0
        await broker.shutdown()

    asyncio.run(scenario())


def test_shutdown_joins_a_blocked_render_without_starting_a_replacement():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_block_first_render_target,
            target_args=(counter,),
        )
        await broker.start()
        task = asyncio.create_task(
            render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=10.0,
                broker=broker,
            )
        )
        deadline = asyncio.get_running_loop().time() + 2.0
        while counter.value < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        assert counter.value == 1

        started = time.monotonic()
        await broker.shutdown()
        elapsed = time.monotonic() - started
        assert elapsed < 0.05
        assert broker.active_request_count == 0
        with pytest.raises(AttemptWorkerError):
            await task
        assert broker.pid is None
        assert not broker.ready
        assert broker.active_request_count == 0
        assert multiprocessing.active_children() == []

    asyncio.run(scenario())


def test_blocked_render_timeout_joins_before_controlled_recovery():
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    garment = np.full((64, 48, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    garment_png = encoded.tobytes()
    frame = np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

    async def scenario():
        broker = FastRenderBroker(
            context=context,
            target=_block_first_render_target,
            target_args=(counter,),
        )
        await broker.start()
        first_pid = broker.pid
        with pytest.raises(TimeoutError):
            await render_attempt_frame(
                frame,
                garment_png,
                digest="sha256:" + hashlib.sha256(garment_png).hexdigest(),
                template="tshirt_short_sleeve",
                timeout=0.1,
                broker=broker,
            )
        assert broker.ready
        assert broker.pid is not None and broker.pid != first_pid
        assert broker.active_request_count == 0
        assert {child.pid for child in multiprocessing.active_children()} == {
            broker.pid
        }
        await broker.shutdown()

    asyncio.run(scenario())
