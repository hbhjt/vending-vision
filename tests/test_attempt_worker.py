import asyncio
import hashlib
import multiprocessing
import os
import threading
import time

import cv2
import numpy as np
import pytest

from vision.attempt_worker import AttemptWorkerError, FastRenderBroker, render_attempt_frame


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
        broker = FastRenderBroker()
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


def test_stubborn_kill_oserror_retains_handle_and_fails_closed_without_restart():
    class Connection:
        def send(self, _message):
            return None

        def poll(self, _timeout):
            return True

        def recv(self):
            return "ready", {"pid": 9292}

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

    async def scenario():
        context = Context()
        broker = FastRenderBroker(context=context)
        await broker.start()
        assert broker.ready
        with pytest.raises(AttemptWorkerError, match="shutdown incomplete"):
            await broker.shutdown()
        assert not broker.ready
        assert broker.pid == 9292
        with pytest.raises(AttemptWorkerError, match="unavailable"):
            await broker.start()
        assert len(context.processes) == 1

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

        await broker.shutdown()
        assert broker.active_request_count == 0
        with pytest.raises(AttemptWorkerError):
            await task
        assert broker.pid is None
        assert not broker.ready
        assert broker.active_request_count == 0
        assert multiprocessing.active_children() == []

    asyncio.run(scenario())


def test_blocked_render_timeout_joins_before_return_and_prestarts_recovery():
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
        assert {child.pid for child in multiprocessing.active_children()} == {broker.pid}
        await broker.shutdown()

    asyncio.run(scenario())
