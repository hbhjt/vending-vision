import asyncio
import multiprocessing
import threading
import time

import pytest

from vision.acquisition_observer import AcquisitionObservation, AcquisitionObservationWorker


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


def test_single_slot_observation_keeps_the_event_loop_responsive_until_cancel_cleanup():
    """A blocked production-boundary observation cannot starve ping/cancel work."""
    async def scenario():
        worker = AcquisitionObservationWorker(
            context=multiprocessing.get_context("spawn"), target=_blocking_observer_target
        )
        request = asyncio.create_task(worker.observe(object()))
        await asyncio.sleep(0.05)
        ticks = 0
        deadline = asyncio.get_running_loop().time() + 0.04
        while asyncio.get_running_loop().time() < deadline:
            ticks += 1
            await asyncio.sleep(0.001)
        assert ticks >= 5
        second = asyncio.create_task(worker.observe(object()))
        with pytest.raises(RuntimeError, match="busy"):
            await second
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=1)
        assert worker.assert_dead
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
        first = await worker.observe(b"frame-1")
        first_pid = worker.pid
        second = await worker.observe(b"frame-2")
        assert first == AcquisitionObservation(b"jpeg", "single", True)
        assert second == first
        assert starts.value == 1
        assert worker.pid == first_pid
        assert worker.active_request_count == 0
        await worker.shutdown()
        assert worker.assert_dead

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
