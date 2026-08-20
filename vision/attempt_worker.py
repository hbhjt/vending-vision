"""Lifecycle-owned, bounded Fast render broker.

The Vision application starts one spawn-compatible child before accepting Fast
attempt work.  Attempt requests submit at most one bounded raw-frame job and
never synchronously create a process on the event-loop thread.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_sentinels
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import numpy as np

from vision.render_worker_target import render_worker_entry
from vision.shared_ipc_slot import (
    SharedIpcError,
    SharedIpcSlot,
    run_shared_ipc_child,
    wait_for_event,
)


class AttemptWorkerError(RuntimeError):
    pass


_MAX_GARMENT_BYTES = 8 * 1024 * 1024
_MAX_FRAME_WIDTH = 1920
_MAX_FRAME_HEIGHT = 1920
_MAX_FRAME_RAW_BYTES = _MAX_FRAME_WIDTH * _MAX_FRAME_HEIGHT * 3
_CONSERVATIVE_PREPARE_BYTES_PER_SECOND = 64 * 1024 * 1024
_CONSERVATIVE_PREPARE_FIXED_SECONDS = 0.020
_START_TIMEOUT_SECONDS = 25.0
_STOP_CONFIRM_TIMEOUT_SECONDS = 1.0
_GRACEFUL_STOP_CONFIRM_TIMEOUT_SECONDS = 2.0


def _write_shared_render_frame(
    frame: np.ndarray, *, process_generation: int, request_generation: int
) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(
        create=True,
        size=int(frame.nbytes),
        name=f"vem_render_{uuid4().hex}",
    )
    try:
        shm.buf[: frame.nbytes] = frame.reshape(-1).view(np.uint8)
    except BaseException:
        try:
            shm.close()
        finally:
            shm.unlink()
        raise
    return (
        {
            "kind": "shared_frame",
            "name": shm.name,
            "shape": tuple(int(value) for value in frame.shape),
            "dtype": str(frame.dtype),
            "nbytes": int(frame.nbytes),
            "generation": int(request_generation),
            "processGeneration": int(process_generation),
        },
        shm,
    )


def _unlink_shared_render_frame(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


async def _run_broker_control(name: str, operation):
    """Run one bounded process-control operation off the event loop."""
    task = asyncio.create_task(asyncio.to_thread(operation), name=name)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _finish_cancelled_control(operation) -> Any:
    """Finish one shielded control coroutine despite repeated cancellation."""
    task = asyncio.create_task(operation)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _wait_process_dead(process: Any, timeout: float) -> bool:
    try:
        sentinel = getattr(process, "sentinel", None)
    except ValueError:
        return True
    if sentinel is not None:
        try:
            if wait_for_sentinels([sentinel], timeout=max(timeout, 0.0)):
                join = getattr(process, "join", None)
                if join is not None:
                    try:
                        join(timeout=0.25)
                    except (AssertionError, OSError, PermissionError, ValueError):
                        pass
                # A readable multiprocessing sentinel is the operating-system
                # proof that the process has physically exited.  is_alive()
                # may still expose a stale parent-side reap/cache view under
                # runner load and must not override that stronger proof.
                return True
            return not _process_is_alive(process)
        except (AttributeError, OSError, PermissionError, ValueError):
            pass
    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if not _process_is_alive(process):
            return True
        time.sleep(0.002)
    return not _process_is_alive(process)


def _process_is_alive(process: Any) -> bool:
    try:
        return bool(process.is_alive())
    except ValueError:
        return False


def _process_exitcode(process: Any) -> int | None:
    try:
        return process.exitcode
    except ValueError:
        return None


def _close_dead_process(process: Any) -> None:
    try:
        sentinel = getattr(process, "sentinel", None)
    except ValueError:
        return
    if sentinel is None:
        return
    close = getattr(process, "close", None)
    if close is not None:
        try:
            close()
        except (AttributeError, OSError, PermissionError, ValueError):
            pass


def _should_call_kill(process: Any) -> bool:
    return getattr(process, "sentinel", None) is not None or hasattr(
        process, "kill_attempted"
    )


class FastRenderBroker:
    """Parent-side owner of the single Fast render process and job slot."""

    def __init__(
        self, *, context=None, target=render_worker_entry, target_args: tuple = ()
    ):
        if target is render_worker_entry and target_args:
            raise ValueError("production render target does not accept test arguments")
        self._context = context
        self._target = target
        self._target_args = tuple(target_args)
        self._state_lock = threading.RLock()
        self._job_slot = threading.Lock()
        self._parent: Connection | None = None
        self._process = None
        self._slot: SharedIpcSlot | None = None
        self._fatal_error: str | None = None
        self._ready = False
        self._pose_ready = True
        self._quiesced = False
        self._shutdown_in_progress = False
        self._request_threads: set[threading.Thread] = set()
        self._active_request = False
        self._process_generation = 0
        self._request_generation = 0
        self._start_lock = asyncio.Lock()
        self._start_task: asyncio.Task | None = None

    def _mp_context(self):
        if self._context is not None:
            return self._context
        multiprocessing.freeze_support()
        return multiprocessing.get_context("spawn")

    def _reconcile_dead_shutdown_locked(self) -> None:
        self._request_threads = {
            thread for thread in self._request_threads if thread.is_alive()
        }
        if self._fatal_error != "render_broker_shutdown_failed":
            return
        if self._active_request or self._request_threads:
            return
        process = self._process
        if process is not None and _process_is_alive(process):
            return
        slot = self._slot
        if process is not None:
            _close_dead_process(process)
        if slot is not None:
            slot.close(unlink=True)
        self._parent = None
        self._process = None
        self._slot = None
        self._ready = False
        self._fatal_error = None

    @property
    def ready(self) -> bool:
        with self._state_lock:
            self._reconcile_dead_shutdown_locked()
            process = self._process
            return bool(
                self._ready
                and self._fatal_error is None
                and process is not None
                and _process_is_alive(process)
            )

    @property
    def pose_ready(self) -> bool:
        with self._state_lock:
            self._reconcile_dead_shutdown_locked()
            return bool(self._pose_ready and self.ready)

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            self._reconcile_dead_shutdown_locked()
            process = self._process
            if process is None:
                return None
            if not self._ready:
                slot = self._slot
                if _process_is_alive(process):
                    try:
                        process.terminate()
                    except (AttributeError, NotImplementedError, OSError, PermissionError):
                        pass
                    _wait_process_dead(process, 0.1)
                if not _process_is_alive(process):
                    if slot is not None:
                        slot.close(unlink=True)
                    self._parent = None
                    self._process = None
                    self._slot = None
                    return None
            if not _process_is_alive(process):
                slot = self._slot
                _close_dead_process(process)
                if slot is not None:
                    slot.close(unlink=True)
                self._parent = None
                self._process = None
                self._slot = None
                self._ready = False
                return None
            return process.pid

    @property
    def active_request_count(self) -> int:
        with self._state_lock:
            return sum(thread.is_alive() for thread in self._request_threads)

    def _start_sync(self) -> None:
        with self._state_lock:
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            if self._process is not None and _process_is_alive(self._process):
                return
            if self._process is not None:
                _close_dead_process(self._process)
                self._process = None
            if self._parent is not None:
                self._parent.close()
                self._parent = None
            context = self._mp_context()
            if self._slot is not None:
                self._slot.close(unlink=True)
                self._slot = None
            slot = SharedIpcSlot(
                context=context,
                name_prefix="vem_render",
                request_bytes=_MAX_GARMENT_BYTES,
                response_bytes=16 * 1024 * 1024,
            )
            next_process_generation = self._process_generation + 1
            slot_config = dict(slot.config)
            slot_config["expectedProcessGeneration"] = next_process_generation
            process = context.Process(
                target=run_shared_ipc_child,
                args=(self._target, slot_config, self._target_args),
                daemon=True,
            )
            try:
                process.start()
            except BaseException as exc:
                slot.close(unlink=True)
                try:
                    process.close()
                except Exception:
                    pass
                self._fatal_error = (
                    f"render_broker_start_failed: {type(exc).__name__}: {exc}"
                )
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                ) from exc
            self._parent = None
            self._process = process
            self._slot = slot
            self._ready = False
            self._process_generation = next_process_generation
            self._request_generation = 0

        def finalize_dead_child(reason: str) -> None:
            with self._state_lock:
                process, slot = self._process, self._slot
                if (
                    process is not None
                    and _process_is_alive(process)
                    and not _wait_process_dead(process, _STOP_CONFIRM_TIMEOUT_SECONDS)
                ):
                    self._ready = False
                    self._fatal_error = reason
                    return
                if process is not None:
                    _close_dead_process(process)
                if slot is not None:
                    slot.close(unlink=True)
                self._parent = None
                self._process = None
                self._slot = None
                self._ready = False
                self._fatal_error = reason

        if not wait_for_event(slot.config["responseEvent"], _START_TIMEOUT_SECONDS, process=process):
            with self._state_lock:
                self._ready = False
                self._fatal_error = "render_broker_readiness_timeout"
                if self._process is process and not _process_is_alive(process):
                    self._parent = None
                    self._process = None
                    self._slot = None
                    slot.close(unlink=True)
            raise AttemptWorkerError("render broker readiness timeout")
        try:
            kind, payload, response_process_generation, _response_request_generation = slot.recv_response()
        except (SharedIpcError, EOFError, OSError) as exc:
            self._stop_sync(graceful=False, reason="render_broker_readiness_failed")
            finalize_dead_child(
                f"render_broker_readiness_failed: {type(exc).__name__}: {exc}"
            )
            raise AttemptWorkerError(f"render broker readiness failed: {exc}") from exc
        if (
            kind != "ready"
            or not isinstance(payload, dict)
            or payload.get("pid") != process.pid
            or response_process_generation != 0
        ):
            self._stop_sync(graceful=False, reason="render_broker_readiness_invalid")
            finalize_dead_child("render_broker_readiness_invalid")
            raise AttemptWorkerError("render broker returned invalid readiness")
        with self._state_lock:
            # Older injected test targets do not advertise poseReady and are
            # treated as compatible; the production target explicitly sets
            # false when MediaPipe cannot initialize.
            self._pose_ready = payload.get("poseReady", True) is True
            self._ready = True

    async def start(self) -> None:
        with self._state_lock:
            self._reconcile_dead_shutdown_locked()
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            if self._shutdown_in_progress:
                raise AttemptWorkerError("render broker is shutting down")
            # Only an explicit owner start may reopen a broker after completed
            # shutdown. Request-side recovery calls _start_async directly and
            # must remain barred once shutdown has quiesced the broker.
            self._quiesced = False
        await self._start_async()

    async def _start_async(self) -> None:
        """Join one broker-owned start task without propagating caller cancellation."""
        async with self._start_lock:
            with self._state_lock:
                self._reconcile_dead_shutdown_locked()
                if self._fatal_error is not None:
                    raise AttemptWorkerError(
                        f"render broker is unavailable: {self._fatal_error}"
                    )
                if self._process is not None and _process_is_alive(self._process):
                    if self._ready:
                        return
                if self._quiesced:
                    raise AttemptWorkerError("render broker is shut down")
                task = self._start_task
                if task is None or task.done():
                    task = asyncio.create_task(
                        self._start_once(), name="fast-render-shared-start"
                    )
                    self._start_task = task
        await asyncio.shield(task)

    async def _start_once(self) -> None:
        """Start and prewarm exactly one process without holding a thread lock across await."""
        with self._state_lock:
            self._reconcile_dead_shutdown_locked()
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            if self._process is not None and _process_is_alive(self._process):
                return
            if self._process is not None:
                _close_dead_process(self._process)
                self._process = None
            if self._parent is not None:
                self._parent.close()
                self._parent = None
            context = self._mp_context()
            if self._slot is not None:
                self._slot.close(unlink=True)
                self._slot = None
            slot = SharedIpcSlot(
                context=context,
                name_prefix="vem_render",
                request_bytes=_MAX_GARMENT_BYTES,
                response_bytes=16 * 1024 * 1024,
            )
            next_process_generation = self._process_generation + 1
            slot_config = dict(slot.config)
            slot_config["expectedProcessGeneration"] = next_process_generation
            process = context.Process(
                target=run_shared_ipc_child,
                args=(self._target, slot_config, self._target_args),
                daemon=True,
            )
        start_task = asyncio.create_task(
            asyncio.to_thread(process.start),
            name="fast-render-process-start",
        )
        while not start_task.done():
            try:
                await asyncio.shield(start_task)
            except asyncio.CancelledError:
                continue
        try:
            start_task.result()
        except BaseException as exc:
            slot.close(unlink=True)
            try:
                process.close()
            except Exception:
                pass
            with self._state_lock:
                self._fatal_error = (
                    f"render_broker_start_failed: {type(exc).__name__}: {exc}"
                )
            raise AttemptWorkerError(
                f"render broker is unavailable: {self._fatal_error}"
            ) from exc
        with self._state_lock:
            self._parent = None
            self._process = process
            self._slot = slot
            self._ready = False
            self._process_generation = next_process_generation
            self._request_generation = 0

        deadline = asyncio.get_running_loop().time() + _START_TIMEOUT_SECONDS
        while (
            not slot.config["responseEvent"].is_set()
            and _process_is_alive(process)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        if not slot.config["responseEvent"].is_set():
            with self._state_lock:
                self._ready = False
                self._fatal_error = "render_broker_readiness_timeout"
                if self._process is process and not _process_is_alive(process):
                    self._parent = None
                    self._process = None
                    self._slot = None
                    slot.close(unlink=True)
            raise AttemptWorkerError("render broker readiness timeout")

        def finalize_dead_child(reason: str) -> None:
            with self._state_lock:
                current, current_slot = self._process, self._slot
                if (
                    current is not None
                    and _process_is_alive(current)
                    and not _wait_process_dead(current, _STOP_CONFIRM_TIMEOUT_SECONDS)
                ):
                    self._ready = False
                    self._fatal_error = reason
                    return
                if current is not None:
                    _close_dead_process(current)
                if current_slot is not None:
                    current_slot.close(unlink=True)
                self._parent = None
                self._process = None
                self._slot = None
                self._ready = False
                self._fatal_error = reason

        try:
            kind, payload, response_process_generation, _response_request_generation = slot.recv_response()
        except (SharedIpcError, EOFError, OSError) as exc:
            self._stop_sync(graceful=False, reason="render_broker_readiness_failed")
            finalize_dead_child(
                f"render_broker_readiness_failed: {type(exc).__name__}: {exc}"
            )
            raise AttemptWorkerError(f"render broker readiness failed: {exc}") from exc
        if (
            kind != "ready"
            or not isinstance(payload, dict)
            or payload.get("pid") != process.pid
            or response_process_generation != 0
        ):
            self._stop_sync(graceful=False, reason="render_broker_readiness_invalid")
            finalize_dead_child("render_broker_readiness_invalid")
            raise AttemptWorkerError("render broker returned invalid readiness")
        with self._state_lock:
            self._pose_ready = payload.get("poseReady", True) is True
            self._ready = True

    def quiesce(self) -> None:
        """Prevent recovery from spawning during application shutdown."""
        with self._state_lock:
            self._quiesced = True

    def _stop_sync(self, *, graceful: bool, reason: str) -> bool:
        """Stop and join truthfully; retain a live handle on any failed stop."""
        with self._state_lock:
            parent = self._parent
            process = self._process
            slot = self._slot
            self._ready = False
            has_active_request = any(
                thread.is_alive() for thread in self._request_threads
            ) or self._active_request
            errors: list[str] = []

            def record(action: str, exc: BaseException) -> None:
                errors.append(f"{action} {type(exc).__name__}: {exc}")

            def terminate(action: str) -> None:
                try:
                    process.terminate()
                except (OSError, PermissionError) as exc:
                    record(action, exc)

            if (
                graceful
                and not has_active_request
                and slot is not None
                and process is not None
                and _process_is_alive(process)
            ):
                shutdown_request_generation = self._request_generation + 1
                shutdown_acknowledged = False
                try:
                    slot.submit(
                        "shutdown",
                        None,
                        process_generation=self._process_generation,
                        request_generation=shutdown_request_generation,
                    )
                    if wait_for_event(
                        slot.config["responseEvent"], 0.25, process=process
                    ):
                        kind, payload, _, _ = slot.recv_response(
                            expected_process_generation=self._process_generation,
                            expected_request_generation=shutdown_request_generation,
                        )
                        shutdown_acknowledged = kind == "ok" and payload is None
                except Exception:
                    pass
                if shutdown_acknowledged and _process_is_alive(process):
                    # The child has completed its owned work and acknowledged
                    # shutdown.  Do not wait for heavyweight native runtimes to
                    # run interpreter-exit destructors: they can strand the
                    # parent-owned IPC mapping under hosted-runner load.
                    try:
                        process.kill()
                    except (AttributeError, NotImplementedError):
                        terminate("terminate-after-shutdown-ack")
                    except (OSError, PermissionError) as exc:
                        record("kill-after-shutdown-ack", exc)
                        terminate("terminate-after-shutdown-ack-error")
                if _wait_process_dead(process, _GRACEFUL_STOP_CONFIRM_TIMEOUT_SECONDS):
                    _close_dead_process(process)
                    self._parent = None
                    self._process = None
                    self._slot = None
                    self._fatal_error = None
                    if slot is not None:
                        slot.close(unlink=True)
                    return True
            if parent is not None:
                try:
                    parent.close()
                except Exception:
                    pass
            if process is None:
                self._parent = None
                self._slot = None
                if slot is not None:
                    slot.close(unlink=True)
                return True
            physical_dead = not _process_is_alive(process)
            if not physical_dead:
                if _should_call_kill(process):
                    try:
                        process.kill()
                    except (AttributeError, NotImplementedError):
                        terminate("terminate-fallback")
                    except (OSError, PermissionError) as exc:
                        record("kill", exc)
                        terminate("terminate-after-kill-error")
                else:
                    terminate("terminate-fallback")
                physical_dead = _wait_process_dead(
                    process, _STOP_CONFIRM_TIMEOUT_SECONDS
                )
            if not physical_dead and _process_is_alive(process):
                terminate("terminate")
                physical_dead = _wait_process_dead(
                    process, _STOP_CONFIRM_TIMEOUT_SECONDS
                )
            if not physical_dead and _process_is_alive(process):
                physical_dead = _wait_process_dead(process, 0.05)
            if not physical_dead:
                self._parent = parent
                self._process = process
                self._slot = slot
                detail = "; ".join(errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            _close_dead_process(process)
            self._parent = None
            self._process = None
            self._slot = None
            self._fatal_error = None
            if slot is not None:
                slot.close(unlink=True)
            return True

    async def shutdown(self) -> None:
        with self._state_lock:
            self._quiesced = True
            self._shutdown_in_progress = True
        async with self._start_lock:
            start_task = self._start_task
        if start_task is not None:
            while not start_task.done():
                try:
                    await asyncio.shield(start_task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if start_task.done():
                try:
                    start_task.result()
                except BaseException:
                    pass

        def stop_and_join_requests() -> bool:
            dead = self._stop_sync(
                graceful=True, reason="render_broker_shutdown_failed"
            )
            with self._state_lock:
                threads = list(self._request_threads)
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            with self._state_lock:
                requests_done = not any(
                    thread.is_alive() for thread in self._request_threads
                )
                if not requests_done and self._fatal_error is None:
                    self._fatal_error = (
                        "render_broker_shutdown_failed: request thread remained alive"
                    )
                process = self._process
                slot = self._slot
            if not dead and requests_done and process is not None:
                try:
                    deadline = time.monotonic() + _STOP_CONFIRM_TIMEOUT_SECONDS
                    delayed_dead = _wait_process_dead(
                        process, max(0.0, deadline - time.monotonic())
                    )
                    if delayed_dead:
                        _close_dead_process(process)
                except (OSError, PermissionError):
                    delayed_dead = False
                if delayed_dead:
                    if slot is not None:
                        slot.close(unlink=True)
                    with self._state_lock:
                        if self._process is process:
                            self._process = None
                            self._parent = None
                            self._slot = None
                        if self._fatal_error == "render_broker_shutdown_failed":
                            self._fatal_error = None
                    dead = True
            return dead and requests_done

        dead = await _run_broker_control(
            "fast-render-shutdown", stop_and_join_requests
        )
        if not dead:
            with self._state_lock:
                self._reconcile_dead_shutdown_locked()
                dead = (
                    self._fatal_error is None
                    and self._process is None
                    and not self._request_threads
                    and not self._active_request
                )
        if not dead:
            with self._state_lock:
                self._shutdown_in_progress = False
            raise AttemptWorkerError("render broker shutdown incomplete")
        with self._state_lock:
            self._shutdown_in_progress = False

    def _request_sync(self, payload: dict, deadline: float):
        shm = None
        with self._state_lock:
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            process = self._process
            slot = self._slot
            if (
                not self._ready
                or slot is None
                or process is None
                or not _process_is_alive(process)
            ):
                raise AttemptWorkerError("render broker is not ready")
            self._request_generation += 1
            request_generation = self._request_generation
            process_generation = self._process_generation
            self._active_request = True
        frame = payload["frame"]
        wire_payload = {
            key: value for key, value in payload.items() if key != "frame"
        }
        wire_payload.setdefault("garmentScale", 1.0)
        try:
            wire_payload["frameShared"], shm = _write_shared_render_frame(
                frame,
                process_generation=process_generation,
                request_generation=request_generation,
            )
            if time.monotonic() >= deadline:
                raise TimeoutError("render broker deadline exceeded during frame copy")
            slot.submit(
                "render",
                wire_payload,
                process_generation=process_generation,
                request_generation=request_generation,
            )
            while time.monotonic() < deadline:
                if slot.poll_response():
                    try:
                        kind, response, response_process_generation, response_request_generation = slot.recv_response(
                            expected_process_generation=process_generation,
                            expected_request_generation=request_generation,
                        )
                    except (EOFError, OSError) as exc:
                        raise AttemptWorkerError(
                            f"render broker connection closed: {exc}"
                        ) from exc
                    if (
                        response_process_generation != process_generation
                        or response_request_generation != request_generation
                    ):
                        raise AttemptWorkerError("render broker returned stale response")
                    if kind == "ok":
                        if not isinstance(response, bytes):
                            raise AttemptWorkerError(
                                "render broker returned corrupt response"
                            )
                        return response
                    if kind == "garment_error":
                        from vision.garment_composer import GarmentFetchError

                        raise GarmentFetchError(response)
                    if kind == "pose_error":
                        from vision.garment_composer import PoseUnavailableError

                        # Child diagnostics may contain model paths or native
                        # exception detail.  Pose absence is a normal attempt
                        # outcome, and its parent-facing contract is stable.
                        raise PoseUnavailableError("pose_unavailable")
                    raise AttemptWorkerError(response)
                if not _process_is_alive(process):
                    _close_dead_process(process)
                    raise AttemptWorkerError(
                        f"render broker exited with {_process_exitcode(process)}"
                    )
            raise TimeoutError("render broker job deadline exceeded")
        finally:
            _unlink_shared_render_frame(shm)
            with self._state_lock:
                self._active_request = False

    async def _recover(self, reason: str, *, restart: bool = True) -> None:
        """Terminate/join the failed job owner and optionally prestart recovery."""

        def recover() -> None:
            dead = self._stop_sync(graceful=False, reason=reason)
            if not dead:
                with self._state_lock:
                    process = self._process
                    slot = self._slot
                if process is not None:
                    deadline = time.monotonic() + _STOP_CONFIRM_TIMEOUT_SECONDS
                    while _process_is_alive(process) and time.monotonic() < deadline:
                        time.sleep(0.002)
                if process is None or _process_is_alive(process):
                    raise AttemptWorkerError("render broker recovery incomplete")
                _close_dead_process(process)
                if slot is not None:
                    slot.close(unlink=True)
                with self._state_lock:
                    if self._process is process:
                        self._process = None
                        self._parent = None
                        self._slot = None
                    if self._fatal_error == reason:
                        self._fatal_error = None
            with self._state_lock:
                if self._quiesced or not restart:
                    if not restart and self._fatal_error is None:
                        self._fatal_error = reason
                    return

        await _run_broker_control("fast-render-recover", recover)
        with self._state_lock:
            should_restart = bool(restart and not self._quiesced)
        if should_restart:
            await self._start_async()

    async def _restart_after_recovery(self) -> None:
        """Prestart one replacement only after the prior request was joined."""

        def restart() -> None:
            with self._state_lock:
                if self._quiesced:
                    return
            return

        await _run_broker_control("fast-render-restart", restart)
        with self._state_lock:
            if self._quiesced:
                return
        await self._start_async()

    async def _restart_after_delayed_cancel_death(self, reason: str) -> None:
        """Recover when kill is observed slightly after the control timeout."""

        def restart() -> None:
            with self._state_lock:
                process = self._process
                slot = self._slot
                if self._fatal_error != reason:
                    return
            if process is not None:
                deadline = time.monotonic() + _STOP_CONFIRM_TIMEOUT_SECONDS
                while _process_is_alive(process) and time.monotonic() < deadline:
                    time.sleep(0.002)
                if _process_is_alive(process):
                    return
                _close_dead_process(process)
            if slot is not None:
                slot.close(unlink=True)
            with self._state_lock:
                if self._process is process:
                    self._process = None
                    self._parent = None
                    self._slot = None
                if self._fatal_error == reason:
                    self._fatal_error = None
                if self._quiesced:
                    return
            return

        await _run_broker_control("fast-render-delayed-cancel-restart", restart)
        with self._state_lock:
            if self._quiesced or self._fatal_error == reason:
                return
        await self._start_async()

    async def render(self, payload: dict, *, deadline: float) -> bytes:
        """Submit exactly one non-queued job and keep pipe waits off the loop."""
        with self._state_lock:
            if self._quiesced:
                raise AttemptWorkerError("render broker is shut down")
        if not self._job_slot.acquire(blocking=False):
            raise AttemptWorkerError("render broker already has an active job")
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def request() -> None:
            try:
                outcome["value"] = self._request_sync(payload, deadline)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=request,
            name="fast-render-request",
            daemon=False,
        )
        with self._state_lock:
            self._request_threads.add(thread)
        request_released = False
        try:
            thread.start()
        except BaseException:
            with self._state_lock:
                self._request_threads.discard(thread)
            self._job_slot.release()
            raise
        try:
            while not done.is_set():
                await asyncio.sleep(0.002)
        except asyncio.CancelledError:
            try:
                await _finish_cancelled_control(
                    self._recover("render_job_cancelled", restart=True)
                )
            except BaseException:
                # _stop_sync retains the live handle and fatal state when it
                # cannot prove the prior worker dead.  Replacement must still
                # finish registry cleanup instead of escaping admission.
                pass
            while not done.is_set():
                try:
                    await asyncio.shield(asyncio.sleep(0.002))
                except asyncio.CancelledError:
                    continue
            thread.join()
            with self._state_lock:
                self._request_threads.discard(thread)
            self._job_slot.release()
            request_released = True
            try:
                await _finish_cancelled_control(
                    self._restart_after_delayed_cancel_death("render_job_cancelled")
                )
            except BaseException:
                pass
            raise
        finally:
            if done.is_set() and not request_released:
                thread.join()
                with self._state_lock:
                    self._request_threads.discard(thread)
                self._job_slot.release()
        if "error" in outcome:
            error = outcome["error"]
            # A valid typed attempt outcome proves the worker stayed alive;
            # keep its warmed model and PID for the next captured person.
            # Transport/protocol/corrupt-response failures still fall through
            # to bounded recovery below.
            from vision.garment_composer import GarmentFetchError, PoseUnavailableError

            if isinstance(error, (GarmentFetchError, PoseUnavailableError)):
                raise error
            if isinstance(error, SharedIpcError):
                await self._recover("render_ipc_corrupt", restart=False)
                raise AttemptWorkerError(f"render broker IPC corrupt: {error}") from error
            await self._recover("render_job_failed")
            raise error
        return outcome["value"]


def _validate_render_inputs(frame: np.ndarray, garment_png: bytes, template: str) -> int:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("fast frame must be a BGR image")
    height, width = frame.shape[:2]
    if (
        height <= 0
        or width <= 0
        or height > _MAX_FRAME_HEIGHT
        or width > _MAX_FRAME_WIDTH
    ):
        raise ValueError("fast frame dimensions exceed render cap")
    if frame.nbytes > _MAX_FRAME_RAW_BYTES:
        raise ValueError("fast frame bytes exceed render cap")
    if frame.dtype != np.uint8:
        raise ValueError("fast frame dtype must be uint8")
    if not frame.flags.c_contiguous:
        raise ValueError("fast frame must be C-contiguous")
    if not isinstance(garment_png, bytes) or not garment_png:
        raise ValueError("garment PNG bytes are required")
    if len(garment_png) > _MAX_GARMENT_BYTES:
        raise ValueError("garment PNG bytes exceed render cap")
    if template not in {"tshirt_short_sleeve", "tshirt_long_sleeve"}:
        raise ValueError("unsupported garment template")
    return frame.nbytes + len(garment_png)


async def render_attempt_frame(
    frame: np.ndarray,
    garment_png: bytes,
    *,
    digest: str,
    template: str,
    timeout: float,
    broker: FastRenderBroker,
    garment_scale: float = 1.0,
) -> bytes:
    """Apply one absolute deadline before any bounded frame copy or IPC."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    source_bytes = _validate_render_inputs(frame, garment_png, template)
    garment_scale = float(garment_scale)
    if not math.isfinite(garment_scale) or not 0.8 <= garment_scale <= 1.6:
        raise AttemptWorkerError("garment scale is invalid")
    conservative_seconds = (
        _CONSERVATIVE_PREPARE_FIXED_SECONDS
        + source_bytes / _CONSERVATIVE_PREPARE_BYTES_PER_SECOND
    )
    if not broker.ready:
        raise AttemptWorkerError("render broker is not ready")
    if not broker.pose_ready:
        from vision.garment_composer import PoseUnavailableError

        raise PoseUnavailableError("pose_unavailable")
    if timeout <= 0 or loop.time() + conservative_seconds >= deadline:
        raise TimeoutError("render deadline exceeded before frame copy")
    remaining = deadline - loop.time()
    if remaining <= 0:
        raise TimeoutError("render deadline exceeded before broker request")
    return await broker.render(
        {
            "frame": frame,
            "garmentPng": garment_png,
            "garmentDigest": digest,
            "template": template,
            "garmentScale": garment_scale,
        },
        deadline=time.monotonic() + remaining,
    )
