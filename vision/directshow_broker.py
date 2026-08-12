"""Attempt-independent DirectShow camera broker.

This module is intentionally importable without importing ``app``.  Windows
spawn/PyInstaller children enter here, own the native DirectShow capture, and
return bounded frame payloads to the parent camera manager.  If a native read
blocks, the parent can terminate and join this broker before admitting or
serving the next camera request.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import threading
import time
from enum import Enum, auto
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_sentinels
from multiprocessing.process import BaseProcess
from typing import Any

import numpy as np


MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_FRAME_WIDTH = 4096
MAX_FRAME_HEIGHT = 4096
DEFAULT_READ_TIMEOUT_SECONDS = 10.0
STOP_CONFIRM_TIMEOUT_SECONDS = 2.0
GRACEFUL_STOP_CONFIRM_TIMEOUT_SECONDS = 2.0


class DirectShowCameraUnavailable(RuntimeError):
    """The physical DirectShow owner could not be stopped truthfully."""


class _ProcessLiveness(Enum):
    ALIVE = auto()
    DEAD = auto()
    UNKNOWN = auto()


class _SentinelWaitResult(Enum):
    READY = auto()
    TIMED_OUT = auto()
    UNAVAILABLE = auto()


def _uncancel_current_task() -> None:
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        task.uncancel()


async def _await_task_uncancelled(task: asyncio.Task) -> tuple[Any, bool]:
    """Finish a cleanup task even if its caller is repeatedly cancelled."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            _uncancel_current_task()
    return task.result(), cancelled


def _is_process_alive(process: Any) -> _ProcessLiveness:
    try:
        if getattr(process, "exitcode", None) is not None:
            return _ProcessLiveness.DEAD
    except (OSError, PermissionError, ValueError):
        return _ProcessLiveness.UNKNOWN
    try:
        return (
            _ProcessLiveness.ALIVE
            if process.is_alive()
            else _ProcessLiveness.DEAD
        )
    except (OSError, PermissionError, ValueError):
        return _ProcessLiveness.UNKNOWN


def _wait_process_dead(process: Any, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    try:
        sentinel = getattr(process, "sentinel", None)
    except (OSError, PermissionError, ValueError):
        sentinel = None
    if sentinel is not None:
        try:
            if wait_for_sentinels(
                [sentinel], timeout=max(deadline - time.monotonic(), 0.0)
            ):
                join = getattr(process, "join", None)
                if join is not None and isinstance(process, BaseProcess):
                    try:
                        join(timeout=0)
                    except (AssertionError, OSError, PermissionError, ValueError):
                        pass
                if _is_process_alive(process) is _ProcessLiveness.DEAD:
                    return True
            elif _is_process_alive(process) is _ProcessLiveness.DEAD:
                return True
        except (OSError, PermissionError, ValueError):
            pass
    while time.monotonic() < deadline:
        if _is_process_alive(process) is _ProcessLiveness.DEAD:
            return True
        time.sleep(0.002)
    return _is_process_alive(process) is _ProcessLiveness.DEAD


def _wait_process_sentinel(
    process: Any, sentinel: Any, deadline: float
) -> _SentinelWaitResult:
    """Wait in a bounded worker thread for the OS process-death sentinel."""
    try:
        ready = wait_for_sentinels(
            [sentinel], timeout=max(deadline - time.monotonic(), 0.0)
        )
    except (OSError, PermissionError, TypeError, ValueError):
        return _SentinelWaitResult.UNAVAILABLE
    if not ready:
        return _SentinelWaitResult.TIMED_OUT
    if isinstance(process, BaseProcess):
        try:
            process.join(timeout=0)
        except (AssertionError, OSError, PermissionError, ValueError):
            pass
    return _SentinelWaitResult.READY


def _close_dead_process(process: Any) -> None:
    try:
        sentinel = getattr(process, "sentinel", None)
    except (OSError, PermissionError, ValueError):
        return
    if sentinel is None:
        return
    close = getattr(process, "close", None)
    if close is not None:
        try:
            close()
        except (OSError, PermissionError, ValueError):
            pass


def _should_call_kill(process: Any) -> bool:
    try:
        has_sentinel = getattr(process, "sentinel", None) is not None
    except (OSError, PermissionError, ValueError):
        has_sentinel = False
    return has_sentinel or hasattr(process, "kill_attempted")


def _keep_open(config: dict) -> bool:
    value = config.get("keep_open", False)
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _validate_frame(image: Any) -> None:
    if not isinstance(image, np.ndarray):
        raise RuntimeError("directshow broker returned a non-array frame")
    if image.ndim not in (2, 3):
        raise RuntimeError("directshow broker returned an invalid frame rank")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise RuntimeError("directshow broker returned an empty frame")
    if height > MAX_FRAME_HEIGHT or width > MAX_FRAME_WIDTH:
        raise RuntimeError("directshow broker frame dimensions exceed IPC cap")
    if image.nbytes > MAX_FRAME_BYTES:
        raise RuntimeError("directshow broker frame bytes exceed IPC cap")


def _open_bound_capture(config: dict):
    from vision.camera import open_camera
    from vision.camera_binding import acquire_runtime_camera_lease
    from vision.config import settings

    candidate_id = config.get("stableId")
    role = config.get("role")
    if not isinstance(candidate_id, str) or not isinstance(role, str):
        raise RuntimeError("camera runtime requires a resolved stable role binding")
    lease = acquire_runtime_camera_lease(candidate_id, role)
    try:
        capture = open_camera(
            camera_index=int(config["index"]),
            backend_name=config.get("backend", settings.CAMERA_BACKEND),
            width=int(config.get("width", settings.CAMERA_WIDTH) or 0),
            height=int(config.get("height", settings.CAMERA_HEIGHT) or 0),
            fps=int(config.get("fps", settings.CAMERA_FPS) or 0),
            fourcc=config.get("fourcc", settings.CAMERA_FOURCC),
        )
        return capture, lease
    except Exception:
        lease.release()
        raise


def _read_transformed_frame(capture, config: dict, warmup_frames: int | None):
    from vision.camera import read_warmup_frame
    from vision.config import settings
    from vision.frame_transform import camera_rotation, rotate_frame

    raw = read_warmup_frame(
        capture,
        settings.CAMERA_WARMUP_FRAMES if warmup_frames is None else warmup_frames,
    )
    image = rotate_frame(raw, camera_rotation(config))
    _validate_frame(image)
    return image


def _capture_status(capture, config: dict, image):
    from vision.camera import describe_capture
    from vision.frame_transform import camera_rotation

    height, width = image.shape[:2]
    return {
        "ok": True,
        "role": config.get("logicalRole"),
        "backend": config.get("backend"),
        "actual": describe_capture(capture),
        "transform": {
            "rotate": camera_rotation(config),
            "rawWidth": width,
            "rawHeight": height,
        },
        "frame": {
            "width": width,
            "height": height,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
        },
        "broker": {
            "pid": os.getpid(),
        },
    }


def directshow_broker_entry(connection: Connection, config: dict) -> None:
    capture = None
    lease = None

    def close_capture() -> None:
        nonlocal capture, lease
        if capture is not None:
            try:
                capture.release()
            finally:
                capture = None
        if lease is not None:
            lease.release()
            lease = None

    try:
        if config.get("_brokerReadyHandshake", False):
            # Recovery starts this child before releasing the canceled attempt's
            # cleanup barrier.  This signal proves the spawned runtime reached
            # its command loop; the next attempt's frame budget never pays for
            # import and process bootstrap.
            connection.send(("ready", {"pid": os.getpid()}))
        while True:
            try:
                command, payload = connection.recv()
            except EOFError:
                return

            if command == "shutdown":
                close_capture()
                connection.send(("ok", None))
                return

            try:
                if capture is None or not capture.isOpened():
                    close_capture()
                    capture, lease = _open_bound_capture(config)

                if command == "read":
                    image = _read_transformed_frame(capture, config, payload)
                    if not _keep_open(config):
                        close_capture()
                    connection.send(("ok", {"pid": os.getpid(), "image": image}))
                    continue

                if command == "status":
                    image = _read_transformed_frame(capture, config, payload)
                    status = _capture_status(capture, config, image)
                    if not _keep_open(config):
                        close_capture()
                    connection.send(("ok", status))
                    continue

                raise RuntimeError(f"unknown directshow broker command: {command}")
            except BaseException as exc:
                close_capture()
                connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        close_capture()
        connection.close()


class DirectShowCameraBroker:
    """Parent-side owner for one role's DirectShow broker process."""

    def __init__(
        self,
        role: str,
        config: dict,
        *,
        context=None,
        target=directshow_broker_entry,
        stop_timeout_seconds: float = STOP_CONFIRM_TIMEOUT_SECONDS,
    ):
        self.role = role
        self.config = dict(config)
        self.config["logicalRole"] = role
        self._context = context
        self._target = target
        # Native pipe waits never hold this state lock.  Abort/maintenance can
        # therefore close and kill the child from an independent control path.
        self._state_lock = threading.RLock()
        self._request_slot = threading.Lock()
        self._parent: Connection | None = None
        self._child: Connection | None = None
        self._process = None
        self._request_threads: set[threading.Thread] = set()
        self._request_abort = threading.Event()
        self._ready = False
        self._fatal_error: str | None = None
        self._quiesced = False
        self.opened_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_count = 0
        self.restart_count = 0
        self.last_pid: int | None = None
        self._stop_timeout_seconds = max(float(stop_timeout_seconds), 0.001)
        self._async_stop_lock = asyncio.Lock()

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process is None:
                return None
            if _is_process_alive(process) is _ProcessLiveness.DEAD:
                _close_dead_process(process)
                self._parent = None
                self._process = None
                self.opened_at = None
                return None
            return process.pid

    @property
    def active_request_count(self) -> int:
        with self._state_lock:
            return sum(thread.is_alive() for thread in self._request_threads)

    def _mp_context(self):
        if self._context is not None:
            return self._context
        multiprocessing.freeze_support()
        return multiprocessing.get_context("spawn")

    def _start_locked(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(f"directshow broker is unavailable: {self._fatal_error}")
        if self._quiesced:
            raise RuntimeError("directshow broker is quiesced for maintenance")
        if self._process is not None:
            liveness = _is_process_alive(self._process)
            if liveness is _ProcessLiveness.ALIVE:
                return
            if liveness is _ProcessLiveness.UNKNOWN:
                self._fatal_error = "existing broker process liveness is unknown"
                raise RuntimeError(
                    f"directshow broker is unavailable: {self._fatal_error}"
                )
        if self._process is not None:
            _close_dead_process(self._process)
            self._process = None
        if self._parent is not None:
            self._parent.close()
            self._parent = None
        context = self._mp_context()
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=self._target,
            args=(child, self.config),
            daemon=True,
        )
        process.start()
        child.close()
        self._parent = parent
        self._child = None
        self._process = process
        self._ready = False
        self.opened_at = time.time()
        self.restart_count += 1

    def _wait_ready(self, parent: Connection, process: Any, deadline: float) -> None:
        if not self.config.get("_brokerReadyHandshake", False) or self._ready:
            return
        while time.monotonic() < deadline:
            if self._request_abort.is_set():
                raise RuntimeError("directshow broker startup was aborted")
            if parent.poll(0.005):
                kind, response = parent.recv()
                if kind == "ready":
                    if not isinstance(response, dict) or not isinstance(response.get("pid"), int):
                        raise RuntimeError("directshow broker returned invalid startup readiness")
                    self._ready = True
                    return
                raise RuntimeError(f"directshow broker startup failed: {response}")
            if _is_process_alive(process) is _ProcessLiveness.DEAD:
                _close_dead_process(process)
                raise RuntimeError(
                    f"directshow broker exited during startup with {process.exitcode}"
                )
        raise TimeoutError("directshow broker startup readiness deadline exceeded")

    async def start_async(self, *, timeout: float = STOP_CONFIRM_TIMEOUT_SECONDS) -> None:
        """Start and prove a replacement child is command-loop ready.

        This is intentionally a recovery boundary, not a read timeout: callers
        invoke it only after the canceled request thread has joined.
        """
        if not self._request_slot.acquire(blocking=False):
            raise RuntimeError("directshow broker already has an active request")
        self._request_abort.clear()
        try:
            with self._state_lock:
                self._start_locked()
                assert self._parent is not None
                parent = self._parent
                process = self._process
            await asyncio.to_thread(
                self._wait_ready,
                parent,
                process,
                time.monotonic() + max(float(timeout), 0.001),
            )
        except BaseException:
            await self.abort_async(reason="startup_readiness_failed")
            raise
        finally:
            self._request_slot.release()

    def _stop_process(self, *, graceful: bool, reason: str) -> bool:
        """Stop and join the child without ever faking death or losing its handle."""
        self._request_abort.set()
        with self._state_lock:
            parent = self._parent
            process = self._process
            stop_errors: list[str] = []

            def record_stop_error(action: str, exc: BaseException) -> None:
                stop_errors.append(f"{action} {type(exc).__name__}: {exc}")

            def terminate_process(action: str) -> None:
                try:
                    process.terminate()
                except (
                    AttributeError,
                    NotImplementedError,
                    OSError,
                    PermissionError,
                    ValueError,
                ) as exc:
                    record_stop_error(action, exc)

            if (
                graceful
                and parent is not None
                and process is not None
                and _is_process_alive(process) is _ProcessLiveness.ALIVE
            ):
                try:
                    parent.send(("shutdown", None))
                    parent.poll(0.25)
                except Exception:
                    pass
                if _wait_process_dead(process, GRACEFUL_STOP_CONFIRM_TIMEOUT_SECONDS):
                    _close_dead_process(process)
                    self._parent = None
                    self._process = None
                    self.opened_at = None
                    self._fatal_error = None
                    return True
            if parent is not None:
                try:
                    parent.close()
                except Exception:
                    pass
            if process is None:
                self._parent = None
                self.opened_at = None
                return True
            if _is_process_alive(process) is not _ProcessLiveness.DEAD:
                if _should_call_kill(process):
                    try:
                        process.kill()
                    except (AttributeError, NotImplementedError):
                        terminate_process("terminate-fallback")
                    except (OSError, PermissionError, ValueError) as exc:
                        record_stop_error("kill", exc)
                        terminate_process("terminate-after-kill-error")
                else:
                    terminate_process("terminate-fallback")
            if _is_process_alive(process) is not _ProcessLiveness.DEAD:
                _wait_process_dead(process, self._stop_timeout_seconds)
            if _is_process_alive(process) is not _ProcessLiveness.DEAD:
                # Fail closed.  The live process handle is the only truthful
                # proof that the physical-camera owner may still exist.
                self._parent = parent
                self._process = process
                self.opened_at = None
                detail = "; ".join(stop_errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            _close_dead_process(process)
            self._parent = None
            self._process = None
            self.opened_at = None
            self._fatal_error = None
            return True

    def _request(
        self,
        command: str,
        payload=None,
        *,
        timeout: float | None = None,
        _slot_owned: bool = False,
        _abort_on_error: bool = True,
    ):
        if not _slot_owned:
            if not self._request_slot.acquire(blocking=False):
                raise RuntimeError("directshow broker already has an active request")
            self._request_abort.clear()
        try:
            deadline = time.monotonic() + (
                DEFAULT_READ_TIMEOUT_SECONDS
                if timeout is None
                else max(float(timeout), 0.001)
            )
            with self._state_lock:
                self._start_locked()
                assert self._parent is not None
                parent = self._parent
                process = self._process
            self._wait_ready(parent, process, deadline)
            with self._state_lock:
                if self._parent is not parent or self._process is not process:
                    raise RuntimeError("directshow broker changed during startup")
                self._parent.send((command, payload))
            while time.monotonic() < deadline:
                if self._request_abort.is_set():
                    raise RuntimeError("directshow broker request was aborted")
                if parent.poll(0.005):
                    kind, response = parent.recv()
                    if kind == "ok":
                        return response
                    raise RuntimeError(response)
                if (
                    process is not None
                    and _is_process_alive(process) is _ProcessLiveness.DEAD
                ):
                    _close_dead_process(process)
                    raise RuntimeError(
                        f"directshow broker exited with {process.exitcode}"
                    )
            raise TimeoutError("directshow broker read deadline exceeded")
        except Exception as exc:
            if not _abort_on_error:
                raise
            stopped = self._stop_process(
                graceful=False,
                reason="request_abort_failed",
            )
            if not stopped:
                with self._state_lock:
                    fatal_error = self._fatal_error or "request_abort_failed"
                raise DirectShowCameraUnavailable(
                    f"directshow broker is unavailable: {fatal_error}"
                ) from exc
            raise
        finally:
            if not _slot_owned:
                self._request_slot.release()

    async def _request_async(
        self, command: str, payload=None, *, timeout: float | None = None
    ):
        """Run exactly one request on a dedicated, non-queued worker thread."""
        if not self._request_slot.acquire(blocking=False):
            raise RuntimeError("directshow broker already has an active request")
        self._request_abort.clear()
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def run_request() -> None:
            try:
                outcome["value"] = self._request(
                    command,
                    payload,
                    timeout=timeout,
                    _slot_owned=True,
                    _abort_on_error=False,
                )
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=run_request,
            name=f"directshow-{self.role}-request",
            daemon=False,
        )
        with self._state_lock:
            self._request_threads.add(thread)
        try:
            thread.start()
        except BaseException:
            with self._state_lock:
                self._request_threads.discard(thread)
            self._request_slot.release()
            raise
        cancelled = False
        while not done.is_set():
            try:
                await asyncio.sleep(0.002)
            except asyncio.CancelledError:
                cancelled = True
                _uncancel_current_task()
                break

        error = outcome.get("error")
        stopped = True
        cleanup_error: BaseException | None = None
        if cancelled or error is not None:
            cleanup = asyncio.create_task(
                self.abort_async(
                    reason=(
                        "request_cancelled" if cancelled else "request_abort_failed"
                    )
                ),
                name=f"directshow-{self.role}-request-cleanup",
            )
            try:
                stopped, cleanup_cancelled = await _await_task_uncancelled(cleanup)
                cancelled = cancelled or cleanup_cancelled
            except BaseException as exc:
                cleanup_error = exc

        while not done.is_set():
            try:
                await asyncio.sleep(0.002)
            except asyncio.CancelledError:
                cancelled = True
                _uncancel_current_task()

        thread.join()
        with self._state_lock:
            self._request_threads.discard(thread)
        self._request_slot.release()

        if cancelled:
            raise asyncio.CancelledError
        if cleanup_error is not None:
            raise cleanup_error
        if error is not None:
            if not stopped:
                with self._state_lock:
                    fatal_error = self._fatal_error or "request_abort_failed"
                raise DirectShowCameraUnavailable(
                    f"directshow broker is unavailable: {fatal_error}"
                ) from error
            raise error
        return outcome["value"]

    async def abort_async(self, *, reason: str = "request_aborted") -> bool:
        """Confirm physical death and request-thread join before admitting a restart."""
        async with self._async_stop_lock:
            deadline = time.monotonic() + self._stop_timeout_seconds
            with self._state_lock:
                parent = self._parent
                process = self._process
                threads = list(self._request_threads)
            self._request_abort.set()
            if parent is not None:
                try:
                    parent.close()
                except Exception:
                    pass
            errors: list[str] = []

            def signal_stop() -> None:
                if (
                    process is None
                    or _is_process_alive(process) is _ProcessLiveness.DEAD
                ):
                    return
                if _should_call_kill(process):
                    try:
                        process.kill()
                        return
                    except (AttributeError, NotImplementedError):
                        pass
                    except (OSError, PermissionError, ValueError) as exc:
                        errors.append(f"kill {type(exc).__name__}: {exc}")
                try:
                    process.terminate()
                except (
                    AttributeError,
                    NotImplementedError,
                    OSError,
                    PermissionError,
                    ValueError,
                ) as exc:
                    errors.append(f"terminate {type(exc).__name__}: {exc}")

            control = asyncio.create_task(
                asyncio.to_thread(signal_stop),
                name=f"directshow-{self.role}-stop-signal",
            )
            while not control.done():
                try:
                    await asyncio.shield(control)
                except asyncio.CancelledError:
                    _uncancel_current_task()
                    continue
            try:
                control.result()
            except BaseException as exc:
                errors.append(f"control {type(exc).__name__}: {exc}")

            sentinel = None
            sentinel_available = False
            sentinel_result = _SentinelWaitResult.UNAVAILABLE
            if process is not None:
                try:
                    sentinel = getattr(process, "sentinel", None)
                    sentinel_available = sentinel is not None
                except (OSError, PermissionError, ValueError):
                    pass
            if sentinel_available:
                sentinel_wait = asyncio.create_task(
                    asyncio.to_thread(
                        _wait_process_sentinel, process, sentinel, deadline
                    ),
                    name=f"directshow-{self.role}-sentinel-wait",
                )
                while not sentinel_wait.done():
                    try:
                        await asyncio.shield(sentinel_wait)
                    except asyncio.CancelledError:
                        _uncancel_current_task()
                        continue
                try:
                    sentinel_result = sentinel_wait.result()
                except BaseException as exc:
                    errors.append(f"sentinel {type(exc).__name__}: {exc}")
                    sentinel_result = _SentinelWaitResult.UNAVAILABLE
                if sentinel_result is _SentinelWaitResult.UNAVAILABLE:
                    sentinel_available = False
            if (
                not sentinel_available
                or sentinel_result is _SentinelWaitResult.READY
            ):
                while (
                    process is not None
                    and _is_process_alive(process) is not _ProcessLiveness.DEAD
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
            while (
                any(thread.is_alive() for thread in threads)
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.01)
            for thread in threads:
                if not thread.is_alive() and thread is not threading.current_thread():
                    thread.join(timeout=0)

            process_alive = (
                process is not None
                and _is_process_alive(process) is not _ProcessLiveness.DEAD
            )
            threads_alive = any(thread.is_alive() for thread in threads)
            if process_alive or threads_alive:
                detail = "; ".join(errors)
                suffix = (
                    "request thread remained alive"
                    if threads_alive
                    else "process remained alive"
                )
                with self._state_lock:
                    self._fatal_error = (
                        f"{reason}: {detail}; {suffix}"
                        if detail
                        else f"{reason}: {suffix}"
                    )
                return False

            if process is not None:
                _close_dead_process(process)
            with self._state_lock:
                if self._process is process:
                    self._parent = None
                    self._process = None
                    self.opened_at = None
                self._request_threads = {
                    thread for thread in self._request_threads if thread.is_alive()
                }
                self._fatal_error = None
            return True

    def abort(self, *, reason: str = "request_aborted") -> bool:
        dead = self._stop_process(graceful=False, reason=reason)
        with self._state_lock:
            threads = [
                thread for thread in self._request_threads
                if thread is not threading.current_thread()
            ]
        for thread in threads:
            thread.join(timeout=1.0)
        with self._state_lock:
            requests_done = not any(thread.is_alive() for thread in self._request_threads)
            if not requests_done:
                self._fatal_error = f"{reason}: request thread remained alive"
        return dead and requests_done

    def read(self, warmup_frames: int | None = None, *, timeout: float | None = None):
        response = self._request("read", warmup_frames, timeout=timeout)
        image = response["image"]
        _validate_frame(image)
        self.last_pid = int(response["pid"])
        self.last_frame_at = time.time()
        self.frame_count += 1
        return image

    async def read_async(
        self, warmup_frames: int | None = None, *, timeout: float | None = None
    ):
        response = await self._request_async("read", warmup_frames, timeout=timeout)
        image = response["image"]
        _validate_frame(image)
        self.last_pid = int(response["pid"])
        self.last_frame_at = time.time()
        self.frame_count += 1
        return image

    def status(self, warmup_frames: int | None = None, *, timeout: float | None = None):
        status = self._request("status", warmup_frames, timeout=timeout)
        status.setdefault("mode", "broker")
        status["broker"].update(
            {
                "openedAt": self.opened_at,
                "frameCount": self.frame_count,
                "restartCount": self.restart_count,
            }
        )
        return status

    def release(self) -> bool:
        dead = self._stop_process(graceful=True, reason="broker_release_failed")
        with self._state_lock:
            threads = list(self._request_threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        with self._state_lock:
            requests_done = not any(thread.is_alive() for thread in self._request_threads)
            if not requests_done:
                self._fatal_error = "broker_release_failed: request thread remained alive"
        return dead and requests_done

    def reset(self) -> bool:
        return self.release()

    def assert_dead(self) -> bool:
        with self._state_lock:
            process = self._process
            return (
                process is None
                or _is_process_alive(process) is _ProcessLiveness.DEAD
            )

    def begin_maintenance(self) -> None:
        with self._state_lock:
            self._quiesced = True
        if not self.release():
            with self._state_lock:
                if self._fatal_error is None:
                    self._fatal_error = "maintenance_quiesce_failed"
            raise RuntimeError("directshow broker could not be stopped for maintenance")

    def end_maintenance(self) -> None:
        with self._state_lock:
            self._quiesced = False
