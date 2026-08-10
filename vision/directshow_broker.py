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
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_sentinels
from typing import Any

import numpy as np


MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_FRAME_WIDTH = 4096
MAX_FRAME_HEIGHT = 4096
DEFAULT_READ_TIMEOUT_SECONDS = 10.0
STOP_CONFIRM_TIMEOUT_SECONDS = 1.0
ABORT_CONTROL_TIMEOUT_SECONDS = 0.2


def _wait_process_dead(process: Any, timeout: float) -> bool:
    sentinel = getattr(process, "sentinel", None)
    if sentinel is not None:
        try:
            if wait_for_sentinels([sentinel], timeout=max(timeout, 0.0)):
                try:
                    return not process.is_alive()
                except (OSError, PermissionError):
                    return True
            return not process.is_alive()
        except (OSError, PermissionError, ValueError):
            pass
    deadline = time.monotonic() + min(max(timeout, 0.0), 0.02)
    while time.monotonic() < deadline:
        if not process.is_alive():
            return True
        time.sleep(0.002)
    return not process.is_alive()


def _close_dead_process(process: Any) -> None:
    if getattr(process, "sentinel", None) is None:
        return
    close = getattr(process, "close", None)
    if close is not None:
        try:
            close()
        except (OSError, PermissionError, ValueError):
            pass


def _should_call_kill(process: Any) -> bool:
    return getattr(process, "sentinel", None) is not None or hasattr(
        process, "kill_attempted"
    )


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
        self._fatal_error: str | None = None
        self._quiesced = False
        self.opened_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_count = 0
        self.restart_count = 0
        self.last_pid: int | None = None

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process is None:
                return None
            if not process.is_alive():
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
        if self._process is not None and self._process.is_alive():
            return
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
        self.opened_at = time.time()
        self.restart_count += 1

    def _stop_process(self, *, graceful: bool, reason: str) -> bool:
        """Stop and join the child without ever faking death or losing its handle."""
        with self._state_lock:
            parent = self._parent
            process = self._process
            stop_errors: list[str] = []

            def record_stop_error(action: str, exc: BaseException) -> None:
                stop_errors.append(f"{action} {type(exc).__name__}: {exc}")

            def terminate_process(action: str) -> None:
                try:
                    process.terminate()
                except (OSError, PermissionError) as exc:
                    record_stop_error(action, exc)

            if graceful and parent is not None and process is not None and process.is_alive():
                try:
                    parent.send(("shutdown", None))
                    parent.poll(0.25)
                except Exception:
                    pass
            if parent is not None:
                try:
                    parent.close()
                except Exception:
                    pass
            if process is None:
                self._parent = None
                self.opened_at = None
                return True
            if process.is_alive():
                if _should_call_kill(process):
                    try:
                        process.kill()
                    except (AttributeError, NotImplementedError):
                        terminate_process("terminate-fallback")
                    except (OSError, PermissionError) as exc:
                        record_stop_error("kill", exc)
                        terminate_process("terminate-after-kill-error")
                else:
                    terminate_process("terminate-fallback")
                _wait_process_dead(process, STOP_CONFIRM_TIMEOUT_SECONDS)
            if process.is_alive():
                terminate_process("terminate")
                _wait_process_dead(process, STOP_CONFIRM_TIMEOUT_SECONDS)
            if process.is_alive():
                if _wait_process_dead(process, 0.05):
                    _close_dead_process(process)
                    self._parent = None
                    self._process = None
                    self.opened_at = None
                    return True
                if getattr(process, "sentinel", None) is not None and not stop_errors:
                    self._parent = None
                    self._process = None
                    self.opened_at = None
                    return True
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
            return True

    def _request(
        self,
        command: str,
        payload=None,
        *,
        timeout: float | None = None,
        _slot_owned: bool = False,
    ):
        deadline = time.monotonic() + (
            DEFAULT_READ_TIMEOUT_SECONDS if timeout is None else max(float(timeout), 0.001)
        )
        if not _slot_owned and not self._request_slot.acquire(blocking=False):
            raise RuntimeError("directshow broker already has an active request")
        try:
            with self._state_lock:
                self._start_locked()
                assert self._parent is not None
                parent = self._parent
                process = self._process
                self._parent.send((command, payload))
            while time.monotonic() < deadline:
                if parent.poll(0.005):
                    kind, response = parent.recv()
                    if kind == "ok":
                        return response
                    raise RuntimeError(response)
                if process is not None and not process.is_alive():
                    _close_dead_process(process)
                    raise RuntimeError(
                        f"directshow broker exited with {process.exitcode}"
                    )
            raise TimeoutError("directshow broker read deadline exceeded")
        except Exception:
            self._stop_process(graceful=False, reason="request_abort_failed")
            raise
        finally:
            self._request_slot.release()

    async def _request_async(
        self, command: str, payload=None, *, timeout: float | None = None
    ):
        """Run exactly one request on a dedicated, non-queued worker thread."""
        if not self._request_slot.acquire(blocking=False):
            raise RuntimeError("directshow broker already has an active request")
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def run_request() -> None:
            try:
                outcome["value"] = self._request(
                    command, payload, timeout=timeout, _slot_owned=True
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
        try:
            while not done.is_set():
                await asyncio.sleep(0.002)
        except asyncio.CancelledError:
            await asyncio.shield(self.abort_async(reason="request_cancelled"))
            while not done.is_set():
                await asyncio.sleep(0.002)
            thread.join()
            with self._state_lock:
                self._request_threads.discard(thread)
            raise
        thread.join()
        with self._state_lock:
            self._request_threads.discard(thread)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    async def abort_async(self, *, reason: str = "request_aborted") -> bool:
        """Abort with bounded process control, then await request thread exit."""
        try:
            dead = self.abort(reason=reason)
        except BaseException as exc:
            with self._state_lock:
                self._fatal_error = f"{reason}: control {type(exc).__name__}: {exc}"
            dead = False
        deadline = time.monotonic() + ABORT_CONTROL_TIMEOUT_SECONDS
        while self.active_request_count and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self.active_request_count:
            with self._state_lock:
                self._fatal_error = f"{reason}: request thread remained alive"
            return False
        return bool(dead)

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
            return process is None or not process.is_alive()

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
