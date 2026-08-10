"""Attempt-independent DirectShow camera broker.

This module is intentionally importable without importing ``app``.  Windows
spawn/PyInstaller children enter here, own the native DirectShow capture, and
return bounded frame payloads to the parent camera manager.  If a native read
blocks, the parent can terminate and join this broker before admitting or
serving the next camera request.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

import numpy as np


MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_FRAME_WIDTH = 4096
MAX_FRAME_HEIGHT = 4096
DEFAULT_READ_TIMEOUT_SECONDS = 10.0


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
        self._lock = threading.RLock()
        self._parent: Connection | None = None
        self._child: Connection | None = None
        self._process = None
        self.opened_at: float | None = None
        self.last_frame_at: float | None = None
        self.frame_count = 0
        self.restart_count = 0
        self.last_pid: int | None = None

    @property
    def pid(self) -> int | None:
        process = self._process
        if process is None or not process.is_alive():
            return None
        return process.pid

    def _mp_context(self):
        if self._context is not None:
            return self._context
        multiprocessing.freeze_support()
        return multiprocessing.get_context("spawn")

    def _start_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._close_locked(kill=True)
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

    def _close_locked(self, *, kill: bool = False) -> None:
        parent = self._parent
        process = self._process
        if parent is not None:
            try:
                parent.close()
            except Exception:
                pass
        if process is not None and process.is_alive():
            if kill:
                try:
                    process.kill()
                except AttributeError:
                    process.terminate()
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        self._parent = None
        self._process = None
        self.opened_at = None

    def _request(self, command: str, payload=None, *, timeout: float | None = None):
        deadline = time.monotonic() + (
            DEFAULT_READ_TIMEOUT_SECONDS if timeout is None else max(float(timeout), 0.001)
        )
        with self._lock:
            self._start_locked()
            assert self._parent is not None
            process = self._process
            try:
                self._parent.send((command, payload))
                while time.monotonic() < deadline:
                    if self._parent.poll(0.005):
                        kind, response = self._parent.recv()
                        if kind == "ok":
                            return response
                        raise RuntimeError(response)
                    if process is not None and not process.is_alive():
                        raise RuntimeError(
                            f"directshow broker exited with {process.exitcode}"
                        )
                raise TimeoutError("directshow broker read deadline exceeded")
            except Exception:
                self._close_locked(kill=True)
                raise

    def read(self, warmup_frames: int | None = None, *, timeout: float | None = None):
        response = self._request("read", warmup_frames, timeout=timeout)
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

    def release(self) -> None:
        with self._lock:
            parent = self._parent
            if parent is not None and self._process is not None and self._process.is_alive():
                try:
                    parent.send(("shutdown", None))
                    parent.poll(0.25)
                except Exception:
                    pass
            self._close_locked(kill=True)

    def reset(self) -> None:
        self.release()

    def assert_dead(self) -> bool:
        process = self._process
        return process is None or not process.is_alive()
