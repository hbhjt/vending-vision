"""Killable one-slot process boundary for V2 acquisition observation.

The child imports only Vision model modules (never ``app``), prewarms YOLO and
MediaPipe once, and owns synchronous JPEG/inference.  The parent keeps exactly
one request thread; cancellation kills and joins the child before the registry
may admit another front-camera owner.  A child that cannot be verified dead is
retained and the boundary fails closed.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1080
MAX_FRAME_EDGE = 1920
MAX_FRAME_RAW_BYTES = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3


@dataclass(frozen=True)
class AcquisitionObservation:
    jpeg: bytes
    occupancy: str
    aligned: bool


@dataclass(frozen=True)
class _SharedFrame:
    name: str
    shape: tuple[int, int, int]
    dtype: str
    nbytes: int
    generation: int


def _coerce_frame_for_shared_memory(frame: Any) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("acquisition frame must be a BGR image")
    height, width, channels = frame.shape
    if (
        height <= 0
        or width <= 0
        or height > MAX_FRAME_EDGE
        or width > MAX_FRAME_EDGE
        or channels != 3
        or frame.dtype != np.uint8
        or frame.nbytes > MAX_FRAME_RAW_BYTES
    ):
        raise ValueError("acquisition frame metadata exceeds cap")
    if frame.flags.c_contiguous:
        return frame
    return np.ascontiguousarray(frame)


def _write_shared_frame(frame: Any, *, generation: int) -> tuple[_SharedFrame, shared_memory.SharedMemory]:
    contiguous = _coerce_frame_for_shared_memory(frame)
    shm = shared_memory.SharedMemory(
        create=True,
        size=int(contiguous.nbytes),
        name=f"vem_acq_{uuid4().hex}",
    )
    try:
        shm.buf[: contiguous.nbytes] = contiguous.reshape(-1).view(np.uint8)
    except BaseException:
        try:
            shm.close()
        finally:
            shm.unlink()
        raise
    return (
        _SharedFrame(
            name=shm.name,
            shape=tuple(int(value) for value in contiguous.shape),
            dtype=str(contiguous.dtype),
            nbytes=int(contiguous.nbytes),
            generation=int(generation),
        ),
        shm,
    )


def _read_shared_frame(metadata: Any, *, generation: int) -> np.ndarray:
    if not isinstance(metadata, dict) or metadata.get("kind") != "shared_frame":
        raise ValueError("invalid acquisition frame metadata")
    shape = metadata.get("shape")
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 3
        or any(type(value) is not int for value in shape)
    ):
        raise ValueError("invalid acquisition frame shape")
    height, width, channels = tuple(int(value) for value in shape)
    nbytes = metadata.get("nbytes")
    dtype = metadata.get("dtype")
    if (
        metadata.get("generation") != generation
        or dtype != "uint8"
        or channels != 3
        or height <= 0
        or width <= 0
        or height > MAX_FRAME_EDGE
        or width > MAX_FRAME_EDGE
        or type(nbytes) is not int
        or nbytes != height * width * channels
        or nbytes > MAX_FRAME_RAW_BYTES
    ):
        raise ValueError("acquisition frame metadata exceeds cap")
    shm = shared_memory.SharedMemory(name=str(metadata.get("name")))
    try:
        return np.ndarray((height, width, channels), dtype=np.uint8, buffer=shm.buf).copy()
    finally:
        name = shm._name  # pyright: ignore[reportPrivateUsage]
        shm.close()
        # The parent owns unlink exactly once.  A spawned child only attaches
        # transiently; unregister its attach handle so Python's resource
        # tracker does not emit a false leak warning or race the parent unlink.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(name, "shared_memory")
        except Exception:
            pass


def _unlink_shared_frame(shm: shared_memory.SharedMemory | None) -> None:
    if shm is None:
        return
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def _pose_is_aligned(estimator, frame: Any) -> bool:
    result = estimator.detect(frame)
    landmarks = getattr(getattr(result, "pose_landmarks", None), "landmark", None)
    if landmarks is None:
        return False
    try:
        left_shoulder, right_shoulder = landmarks[11], landmarks[12]
        left_hip, right_hip = landmarks[23], landmarks[24]
    except (IndexError, TypeError):
        return False
    points = (left_shoulder, right_shoulder, left_hip, right_hip)
    if any(float(point.visibility) < 0.55 for point in points):
        return False
    shoulder_x = (float(left_shoulder.x) + float(right_shoulder.x)) / 2
    shoulder_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2
    hip_x = (float(left_hip.x) + float(right_hip.x)) / 2
    hip_y = (float(left_hip.y) + float(right_hip.y)) / 2
    shoulder_span = abs(float(left_shoulder.x) - float(right_shoulder.x))
    torso_height = hip_y - shoulder_y
    return (
        0.30 <= shoulder_x <= 0.70 and 0.30 <= hip_x <= 0.70
        and 0.10 <= shoulder_y <= 0.68 and 0.35 <= hip_y <= 0.95
        and 0.14 <= shoulder_span <= 0.75 and 0.18 <= torso_height <= 0.62
    )


def _observe_frame(detector, estimator, frame: Any) -> AcquisitionObservation:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("acquisition_preview_encode_failed")
    status = detector.status()
    if not status.get("ready"):
        return AcquisitionObservation(encoded.tobytes(), "none", False)
    detections = detector.detect(frame)
    if len(detections) == 0:
        return AcquisitionObservation(encoded.tobytes(), "none", False)
    if len(detections) > 1:
        return AcquisitionObservation(encoded.tobytes(), "multiple", False)
    x, y, width, height = detections[0]["box"]
    frame_height, frame_width = frame.shape[:2]
    center_x = (x + width / 2) / max(frame_width, 1)
    center_y = (y + height / 2) / max(frame_height, 1)
    area = (width * height) / max(frame_width * frame_height, 1)
    aligned = (
        0.25 <= center_x <= 0.75 and 0.10 <= center_y <= 0.82
        and area >= 0.08 and _pose_is_aligned(estimator, frame)
    )
    return AcquisitionObservation(encoded.tobytes(), "single", aligned)


def acquisition_observer_entry(connection: Connection) -> None:
    """Windows-spawn import target: do not import the FastAPI application here."""
    from vision.person_detector import PersonDetector
    from vision.pose_estimator import PoseEstimator

    try:
        detector, estimator = PersonDetector(), PoseEstimator()
        connection.send(("ready", None))
        generation = 0
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command != "observe":
                raise RuntimeError("unknown acquisition observer command")
            generation += 1
            frame = _read_shared_frame(payload, generation=generation)
            observation = _observe_frame(detector, estimator, frame)
            connection.send(("ok", observation))
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        connection.close()


class AcquisitionObservationWorker:
    """One prewarmed child and one non-queued request slot."""

    def __init__(self, *, context=None, target=acquisition_observer_entry, target_args=()):
        self._context = context
        self._target = target
        self._target_args = tuple(target_args)
        self._state_lock = threading.RLock()
        self._request_slot = threading.Lock()
        self._parent: Connection | None = None
        self._process = None
        self._request_thread: threading.Thread | None = None
        self._fatal_error: str | None = None
        self._ready = False
        self._generation = 0
        self._request_generation = 0
        self._start_lock = asyncio.Lock()

    def _mp_context(self):
        multiprocessing.freeze_support()
        return self._context or multiprocessing.get_context("spawn")

    def _start(self) -> None:
        with self._state_lock:
            if self._fatal_error is not None:
                raise RuntimeError(f"acquisition observer unavailable: {self._fatal_error}")
            if self._process is not None and self._process.is_alive():
                return
            if self._process is not None:
                self._process.join(timeout=0)
                self._process = None
            if self._parent is not None:
                self._parent.close()
            parent, child = self._mp_context().Pipe(duplex=True)
            process = self._mp_context().Process(
                target=self._target,
                args=(child, *self._target_args),
                daemon=True,
            )
            process.start()
            child.close()
            self._parent, self._process = parent, process
            self._generation += 1
            self._request_generation = 0

    async def start(self) -> None:
        async with self._start_lock:
            if self.ready:
                return
            self._start()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                with self._state_lock:
                    parent, process, generation = self._parent, self._process, self._generation
                if parent is not None and parent.poll(0.002):
                    kind, payload = parent.recv()
                    if kind == "ready":
                        with self._state_lock:
                            if (
                                self._parent is parent
                                and self._process is process
                                and self._generation == generation
                            ):
                                self._ready = True
                        return
                    await self.abort_async(reason="prewarm_failed")
                    raise RuntimeError(str(payload))
                if process is None or not process.is_alive():
                    await self.abort_async(reason="prewarm_exited")
                    raise RuntimeError("acquisition observer prewarm exited")
                await asyncio.sleep(0.002)
            await self.abort_async(reason="prewarm_timeout")
            raise RuntimeError("acquisition observer prewarm timed out")

    async def observe(self, frame: Any, *, timeout: float = 15.0) -> AcquisitionObservation:
        await self.start()
        if not self._request_slot.acquire(blocking=False):
            raise RuntimeError("acquisition observer is busy")
        done, outcome = threading.Event(), {}

        def request() -> None:
            shm = None
            try:
                with self._state_lock:
                    parent, process, generation = self._parent, self._process, self._generation
                    if parent is None or process is None or not process.is_alive():
                        raise RuntimeError("acquisition observer unavailable")
                    self._request_generation += 1
                    request_generation = self._request_generation
                shared_frame, shm = _write_shared_frame(frame, generation=request_generation)
                parent.send((
                    "observe",
                    {
                        "kind": "shared_frame",
                        "name": shared_frame.name,
                        "shape": shared_frame.shape,
                        "dtype": shared_frame.dtype,
                        "nbytes": shared_frame.nbytes,
                        "generation": shared_frame.generation,
                        "processGeneration": generation,
                    },
                ))
                deadline = time.monotonic() + max(timeout, 0.001)
                while time.monotonic() < deadline:
                    if parent.poll(0.005):
                        kind, payload = parent.recv()
                        if kind == "ok":
                            with self._state_lock:
                                if (
                                    self._parent is not parent
                                    or self._process is not process
                                    or self._generation != generation
                                ):
                                    raise RuntimeError("acquisition observer response was stale")
                            outcome["value"] = payload
                            return
                        raise RuntimeError(str(payload))
                    if not process.is_alive():
                        raise RuntimeError("acquisition observer exited")
                raise TimeoutError("acquisition observation deadline exceeded")
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                _unlink_shared_frame(shm)
                self._request_slot.release()
                done.set()

        thread = threading.Thread(target=request, name="try-on-acquisition-request", daemon=False)
        with self._state_lock:
            self._request_thread = thread
        thread.start()
        try:
            while not done.is_set():
                await asyncio.sleep(0.002)
        except asyncio.CancelledError:
            await asyncio.shield(self.abort_async(reason="observation_cancelled"))
            while not done.is_set():
                await asyncio.sleep(0.002)
            raise
        if "error" in outcome:
            await self.abort_async(reason="observation_failed")
            raise outcome["error"]
        return outcome["value"]
        
    async def wait_idle(self, *, timeout: float | None = None) -> bool:
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + max(timeout, 0.001)
        thread = self._request_thread
        while thread is not None and thread.is_alive():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.002)
        if thread is not None:
            thread.join()
        self._request_thread = None
        return True

    def abort(self, *, reason: str) -> bool:
        with self._state_lock:
            parent, process = self._parent, self._process
            if process is None:
                self._parent = None
                self._ready = False
                self._generation += 1
                return True
            self._parent = None
            self._process = None
            self._ready = False
            self._generation += 1
            self._request_generation = 0
        if parent is not None:
            try:
                parent.close()
            except Exception:
                pass
        if process is not None:
            if process.is_alive():
                try:
                    process.kill()
                except (AttributeError, NotImplementedError, OSError, PermissionError):
                    try:
                        process.terminate()
                    except (AttributeError, NotImplementedError, OSError, PermissionError):
                        pass
                process.join(timeout=0.5)
            if process.is_alive():
                with self._state_lock:
                    self._process = process
                    self._fatal_error = reason
                return False
            process.join(timeout=0)
        return True

    async def abort_async(self, *, reason: str) -> bool:
        done, result = threading.Event(), {}
        def control() -> None:
            try:
                result["dead"] = self.abort(reason=reason)
            finally:
                done.set()
        thread = threading.Thread(target=control, name="try-on-acquisition-abort", daemon=False)
        thread.start()
        while not done.is_set():
            await asyncio.sleep(0.002)
        thread.join()
        idle = await self.wait_idle(timeout=0.05)
        if not idle:
            with self._state_lock:
                self._fatal_error = reason
            return False
        return bool(result.get("dead"))

    async def shutdown(self) -> None:
        if not await self.abort_async(reason="shutdown"):
            raise RuntimeError("acquisition observer could not be stopped")

    @property
    def assert_dead(self) -> bool:
        with self._state_lock:
            return self._process is None or not self._process.is_alive()

    @property
    def active_request_count(self) -> int:
        thread = self._request_thread
        return int(thread is not None and thread.is_alive())

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process is None or not process.is_alive():
                return None
            return int(process.pid)

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return bool(
                self._ready
                and self._fatal_error is None
                and self._process is not None
                and self._process.is_alive()
            )

    @property
    def fatal_error(self) -> str | None:
        with self._state_lock:
            return self._fatal_error
