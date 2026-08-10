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
from typing import Any

import cv2


@dataclass(frozen=True)
class AcquisitionObservation:
    jpeg: bytes
    occupancy: str
    aligned: bool


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
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command != "observe":
                raise RuntimeError("unknown acquisition observer command")
            observation = _observe_frame(detector, estimator, payload)
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

    async def start(self) -> None:
        async with self._start_lock:
            if self._ready:
                return
            self._start()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                with self._state_lock:
                    parent, process = self._parent, self._process
                if parent is not None and parent.poll(0.002):
                    kind, payload = parent.recv()
                    if kind == "ready":
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
            try:
                with self._state_lock:
                    parent, process = self._parent, self._process
                    if parent is None or process is None or not process.is_alive():
                        raise RuntimeError("acquisition observer unavailable")
                    parent.send(("observe", frame))
                deadline = time.monotonic() + max(timeout, 0.001)
                while time.monotonic() < deadline:
                    if parent.poll(0.005):
                        kind, payload = parent.recv()
                        if kind == "ok":
                            outcome["value"] = payload
                            return
                        raise RuntimeError(str(payload))
                    if not process.is_alive():
                        raise RuntimeError("acquisition observer exited")
                raise TimeoutError("acquisition observation deadline exceeded")
            except BaseException as exc:
                outcome["error"] = exc
            finally:
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
        
    async def wait_idle(self) -> None:
        thread = self._request_thread
        while thread is not None and thread.is_alive():
            await asyncio.sleep(0.002)
        if thread is not None:
            thread.join()
        self._request_thread = None

    def abort(self, *, reason: str) -> bool:
        with self._state_lock:
            parent, process = self._parent, self._process
            if parent is not None:
                try:
                    parent.close()
                except Exception:
                    pass
            if process is None:
                self._parent = None
                return True
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
                self._fatal_error = reason
                return False
            process.join(timeout=0)
            self._parent = None
            self._process = None
            self._ready = False
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
        await self.wait_idle()
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
