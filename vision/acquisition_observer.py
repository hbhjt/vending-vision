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
import re
import time
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_sentinels
from multiprocessing.connection import Connection
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from vision.shared_ipc_slot import SharedIpcError, SharedIpcSlot, run_shared_ipc_child


MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1920
MAX_FRAME_RAW_BYTES = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3
STOP_CONFIRM_TIMEOUT_SECONDS = 2.0
_ACQ_SHARED_NAME = re.compile(r"^vem_acq_[0-9a-f]{32}$")


@dataclass(frozen=True)
class AcquisitionObservation:
    jpeg: bytes
    occupancy: str
    aligned: bool


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
                try:
                    return not process.is_alive()
                except (OSError, PermissionError):
                    return True
            return not process.is_alive()
        except (OSError, PermissionError, ValueError):
            pass
    deadline = time.monotonic() + max(timeout, 0.0)
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
        or height > MAX_FRAME_HEIGHT
        or width > MAX_FRAME_WIDTH
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


def _read_shared_frame(
    metadata: Any, *, generation: int, process_generation: int
) -> np.ndarray:
    if not isinstance(metadata, dict) or set(metadata) != {
        "kind",
        "name",
        "shape",
        "dtype",
        "nbytes",
        "generation",
        "processGeneration",
    } or metadata.get("kind") != "shared_frame":
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
        type(metadata.get("generation")) is not int
        or metadata.get("generation") != generation
        or type(metadata.get("processGeneration")) is not int
        or metadata.get("processGeneration") != process_generation
        or dtype != "uint8"
        or channels != 3
        or height <= 0
        or width <= 0
        or height > MAX_FRAME_HEIGHT
        or width > MAX_FRAME_WIDTH
        or type(nbytes) is not int
        or nbytes != height * width * channels
        or nbytes > MAX_FRAME_RAW_BYTES
    ):
        raise ValueError("acquisition frame metadata exceeds cap")
    name = metadata.get("name")
    if not isinstance(name, str) or not _ACQ_SHARED_NAME.fullmatch(name):
        raise ValueError("invalid acquisition frame shared memory name")
    shm = shared_memory.SharedMemory(name=name)
    try:
        return np.ndarray((height, width, channels), dtype=np.uint8, buffer=shm.buf).copy()
    finally:
        shm.close()


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
    except (IndexError, TypeError):
        return False
    if float(getattr(left_shoulder, "visibility", 1.0)) < 0.30 or float(
        getattr(right_shoulder, "visibility", 1.0)
    ) < 0.30:
        return False
    shoulder_x = (float(left_shoulder.x) + float(right_shoulder.x)) / 2
    shoulder_y = (float(left_shoulder.y) + float(right_shoulder.y)) / 2
    shoulder_span = abs(float(left_shoulder.x) - float(right_shoulder.x))
    return (
        0.15 <= shoulder_x <= 0.85
        and 0.05 <= shoulder_y <= 0.85
        and 0.08 <= shoulder_span <= 0.90
    )


def _observe_frame(detector, estimator, frame: Any) -> AcquisitionObservation:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("acquisition_preview_encode_failed")
    status = detector.status()
    detections = detector.detect(frame) if status.get("ready") else []
    if len(detections) > 1:
        return AcquisitionObservation(encoded.tobytes(), "multiple", False)
    aligned = _pose_is_aligned(estimator, frame)
    occupancy = "single" if (len(detections) == 1 or aligned) else "none"
    return AcquisitionObservation(encoded.tobytes(), occupancy, aligned)


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
            frame = _read_shared_frame(
                payload,
                generation=generation,
                process_generation=connection.expected_process_generation or 0,
            )
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

    def __init__(
        self,
        *,
        context=None,
        target=acquisition_observer_entry,
        target_args=(),
        stop_timeout_seconds: float = STOP_CONFIRM_TIMEOUT_SECONDS,
    ):
        self._context = context
        self._target = target
        self._target_args = tuple(target_args)
        self._state_lock = multiprocessing.get_context("spawn").RLock()
        self._request_slot = multiprocessing.get_context("spawn").Lock()
        self._parent: Connection | None = None
        self._process = None
        self._slot: SharedIpcSlot | None = None
        self._active_request = False
        self._fatal_error: str | None = None
        self._ready = False
        self._generation = 0
        self._request_generation = 0
        self._start_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._stop_timeout_seconds = max(float(stop_timeout_seconds), 0.001)

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
                _close_dead_process(self._process)
                self._process = None
            if self._parent is not None:
                self._parent.close()
            if self._slot is not None:
                self._slot.close(unlink=True)
            slot = SharedIpcSlot(
                context=self._mp_context(),
                name_prefix="vem_acq",
                request_bytes=0,
                response_bytes=MAX_FRAME_RAW_BYTES,
            )
            next_generation = self._generation + 1
            slot_config = dict(slot.config)
            slot_config["expectedProcessGeneration"] = next_generation
            process = self._mp_context().Process(
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
                self._parent = None
                self._process = None
                self._slot = None
                self._ready = False
                self._fatal_error = (
                    f"acquisition_observer_start_failed: {type(exc).__name__}: {exc}"
                )
                raise RuntimeError(self._fatal_error) from exc
            self._parent, self._process, self._slot = None, process, slot
            self._generation = next_generation
            self._request_generation = 0

    def _finalize_dead_child(self, *, reason: str) -> None:
        with self._state_lock:
            process, slot = self._process, self._slot
            if (
                process is not None
                and process.is_alive()
                and not _wait_process_dead(process, STOP_CONFIRM_TIMEOUT_SECONDS)
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

    async def start(self) -> None:
        async with self._start_lock:
            with self._state_lock:
                if (
                    self._ready
                    and self._process is not None
                    and self._process.is_alive()
                    and self._slot is not None
                ):
                    return
            cancelled = False
            start_task = asyncio.create_task(
                asyncio.to_thread(self._start),
                name="acquisition-observer-start",
            )
            while not start_task.done():
                try:
                    await asyncio.shield(start_task)
                except asyncio.CancelledError:
                    cancelled = True
                    continue
            start_task.result()
            if cancelled:
                cleanup = asyncio.create_task(
                    self.abort_async(reason="acquisition_observer_start_cancelled"),
                    name="acquisition-observer-start-cancel-cleanup",
                )
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
                raise asyncio.CancelledError
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                with self._state_lock:
                    process, generation, slot = self._process, self._generation, self._slot
                if slot is not None and slot.poll_response():
                    try:
                        kind, payload, process_generation, _request_generation = slot.recv_response()
                    except (SharedIpcError, EOFError, OSError) as exc:
                        await self.abort_async(reason="prewarm_failed")
                        self._finalize_dead_child(reason=f"prewarm_failed: {exc}")
                        raise RuntimeError(f"prewarm_failed: {exc}") from exc
                    if (
                        kind == "ready"
                        and (payload is None or isinstance(payload, dict))
                        and process_generation == 0
                    ):
                        with self._state_lock:
                            if (
                                self._process is process
                                and self._generation == generation
                            ):
                                self._ready = True
                                return
                    await self.abort_async(reason="prewarm_failed")
                    self._finalize_dead_child(reason="prewarm_failed: invalid readiness")
                    raise RuntimeError("prewarm_failed: invalid readiness")
                if process is None or not process.is_alive():
                    await self.abort_async(reason="prewarm_exited")
                    self._finalize_dead_child(reason="prewarm_exited")
                    raise RuntimeError("acquisition observer prewarm exited")
                await asyncio.sleep(0.002)
            await self.abort_async(reason="prewarm_timeout")
            self._finalize_dead_child(reason="prewarm_timeout")
            raise RuntimeError("acquisition observer prewarm timed out")

    async def observe(self, frame: Any, *, timeout: float = 15.0) -> AcquisitionObservation:
        await self.start()
        if not self._request_slot.acquire(block=False):
            raise RuntimeError("acquisition observer is busy")
        shm = None
        try:
            with self._state_lock:
                process, generation, slot = self._process, self._generation, self._slot
                if slot is None or process is None or not process.is_alive():
                    raise RuntimeError("acquisition observer unavailable")
                self._request_generation += 1
                request_generation = self._request_generation
                self._active_request = True
            shared_frame, shm = _write_shared_frame(frame, generation=request_generation)
            slot.submit(
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
                process_generation=generation,
                request_generation=request_generation,
            )
            deadline = time.monotonic() + max(timeout, 0.001)
            while time.monotonic() < deadline:
                if slot.poll_response():
                    kind, payload, process_generation, response_generation = slot.recv_response(
                        expected_process_generation=generation,
                        expected_request_generation=request_generation,
                    )
                    if kind == "ok":
                        with self._state_lock:
                            if (
                                self._process is not process
                                or self._generation != generation
                                or process_generation != generation
                                or response_generation != request_generation
                            ):
                                raise RuntimeError("acquisition observer response was stale")
                        return payload
                    raise RuntimeError(str(payload))
                if not process.is_alive():
                    raise RuntimeError("acquisition observer exited")
                await asyncio.sleep(0.002)
            raise TimeoutError("acquisition observation deadline exceeded")
        except asyncio.CancelledError:
            cancelled = asyncio.CancelledError()
            cleanup = asyncio.create_task(
                self.abort_async(reason="observation_cancelled"),
                name="acquisition-observer-cancel-cleanup",
            )
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise cancelled
        except BaseException:
            await self.abort_async(reason="observation_failed")
            raise
        finally:
            _unlink_shared_frame(shm)
            with self._state_lock:
                self._active_request = False
            try:
                self._request_slot.release()
            except ValueError:
                pass
        
    async def wait_idle(self, *, timeout: float | None = None) -> bool:
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + max(timeout, 0.001)
        while self.active_request_count:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.002)
        return True

    def abort(self, *, reason: str) -> bool:
        with self._state_lock:
            parent, process, slot = self._parent, self._process, self._slot
            if process is None:
                self._parent = None
                self._slot = None
                self._ready = False
                self._generation += 1
                if slot is not None:
                    slot.close(unlink=True)
                return True
            self._ready = False
            self._request_generation = 0
        errors: list[str] = []
        if parent is not None:
            try:
                parent.close()
            except Exception:
                pass
        if process is not None:
            if process.is_alive():
                if _should_call_kill(process):
                    try:
                        process.kill()
                    except (AttributeError, NotImplementedError):
                        try:
                            process.terminate()
                        except (AttributeError, NotImplementedError, OSError, PermissionError) as exc:
                            errors.append(f"terminate {type(exc).__name__}: {exc}")
                    except (OSError, PermissionError) as exc:
                        errors.append(f"kill {type(exc).__name__}: {exc}")
                        try:
                            process.terminate()
                        except (AttributeError, NotImplementedError, OSError, PermissionError) as terminate_exc:
                            errors.append(
                                f"terminate {type(terminate_exc).__name__}: {terminate_exc}"
                            )
                else:
                    try:
                        process.terminate()
                    except (AttributeError, NotImplementedError, OSError, PermissionError) as exc:
                        errors.append(f"terminate {type(exc).__name__}: {exc}")
            if process.is_alive():
                _wait_process_dead(process, self._stop_timeout_seconds)
            if process.is_alive():
                with self._state_lock:
                    self._parent = parent
                    self._process = process
                    self._slot = slot
                    self._fatal_error = reason
                return False
            _close_dead_process(process)
        if slot is not None:
            slot.close(unlink=True)
        with self._state_lock:
            self._parent = None
            self._process = None
            self._slot = None
            self._generation += 1
            self._fatal_error = None
        return True

    async def abort_async(self, *, reason: str) -> bool:
        async with self._stop_lock:
            return await self._abort_async_once(reason=reason)

    async def _abort_async_once(self, *, reason: str) -> bool:
        """Stop physically, preserving the live handle until death is observed."""
        with self._state_lock:
            parent, process, slot = self._parent, self._process, self._slot
            self._ready = False
            self._request_generation = 0
        if parent is not None:
            try:
                parent.close()
            except Exception:
                pass
        if process is None:
            if slot is not None:
                slot.close(unlink=True)
            with self._state_lock:
                if self._process is None:
                    self._parent = None
                    self._slot = None
                    self._generation += 1
                    self._fatal_error = None
            return True

        errors: list[str] = []

        def signal_stop() -> None:
            if not process.is_alive():
                return
            if _should_call_kill(process):
                try:
                    process.kill()
                    return
                except (AttributeError, NotImplementedError):
                    pass
                except (OSError, PermissionError) as exc:
                    errors.append(f"kill {type(exc).__name__}: {exc}")
            try:
                process.terminate()
            except (AttributeError, NotImplementedError, OSError, PermissionError) as exc:
                errors.append(f"terminate {type(exc).__name__}: {exc}")

        control = asyncio.create_task(
            asyncio.to_thread(signal_stop),
            name="acquisition-observer-stop-signal",
        )
        while not control.done():
            try:
                await asyncio.shield(control)
            except asyncio.CancelledError:
                continue
        try:
            control.result()
        except BaseException as exc:
            errors.append(f"control {type(exc).__name__}: {exc}")

        deadline = asyncio.get_running_loop().time() + self._stop_timeout_seconds
        while process.is_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        if process.is_alive():
            with self._state_lock:
                if self._process is process:
                    self._parent = parent
                    self._process = process
                    self._slot = slot
                    detail = "; ".join(errors)
                    self._fatal_error = f"{reason}: {detail}" if detail else reason
            return False

        _close_dead_process(process)
        if slot is not None:
            slot.close(unlink=True)
        with self._state_lock:
            if self._process is process:
                self._parent = None
                self._process = None
                self._slot = None
                self._generation += 1
                self._fatal_error = None
        return True

    async def shutdown(self) -> None:
        with self._state_lock:
            process, slot, generation = self._process, self._slot, self._generation
            can_graceful = (
                self._ready
                and process is not None
                and process.is_alive()
                and slot is not None
                and not self._active_request
            )
        if can_graceful:
            try:
                slot.submit(
                    "shutdown",
                    None,
                    process_generation=generation,
                    request_generation=self._request_generation + 1,
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    if slot.poll_response():
                        try:
                            slot.recv_response(
                                expected_process_generation=generation,
                                expected_request_generation=self._request_generation + 1,
                            )
                        except (SharedIpcError, EOFError, OSError):
                            break
                        _wait_process_dead(process, 1.0)
                        self._finalize_dead_child(reason="shutdown")
                        if self.assert_dead:
                            with self._state_lock:
                                if self._fatal_error == "shutdown":
                                    self._fatal_error = None
                            return
                        break
                    if not process.is_alive():
                        self._finalize_dead_child(reason="shutdown")
                        with self._state_lock:
                            if self._fatal_error == "shutdown":
                                self._fatal_error = None
                        return
                    await asyncio.sleep(0.01)
            except (SharedIpcError, EOFError, OSError):
                pass
        if not await self.abort_async(reason="shutdown"):
            raise RuntimeError("acquisition observer could not be stopped")
        with self._state_lock:
            if self._fatal_error == "shutdown":
                self._fatal_error = None

    @property
    def assert_dead(self) -> bool:
        with self._state_lock:
            if self._process is None:
                return True
            try:
                return not self._process.is_alive()
            except ValueError:
                return True

    @property
    def active_request_count(self) -> int:
        with self._state_lock:
            return int(self._active_request)

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process is None:
                return None
            if not process.is_alive():
                slot = self._slot
                _close_dead_process(process)
                if slot is not None:
                    slot.close(unlink=True)
                self._parent = None
                self._process = None
                self._slot = None
                self._ready = False
                return None
            return int(process.pid)

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return bool(
                self._fatal_error is None
                and (
                    self._process is None
                    or self._process.is_alive()
                )
            )

    @property
    def fatal_error(self) -> str | None:
        with self._state_lock:
            return self._fatal_error
