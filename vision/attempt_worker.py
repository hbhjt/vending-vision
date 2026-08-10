"""Lifecycle-owned, bounded Fast render broker.

The Vision application starts one spawn-compatible child before accepting Fast
attempt work.  Attempt requests submit at most one bounded raw-frame job and
never synchronously create a process on the event-loop thread.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import numpy as np

from vision.render_worker_target import render_worker_entry
from vision.shared_ipc_slot import SharedIpcSlot, run_shared_ipc_child, wait_for_event


class AttemptWorkerError(RuntimeError):
    pass


_MAX_GARMENT_BYTES = 8 * 1024 * 1024
_MAX_FRAME_WIDTH = 1920
_MAX_FRAME_HEIGHT = 1080
_MAX_FRAME_RAW_BYTES = _MAX_FRAME_WIDTH * _MAX_FRAME_HEIGHT * 3
_CONSERVATIVE_PREPARE_BYTES_PER_SECOND = 64 * 1024 * 1024
_CONSERVATIVE_PREPARE_FIXED_SECONDS = 0.020
_START_TIMEOUT_SECONDS = 5.0


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
    """Keep bounded process-control calls off the event loop and join them."""
    done = threading.Event()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = operation()
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=run, name=name, daemon=False)
    thread.start()
    try:
        while not done.is_set():
            await asyncio.sleep(0.002)
    except asyncio.CancelledError:
        while not done.is_set():
            await asyncio.shield(asyncio.sleep(0.002))
        thread.join()
        raise
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


async def _finish_cancelled_control(operation) -> Any:
    """Finish one shielded control coroutine despite repeated cancellation."""
    task = asyncio.create_task(operation)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


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
        self._request_threads: set[threading.Thread] = set()
        self._active_request = False
        self._process_generation = 0
        self._request_generation = 0

    def _mp_context(self):
        if self._context is not None:
            return self._context
        multiprocessing.freeze_support()
        return multiprocessing.get_context("spawn")

    @property
    def ready(self) -> bool:
        with self._state_lock:
            process = self._process
            return bool(
                self._ready
                and self._fatal_error is None
                and process is not None
                and process.is_alive()
            )

    @property
    def pose_ready(self) -> bool:
        with self._state_lock:
            return bool(self._pose_ready and self.ready)

    @property
    def pid(self) -> int | None:
        with self._state_lock:
            process = self._process
            if process is None or not process.is_alive():
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
            if self._process is not None and self._process.is_alive():
                return
            if self._process is not None:
                try:
                    self._process.join(timeout=0)
                except (OSError, PermissionError) as exc:
                    self._fatal_error = (
                        "render_broker_reap_failed: "
                        f"join {type(exc).__name__}: {exc}"
                    )
                    raise AttemptWorkerError(
                        f"render broker is unavailable: {self._fatal_error}"
                    ) from exc
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
            process = context.Process(
                target=run_shared_ipc_child,
                args=(self._target, slot.config, self._target_args),
                daemon=True,
            )
            try:
                process.start()
            except BaseException as exc:
                slot.close(unlink=True)
                self._fatal_error = (
                    f"render_broker_start_failed: {type(exc).__name__}: {exc}"
                )
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                ) from exc
            self._parent = None
            self._process = process
            self._slot = slot
            self._process_generation += 1
            self._request_generation = 0

        if not wait_for_event(slot.config["responseEvent"], _START_TIMEOUT_SECONDS, process=process):
            self._stop_sync(graceful=False, reason="render_broker_readiness_timeout")
            with self._state_lock:
                if self._fatal_error is None:
                    self._fatal_error = "render_broker_readiness_timeout"
            raise AttemptWorkerError("render broker readiness timeout")
        try:
            kind, payload, response_process_generation, _response_request_generation = slot.recv_response()
        except (EOFError, OSError) as exc:
            self._stop_sync(graceful=False, reason="render_broker_readiness_failed")
            with self._state_lock:
                if self._fatal_error is None:
                    self._fatal_error = (
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
            with self._state_lock:
                if self._fatal_error is None:
                    self._fatal_error = "render_broker_readiness_invalid"
            raise AttemptWorkerError("render broker returned invalid readiness")
        with self._state_lock:
            # Older injected test targets do not advertise poseReady and are
            # treated as compatible; the production target explicitly sets
            # false when MediaPipe cannot initialize.
            self._pose_ready = payload.get("poseReady", True) is True
            self._ready = True

    async def start(self) -> None:
        with self._state_lock:
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            self._quiesced = False
        await _run_broker_control("fast-render-start", self._start_sync)

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

            def join(timeout: float) -> bool:
                try:
                    process.join(timeout=timeout)
                    return True
                except (OSError, PermissionError) as exc:
                    record("join", exc)
                    return False

            if (
                graceful
                and not has_active_request
                and slot is not None
                and process is not None
                and process.is_alive()
            ):
                try:
                    slot.submit(
                        "shutdown",
                        None,
                        process_generation=self._process_generation,
                        request_generation=self._request_generation + 1,
                    )
                    wait_for_event(slot.config["responseEvent"], 0.25, process=process)
                except Exception:
                    pass
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
            if process.is_alive():
                try:
                    process.kill()
                except (AttributeError, NotImplementedError):
                    terminate("terminate-fallback")
                except (OSError, PermissionError) as exc:
                    record("kill", exc)
                    terminate("terminate-after-kill-error")
                join(0.5)
            if process.is_alive():
                terminate("terminate")
                join(0.5)
            if process.is_alive():
                self._parent = parent
                self._process = process
                self._slot = slot
                detail = "; ".join(errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            if not join(0):
                self._parent = parent
                self._process = process
                self._slot = slot
                detail = "; ".join(errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            self._parent = None
            self._process = None
            self._slot = None
            if slot is not None:
                slot.close(unlink=True)
            return True

    async def shutdown(self) -> None:
        self.quiesce()

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
                    process.join(timeout=0.5)
                    delayed_dead = not process.is_alive()
                    if delayed_dead:
                        process.join(timeout=0)
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
            raise AttemptWorkerError("render broker shutdown incomplete")

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
                or not process.is_alive()
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
                        kind, response, response_process_generation, response_request_generation = slot.recv_response()
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
                        from vision.fast_tryon import GarmentFetchError

                        raise GarmentFetchError(response)
                    if kind == "pose_error":
                        from vision.fast_tryon import PoseUnavailableError

                        # Child diagnostics may contain model paths or native
                        # exception detail.  Pose absence is a normal attempt
                        # outcome, and its parent-facing contract is stable.
                        raise PoseUnavailableError("pose_unavailable")
                    raise AttemptWorkerError(response)
                if not process.is_alive():
                    process.join(timeout=0)
                    raise AttemptWorkerError(
                        f"render broker exited with {process.exitcode}"
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
                    try:
                        process.join(timeout=0.5)
                    except (OSError, PermissionError) as exc:
                        raise AttemptWorkerError(
                            f"render broker recovery incomplete: join {type(exc).__name__}: {exc}"
                        ) from exc
                if process is None or process.is_alive():
                    raise AttemptWorkerError("render broker recovery incomplete")
                try:
                    process.join(timeout=0)
                except (OSError, PermissionError) as exc:
                    raise AttemptWorkerError(
                        f"render broker recovery incomplete: join {type(exc).__name__}: {exc}"
                    ) from exc
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
                    return
            self._start_sync()

        await _run_broker_control("fast-render-recover", recover)

    async def _restart_after_recovery(self) -> None:
        """Prestart one replacement only after the prior request was joined."""

        def restart() -> None:
            with self._state_lock:
                if self._quiesced:
                    return
            self._start_sync()

        await _run_broker_control("fast-render-restart", restart)

    async def _restart_after_delayed_cancel_death(self, reason: str) -> None:
        """Recover when kill is observed slightly after the control timeout."""

        def restart() -> None:
            with self._state_lock:
                process = self._process
                slot = self._slot
                if self._fatal_error != reason:
                    return
            if process is not None:
                try:
                    process.join(timeout=0.2)
                except (OSError, PermissionError) as exc:
                    raise AttemptWorkerError(
                        f"render broker recovery incomplete: join {type(exc).__name__}: {exc}"
                    ) from exc
                if process.is_alive():
                    return
                try:
                    process.join(timeout=0)
                except (OSError, PermissionError) as exc:
                    raise AttemptWorkerError(
                        f"render broker recovery incomplete: join {type(exc).__name__}: {exc}"
                    ) from exc
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
            self._start_sync()

        await _run_broker_control("fast-render-delayed-cancel-restart", restart)

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
            from vision.fast_tryon import GarmentFetchError, PoseUnavailableError

            if isinstance(error, (GarmentFetchError, PoseUnavailableError)):
                raise error
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
) -> bytes:
    """Apply one absolute deadline before any bounded frame copy or IPC."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    source_bytes = _validate_render_inputs(frame, garment_png, template)
    conservative_seconds = (
        _CONSERVATIVE_PREPARE_FIXED_SECONDS
        + source_bytes / _CONSERVATIVE_PREPARE_BYTES_PER_SECOND
    )
    if not broker.ready:
        raise AttemptWorkerError("render broker is not ready")
    if not broker.pose_ready:
        from vision.fast_tryon import PoseUnavailableError

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
        },
        deadline=time.monotonic() + remaining,
    )
