"""Lifecycle-owned, bounded Fast render broker.

The Vision application starts one spawn-compatible child before accepting Fast
attempt work.  Attempt requests submit at most one bounded encoded job and
never synchronously create a process on the event-loop thread.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

import cv2
import numpy as np

from vision.render_worker_target import render_worker_entry


class AttemptWorkerError(RuntimeError):
    pass


_MAX_GARMENT_BYTES = 8 * 1024 * 1024
_MAX_FRAME_WIDTH = 1920
_MAX_FRAME_HEIGHT = 1080
_MAX_FRAME_RAW_BYTES = _MAX_FRAME_WIDTH * _MAX_FRAME_HEIGHT * 4
_CONSERVATIVE_PREPARE_BYTES_PER_SECOND = 64 * 1024 * 1024
_CONSERVATIVE_PREPARE_FIXED_SECONDS = 0.020
_START_TIMEOUT_SECONDS = 5.0


async def _run_joined_thread(name: str, operation):
    """Run one blocking control operation and never return before its thread."""
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


class FastRenderBroker:
    """Parent-side owner of the single Fast render process and job slot."""

    def __init__(
        self, *, context=None, target=render_worker_entry, target_args: tuple = ()
    ):
        self._context = context
        self._target = target
        self._target_args = tuple(target_args)
        self._state_lock = threading.RLock()
        self._job_slot = threading.Lock()
        self._parent: Connection | None = None
        self._process = None
        self._fatal_error: str | None = None
        self._ready = False
        self._quiesced = False
        self._request_threads: set[threading.Thread] = set()

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
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=self._target,
                args=(child, *self._target_args),
                daemon=True,
            )
            try:
                process.start()
            except BaseException as exc:
                parent.close()
                child.close()
                self._fatal_error = (
                    f"render_broker_start_failed: {type(exc).__name__}: {exc}"
                )
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                ) from exc
            child.close()
            self._parent = parent
            self._process = process

        if not parent.poll(_START_TIMEOUT_SECONDS):
            self._stop_sync(graceful=False, reason="render_broker_readiness_timeout")
            raise AttemptWorkerError("render broker readiness timeout")
        try:
            kind, payload = parent.recv()
        except (EOFError, OSError) as exc:
            self._stop_sync(graceful=False, reason="render_broker_readiness_failed")
            raise AttemptWorkerError(f"render broker readiness failed: {exc}") from exc
        if (
            kind != "ready"
            or not isinstance(payload, dict)
            or payload.get("pid") != process.pid
        ):
            self._stop_sync(graceful=False, reason="render_broker_readiness_invalid")
            raise AttemptWorkerError("render broker returned invalid readiness")
        with self._state_lock:
            self._ready = True

    async def start(self) -> None:
        with self._state_lock:
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            self._quiesced = False
        await _run_joined_thread("fast-render-start", self._start_sync)

    def _stop_sync(self, *, graceful: bool, reason: str) -> bool:
        """Stop and join truthfully; retain a live handle on any failed stop."""
        with self._state_lock:
            parent = self._parent
            process = self._process
            self._ready = False
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
                and parent is not None
                and process is not None
                and process.is_alive()
            ):
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
                detail = "; ".join(errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            if not join(0):
                self._parent = parent
                self._process = process
                detail = "; ".join(errors)
                self._fatal_error = f"{reason}: {detail}" if detail else reason
                return False
            self._parent = None
            self._process = None
            return True

    async def shutdown(self) -> None:
        with self._state_lock:
            self._quiesced = True

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
            return dead and requests_done

        dead = await _run_joined_thread("fast-render-shutdown", stop_and_join_requests)
        if not dead:
            raise AttemptWorkerError("render broker shutdown incomplete")

    def _request_sync(self, payload: dict, deadline: float):
        with self._state_lock:
            if self._fatal_error is not None:
                raise AttemptWorkerError(
                    f"render broker is unavailable: {self._fatal_error}"
                )
            parent = self._parent
            process = self._process
            if (
                not self._ready
                or parent is None
                or process is None
                or not process.is_alive()
            ):
                raise AttemptWorkerError("render broker is not ready")
        parent.send(("render", payload))
        while time.monotonic() < deadline:
            if parent.poll(0.005):
                try:
                    kind, response = parent.recv()
                except (EOFError, OSError) as exc:
                    raise AttemptWorkerError(
                        f"render broker connection closed: {exc}"
                    ) from exc
                if kind == "ok":
                    return response
                if kind == "garment_error":
                    from vision.fast_tryon import GarmentFetchError

                    raise GarmentFetchError(response)
                raise AttemptWorkerError(response)
            if not process.is_alive():
                process.join(timeout=0)
                raise AttemptWorkerError(
                    f"render broker exited with {process.exitcode}"
                )
        raise TimeoutError("render broker job deadline exceeded")

    async def _recover(self, reason: str) -> None:
        """Terminate/join the failed job owner, then prestart one replacement."""
        def recover() -> None:
            dead = self._stop_sync(graceful=False, reason=reason)
            if not dead:
                raise AttemptWorkerError("render broker recovery incomplete")
            with self._state_lock:
                if self._quiesced:
                    return
            self._start_sync()

        await _run_joined_thread("fast-render-recover", recover)

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
            recovery_error: BaseException | None = None
            try:
                await asyncio.shield(self._recover("render_job_cancelled"))
            except BaseException as exc:
                recovery_error = exc
            while not done.is_set():
                await asyncio.shield(asyncio.sleep(0.002))
            thread.join()
            if recovery_error is not None:
                raise recovery_error
            raise
        finally:
            if done.is_set():
                thread.join()
                with self._state_lock:
                    self._request_threads.discard(thread)
                self._job_slot.release()
        if "error" in outcome:
            error = outcome["error"]
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
    """Apply one absolute deadline before any encoding, serialization, or IPC."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    source_bytes = _validate_render_inputs(frame, garment_png, template)
    conservative_seconds = (
        _CONSERVATIVE_PREPARE_FIXED_SECONDS
        + source_bytes / _CONSERVATIVE_PREPARE_BYTES_PER_SECOND
    )
    if not broker.ready:
        raise AttemptWorkerError("render broker is not ready")
    if timeout <= 0 or loop.time() + conservative_seconds >= deadline:
        raise TimeoutError("render deadline exceeded before frame encoding")

    def encode_frame() -> bytes:
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("fast frame encode failed")
        return encoded.tobytes()

    frame_png = await _run_joined_thread("fast-render-encode", encode_frame)
    remaining = deadline - loop.time()
    actual_payload_bytes = len(frame_png) + len(garment_png)
    conservative_ipc_seconds = (
        _CONSERVATIVE_PREPARE_FIXED_SECONDS
        + actual_payload_bytes / _CONSERVATIVE_PREPARE_BYTES_PER_SECOND
    )
    if remaining <= 0 or conservative_ipc_seconds >= remaining:
        raise TimeoutError("render deadline exceeded before IPC")
    return await broker.render(
        {
            "framePng": frame_png,
            "garmentPng": garment_png,
            "garmentDigest": digest,
            "template": template,
        },
        deadline=time.monotonic() + remaining,
    )
