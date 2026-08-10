"""Attempt-owned, forcibly terminable Fast frame and render workers.

OpenCV and DirectShow calls are synchronous and do not expose a dependable
cross-platform cancellation primitive.  They therefore run in a short-lived
process, not in the shared Python thread pool.  The parent owns the process and
terminates it on cancellation or deadline before a replacement is admitted.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from multiprocessing.connection import Connection
from typing import Any


class AttemptWorkerError(RuntimeError):
    pass


def _capture_worker(connection: Connection, role: str, config: dict, warmup_frames: int) -> None:
    try:
        from vision.camera_manager import capture_configured_frame

        connection.send(("ok", capture_configured_frame(role, config, warmup_frames=warmup_frames)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _render_worker(connection: Connection, frame: Any, garment: Any) -> None:
    try:
        from vision.fast_tryon import FastTryOnRuntime

        connection.send(("ok", FastTryOnRuntime().render(frame, garment)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


async def _wait_for_exit(process, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while process.is_alive() and loop.time() < deadline:
        await asyncio.sleep(0.005)
    if process.is_alive():
        process.kill()
        while process.is_alive() and loop.time() < deadline + 0.25:
            await asyncio.sleep(0.005)
    process.join(timeout=0)


async def _run_worker(target, args: tuple[Any, ...], *, timeout: float):
    """Return a child result, leaving no attempt process after any outcome."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(child, *args), daemon=True)
    process.start()
    child.close()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    try:
        while loop.time() < deadline:
            if parent.poll():
                kind, payload = parent.recv()
                await _wait_for_exit(process, loop.time() + 0.25)
                if kind == "ok":
                    return payload
                raise AttemptWorkerError(payload)
            if not process.is_alive():
                if parent.poll():
                    continue
                raise AttemptWorkerError(f"attempt worker exited with {process.exitcode}")
            await asyncio.sleep(0.005)
        raise TimeoutError("attempt worker deadline exceeded")
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        await _wait_for_exit(process, loop.time() + 0.5)


async def capture_attempt_frame(role: str, config: dict, *, warmup_frames: int, timeout: float):
    """Use the configured DirectShow/recorded adapter in an owned worker."""
    return await _run_worker(_capture_worker, (role, config, warmup_frames), timeout=timeout)


async def render_attempt_frame(frame, garment, *, timeout: float) -> bytes:
    """Render in an owned worker so a non-cooperative OpenCV call can end."""
    return await _run_worker(_render_worker, (frame, garment), timeout=timeout)
