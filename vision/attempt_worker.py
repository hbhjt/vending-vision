"""Attempt-owned, forcibly terminable Fast render workers.

Fast frame acquisition stays in the parent Vision process so DirectShow and
recorded-video sources share the one camera-manager owner with profile and
presence work.  CPU rendering runs in a short-lived spawned process owned by the
attempt; cancellation or deadline terminates and joins that child before a
replacement is admitted.
"""

from __future__ import annotations

import asyncio
import multiprocessing
from multiprocessing.connection import Connection
from typing import Any


class AttemptWorkerError(RuntimeError):
    pass


_INLINE_ARGUMENT_BYTE_LIMIT = 1024 * 1024


def _estimated_argument_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, (tuple, list)):
        return sum(_estimated_argument_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _estimated_argument_bytes(key) + _estimated_argument_bytes(item)
            for key, item in value.items()
        )
    return 0


def _render_worker(connection: Connection, frame: Any, garment: Any) -> None:
    try:
        from vision.fast_tryon import FastTryOnRuntime

        connection.send(("ok", FastTryOnRuntime().render(frame, garment)))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _child_entry(connection: Connection, target) -> None:
    try:
        args = connection.recv()
        target(connection, *args)
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        except BaseException:
            pass
    finally:
        connection.close()


async def _wait_for_exit(process, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while process.is_alive() and loop.time() < deadline:
        await asyncio.sleep(0.005)
    if process.is_alive():
        try:
            process.kill()
        except AttributeError:
            process.terminate()
        while process.is_alive() and loop.time() < deadline + 0.25:
            await asyncio.sleep(0.005)
    process.join(timeout=0.05)


async def _run_worker(target, args: tuple[Any, ...], *, timeout: float):
    """Return a child result, leaving no attempt process after any outcome."""
    multiprocessing.freeze_support()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_child_entry, args=(child, target), daemon=True)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    started = False
    try:
        if timeout <= 0:
            raise TimeoutError("attempt worker deadline exceeded before start")
        if (
            timeout < 0.05
            and _estimated_argument_bytes(args) > _INLINE_ARGUMENT_BYTE_LIMIT
        ):
            raise TimeoutError("attempt worker deadline exceeded before IPC")
        process.start()
        started = True
        child.close()
        if loop.time() >= deadline:
            raise TimeoutError("attempt worker deadline exceeded before IPC")
        parent.send(args)
        while loop.time() < deadline:
            if parent.poll():
                try:
                    kind, payload = parent.recv()
                except (EOFError, OSError) as exc:
                    raise AttemptWorkerError(
                        f"attempt worker connection closed: {exc}"
                    ) from exc
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
        child.close()
        if started and process.is_alive():
            process.terminate()
        if started:
            await _wait_for_exit(process, loop.time() + 0.5)


async def render_attempt_frame(frame, garment, *, timeout: float) -> bytes:
    """Render in an owned worker so a non-cooperative OpenCV call can end."""
    return await _run_worker(_render_worker, (frame, garment), timeout=timeout)


async def run_attempt_worker(target, args: tuple[Any, ...], *, timeout: float):
    """Run one attempt-owned child boundary with guaranteed cleanup."""
    return await _run_worker(target, args, timeout=timeout)
