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
        process.kill()
        while process.is_alive() and loop.time() < deadline + 0.25:
            await asyncio.sleep(0.005)
    process.join(timeout=0)


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
        await asyncio.wait_for(asyncio.to_thread(process.start), timeout=max(0, deadline - loop.time()))
        started = True
        child.close()
        await asyncio.wait_for(asyncio.to_thread(parent.send, args), timeout=max(0, deadline - loop.time()))
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
        child.close()
        if started and process.is_alive():
            process.terminate()
        if started:
            await _wait_for_exit(process, loop.time() + 0.5)


async def render_attempt_frame(frame, garment, *, timeout: float) -> bytes:
    """Render in an owned worker so a non-cooperative OpenCV call can end."""
    return await _run_worker(_render_worker, (frame, garment), timeout=timeout)
