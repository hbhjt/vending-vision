"""Packaged spawn/shared-memory self-check for V2 try-on worker boundaries."""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing import shared_memory
from uuid import uuid4

from vision.acquisition_observer import acquisition_observer_entry
from vision.render_worker_target import render_worker_entry
from vision.shared_ipc_slot import SharedIpcSlot, run_shared_ipc_child


def _shared_slot_self_check_entry(connection) -> None:
    """Spawn-safe child entry used by the packaged verifier."""
    # Keep these real production entries import-reachable inside the frozen
    # child.  The self-check avoids loading camera/model dependencies, but it
    # catches missing hidden imports and spawn bootstrap regressions at the same
    # multiprocessing boundary used by acquisition/render workers.
    if render_worker_entry is None or acquisition_observer_entry is None:
        raise RuntimeError("try-on worker entries were not importable")
    connection.send(("ready", {"pid": multiprocessing.current_process().pid}))
    while True:
        command, payload = connection.recv()
        if command == "shutdown":
            connection.send(("ok", {"stopped": True}))
            return
        if command != "self_check":
            raise RuntimeError(f"unknown worker self-check command: {command}")
        name = payload.get("frameSharedName")
        nbytes = payload.get("frameSharedBytes")
        if not isinstance(name, str) or type(nbytes) is not int or nbytes <= 0:
            raise RuntimeError("invalid self-check shared frame metadata")
        shm = shared_memory.SharedMemory(name=name)
        try:
            observed = bytes(shm.buf[:nbytes])
        finally:
            shm.close()
        if observed != b"vem-v2-worker-self-check":
            raise RuntimeError("self-check shared memory payload mismatch")
        connection.send(
            (
                "ok",
                {
                    "pngMagic": list(b"\x89PNG\r\n\x1a\n"),
                    "entries": ["render", "acquisition"],
                },
            )
        )


def verify_v2_try_on_workers() -> None:
    """Exercise the frozen spawn/shared-memory worker boundary without cameras."""
    multiprocessing.freeze_support()
    context = multiprocessing.get_context("spawn")
    slot = SharedIpcSlot(
        context=context,
        name_prefix="vem_worker_probe",
        request_bytes=0,
        response_bytes=64 * 1024,
    )
    frame_shm = shared_memory.SharedMemory(
        create=True,
        size=len(b"vem-v2-worker-self-check"),
        name=f"vem_worker_probe_{uuid4().hex}",
    )
    process = None
    try:
        frame_shm.buf[: len(b"vem-v2-worker-self-check")] = b"vem-v2-worker-self-check"
        slot_config = dict(slot.config)
        slot_config["expectedProcessGeneration"] = 1
        process = context.Process(
            target=run_shared_ipc_child,
            args=(_shared_slot_self_check_entry, slot_config, ()),
            daemon=True,
        )
        process.start()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not slot.poll_response():
            if not process.is_alive():
                raise RuntimeError(f"try-on worker self-check exited early: {process.exitcode}")
            time.sleep(0.01)
        if not slot.poll_response():
            raise TimeoutError("try-on worker self-check readiness timed out")
        kind, payload, process_generation, request_generation = slot.recv_response()
        if (
            kind != "ready"
            or not isinstance(payload, dict)
            or payload.get("pid") != process.pid
            or process_generation != 0
            or request_generation != 0
        ):
            raise RuntimeError(f"invalid try-on worker self-check readiness: {kind} {payload}")
        slot.submit(
            "self_check",
            {
                "frameSharedName": frame_shm.name,
                "frameSharedBytes": len(b"vem-v2-worker-self-check"),
            },
            process_generation=1,
            request_generation=1,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not slot.poll_response():
            if not process.is_alive():
                raise RuntimeError(f"try-on worker self-check exited during request: {process.exitcode}")
            time.sleep(0.01)
        if not slot.poll_response():
            raise TimeoutError("try-on worker self-check request timed out")
        kind, payload, process_generation, request_generation = slot.recv_response(
            expected_process_generation=1,
            expected_request_generation=1,
        )
        if kind != "ok" or payload.get("pngMagic") != list(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"invalid try-on worker self-check result: {kind} {payload}")
        if set(payload.get("entries", [])) != {"render", "acquisition"}:
            raise RuntimeError(f"try-on worker entry imports were incomplete: {payload}")
        slot.submit(
            "shutdown",
            None,
            process_generation=1,
            request_generation=2,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not slot.poll_response():
            if not process.is_alive():
                break
            time.sleep(0.01)
        if slot.poll_response():
            slot.recv_response(expected_process_generation=1, expected_request_generation=2)
        process.join(timeout=5.0)
        if process.is_alive():
            raise RuntimeError("try-on worker self-check did not stop")
        if process.exitcode not in {0, None}:
            raise RuntimeError(f"try-on worker self-check failed with {process.exitcode}")
    finally:
        try:
            frame_shm.close()
        finally:
            try:
                frame_shm.unlink()
            except FileNotFoundError:
                pass
        slot.close(unlink=True)
        if process is not None:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
            try:
                process.close()
            except Exception:
                pass
