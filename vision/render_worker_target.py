"""Spawn-safe target for the lifecycle-owned Fast render process.

The target deliberately has no dependency on the FastAPI application or model
runtime.  PyInstaller and Windows ``spawn`` can import it as a plain module.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from multiprocessing import shared_memory
from multiprocessing.connection import Connection


MAX_GARMENT_BYTES = 8 * 1024 * 1024
MAX_FRAME_WIDTH = 1920
MAX_FRAME_HEIGHT = 1920
MAX_FRAME_RAW_BYTES = MAX_FRAME_WIDTH * MAX_FRAME_HEIGHT * 3
MAX_RESULT_BYTES = 16 * 1024 * 1024
_RENDER_SHARED_NAME = re.compile(r"^vem_render_[0-9a-f]{32}$")

# Kept module-local so the official MediaPipe estimator is initialized once
# per spawn worker, rather than once per attempt.  Dynamic import keeps the
# spawn target free of an app/parent import cycle and remains PyInstaller
# discoverable through the hidden import in the spec.
_FAST_RUNTIME = None
_POSE_READY = False


def _initialize_runtime():
    global _FAST_RUNTIME, _POSE_READY
    from vision.fast_tryon import FastTryOnRuntime

    try:
        pose_module = __import__("vision." + "pose_estimator", fromlist=["PoseEstimator"])
        estimator = pose_module.PoseEstimator()
        _FAST_RUNTIME = FastTryOnRuntime(pose_estimator=estimator)
        _POSE_READY = True
    except Exception:
        # The worker stays alive so the parent can report Fast degradation;
        # camera/presence/health remain owned by the main Vision process.
        _FAST_RUNTIME = FastTryOnRuntime(pose_estimator=None)
        _POSE_READY = False
    return _FAST_RUNTIME


def _render(
    payload: dict,
    *,
    expected_process_generation: int | None = 1,
    expected_request_generation: int | None = None,
) -> bytes:
    """Decode, prepare and render entirely inside the bounded child."""
    import numpy as np

    from vision.fast_tryon import GarmentFetchError, ValidatedGarmentSource

    if not isinstance(payload, dict) or set(payload) != {
        "frameShared",
        "garmentPng",
        "garmentDigest",
        "template",
        "garmentScale",
    }:
        raise ValueError("invalid render payload")
    frame_shared = payload["frameShared"]
    garment_png = payload["garmentPng"]
    if not isinstance(frame_shared, dict) or set(frame_shared) != {
        "kind",
        "name",
        "shape",
        "dtype",
        "nbytes",
        "generation",
        "processGeneration",
    } or frame_shared.get("kind") != "shared_frame":
        raise ValueError("raw frame metadata is invalid")
    frame_shape = frame_shared.get("shape")
    if (
        not isinstance(frame_shape, (tuple, list))
        or len(frame_shape) != 3
        or any(type(value) is not int for value in frame_shape)
    ):
        raise ValueError("raw frame shape is invalid")
    height, width, channels = frame_shape
    frame_nbytes = frame_shared.get("nbytes")
    if (
        height <= 0
        or width <= 0
        or height > MAX_FRAME_HEIGHT
        or width > MAX_FRAME_WIDTH
        or channels != 3
        or frame_shared.get("dtype") != "uint8"
        or type(frame_nbytes) is not int
        or type(frame_shared.get("generation")) is not int
        or type(frame_shared.get("processGeneration")) is not int
        or (
            expected_process_generation is not None
            and frame_shared.get("processGeneration") != expected_process_generation
        )
        or (
            expected_request_generation is not None
            and frame_shared.get("generation") != expected_request_generation
        )
        or frame_nbytes != height * width * channels
        or frame_nbytes > MAX_FRAME_RAW_BYTES
    ):
        raise ValueError("raw frame metadata exceeds render cap")
    name = frame_shared.get("name")
    if not isinstance(name, str) or not _RENDER_SHARED_NAME.fullmatch(name):
        raise ValueError("raw frame shared memory name is invalid")
    if not isinstance(garment_png, bytes) or len(garment_png) > MAX_GARMENT_BYTES:
        raise GarmentFetchError("byte_size")
    garment_scale = payload["garmentScale"]
    if (
        not isinstance(garment_scale, (int, float))
        or isinstance(garment_scale, bool)
        or not math.isfinite(float(garment_scale))
    ):
        raise ValueError("garment scale is invalid")
    garment_scale = float(garment_scale)
    if garment_scale < 0.8 or garment_scale > 1.6:
        raise ValueError("garment scale exceeds bounds")
    digest = "sha256:" + hashlib.sha256(garment_png).hexdigest()
    if digest != payload["garmentDigest"]:
        raise GarmentFetchError("digest")
    shm = shared_memory.SharedMemory(name=name)
    try:
        frame = np.ndarray((height, width, channels), dtype=np.uint8, buffer=shm.buf).copy()
    finally:
        shm.close()
    source = ValidatedGarmentSource(
        png_bytes=garment_png,
        digest=digest,
        template=payload["template"],
    )
    runtime = _FAST_RUNTIME or _initialize_runtime()
    if not _POSE_READY:
        from vision.fast_tryon import PoseUnavailableError

        raise PoseUnavailableError("pose_unavailable")
    result = runtime.render(frame, source, garment_scale=garment_scale)
    if len(result) > MAX_RESULT_BYTES:
        raise RuntimeError("fast result exceeds render cap")
    return result


def render_worker_entry(connection: Connection) -> None:
    """Own the render loop and announce readiness separately from requests."""
    try:
        _initialize_runtime()
        connection.send(("ready", {"pid": os.getpid(), "poseReady": _POSE_READY}))
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command != "render":
                connection.send(("error", f"unknown render command: {command}"))
                continue
            try:
                connection.send(
                    (
                        "ok",
                        _render(
                            payload,
                            expected_process_generation=connection.expected_process_generation,
                            expected_request_generation=connection.current_request_generation,
                        ),
                    )
                )
            except BaseException as exc:
                from vision.fast_tryon import GarmentFetchError, PoseUnavailableError

                kind = (
                    "garment_error"
                    if isinstance(exc, GarmentFetchError)
                    else "pose_error"
                    if isinstance(exc, PoseUnavailableError)
                    else "error"
                )
                connection.send((kind, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        return
    finally:
        connection.close()
