"""Spawn-safe target for the lifecycle-owned Fast render process.

The target deliberately has no dependency on the FastAPI application or model
runtime.  PyInstaller and Windows ``spawn`` can import it as a plain module.
"""

from __future__ import annotations

import hashlib
import os
from multiprocessing.connection import Connection


MAX_GARMENT_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 12 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024


def _render(payload: dict) -> bytes:
    """Decode, prepare and render entirely inside the bounded child."""
    import cv2
    import numpy as np

    from vision.fast_tryon import (
        FastTryOnRuntime,
        GarmentFetchError,
        ValidatedGarmentSource,
    )

    if not isinstance(payload, dict) or set(payload) != {
        "framePng",
        "garmentPng",
        "garmentDigest",
        "template",
    }:
        raise ValueError("invalid render payload")
    frame_png = payload["framePng"]
    garment_png = payload["garmentPng"]
    if not isinstance(frame_png, bytes) or len(frame_png) > MAX_FRAME_BYTES:
        raise ValueError("frame PNG exceeds render cap")
    if not isinstance(garment_png, bytes) or len(garment_png) > MAX_GARMENT_BYTES:
        raise GarmentFetchError("byte_size")
    digest = "sha256:" + hashlib.sha256(garment_png).hexdigest()
    if digest != payload["garmentDigest"]:
        raise GarmentFetchError("digest")
    frame = cv2.imdecode(np.frombuffer(frame_png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame PNG decode failed")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0 or height > 1080 or width > 1920:
        raise ValueError("frame dimensions exceed render cap")
    source = ValidatedGarmentSource(
        png_bytes=garment_png,
        digest=digest,
        template=payload["template"],
    )
    result = FastTryOnRuntime(max_garment_bytes=MAX_GARMENT_BYTES).render(frame, source)
    if len(result) > MAX_RESULT_BYTES:
        raise RuntimeError("fast result exceeds render cap")
    return result


def render_worker_entry(connection: Connection) -> None:
    """Own the render loop and announce readiness separately from requests."""
    try:
        connection.send(("ready", {"pid": os.getpid()}))
        while True:
            command, payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command != "render":
                connection.send(("error", f"unknown render command: {command}"))
                continue
            try:
                connection.send(("ok", _render(payload)))
            except BaseException as exc:
                from vision.fast_tryon import GarmentFetchError

                kind = (
                    "garment_error" if isinstance(exc, GarmentFetchError) else "error"
                )
                connection.send((kind, f"{type(exc).__name__}: {exc}"))
    except EOFError:
        return
    finally:
        connection.close()
