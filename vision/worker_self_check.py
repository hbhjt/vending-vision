"""Packaged production-worker probe for V2 try-on boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
import time

import cv2
import numpy as np

from vision.acquisition_observer import AcquisitionObservationWorker
from vision.attempt_worker import FastRenderBroker
from vision.fast_tryon import PoseUnavailableError


def _probe_garment_png() -> bytes:
    garment = np.zeros((128, 128, 4), dtype=np.uint8)
    points = np.array(
        [[34, 20], [12, 44], [28, 62], [36, 54], [36, 116], [92, 116],
         [92, 54], [100, 62], [116, 44], [94, 20]],
        dtype=np.int32,
    )
    cv2.fillPoly(garment, [points], (40, 120, 220, 255))
    ok, encoded = cv2.imencode(".png", garment)
    if not ok:
        raise RuntimeError("production worker probe garment encoding failed")
    return encoded.tobytes()


async def _verify_production_workers() -> None:
    context = multiprocessing.get_context("spawn")
    baseline = {child.pid for child in multiprocessing.active_children()}
    observer = AcquisitionObservationWorker(context=context)
    renderer = FastRenderBroker(context=context)
    observation_status = None
    render_status = None
    try:
        observation = await observer.observe(
            np.zeros((96, 128, 3), dtype=np.uint8), timeout=15.0
        )
        if observation.occupancy != "none" or observation.aligned:
            raise RuntimeError(
                f"production acquisition probe returned invalid observation: {observation}"
            )
        if not observation.jpeg.startswith(b"\xff\xd8"):
            raise RuntimeError("production acquisition probe returned invalid JPEG")
        observation_status = observation.occupancy

        await renderer.start()
        garment_png = _probe_garment_png()
        try:
            result = await renderer.render(
                {
                    "frame": np.zeros((192, 256, 3), dtype=np.uint8),
                    "garmentPng": garment_png,
                    "garmentDigest": "sha256:" + hashlib.sha256(garment_png).hexdigest(),
                    "template": "tshirt_short_sleeve",
                },
                deadline=time.monotonic() + 15.0,
            )
        except PoseUnavailableError:
            render_status = "pose_unavailable"
        else:
            decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("production render probe returned an invalid image")
            render_status = "png"
    finally:
        observer_error = renderer_error = None
        try:
            await observer.shutdown()
        except BaseException as exc:
            observer_error = exc
        try:
            await renderer.shutdown()
        except BaseException as exc:
            renderer_error = exc
        if observer_error is not None or renderer_error is not None:
            raise RuntimeError(
                f"production worker probe cleanup failed: observer={observer_error}; "
                f"render={renderer_error}"
            )

    remaining = {
        child.pid for child in multiprocessing.active_children()
        if child.pid not in baseline
    }
    if remaining:
        raise RuntimeError(f"production worker probe leaked children: {sorted(remaining)}")
    print(f"production acquisition observation: {observation_status}")
    print(f"production render response: {render_status}")


def verify_v2_try_on_workers() -> None:
    """Execute real frozen production entries through their shared IPC slots."""
    multiprocessing.freeze_support()
    asyncio.run(_verify_production_workers())
