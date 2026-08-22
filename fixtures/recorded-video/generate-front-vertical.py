"""Regenerate the vertical close-up front-acquisition recording from known inputs.

This fixture deliberately mirrors the physical front-camera field condition:
720x1280 portrait frames whose lower body (hips) falls below the bottom edge
while the shoulders remain fully visible.  It is not part of the Windows
runtime.  Run from the vending-vision repository with the locked environment:

    PYTHONPATH=. python fixtures/recorded-video/generate-front-vertical.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import FPS, FRONT_FRAME_SIZE, PERSON_SOURCE

SOURCE = PERSON_SOURCE
FRAME_SIZE = FRONT_FRAME_SIZE
FRAME_COUNT = 6
SCALE = 1.55


def write_recording(output: Path, frame) -> None:
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
    )
    if not writer.isOpened():
        raise SystemExit("could not open recorded-video writer")
    try:
        for _ in range(FRAME_COUNT):
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    ok, _ = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"generated recording cannot be decoded: {output}")


def main() -> None:
    image = cv2.imread(str(SOURCE))
    if image is None:
        raise SystemExit(f"missing source image: {SOURCE}")
    source_height, source_width = image.shape[:2]
    scaled = cv2.resize(
        image,
        (
            int(source_width * SCALE),
            int(source_height * SCALE),
        ),
        interpolation=cv2.INTER_AREA,
    )
    scaled_height, scaled_width = scaled.shape[:2]
    canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    horizontal_offset = (scaled_width - FRAME_SIZE[0]) // 2
    visible = scaled[
        : FRAME_SIZE[1], horizontal_offset : horizontal_offset + FRAME_SIZE[0]
    ]
    canvas[: visible.shape[0]] = visible
    output = Path(__file__).with_name("front-vertical.mp4")
    write_recording(output, canvas)
    print(f"{output.name} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
