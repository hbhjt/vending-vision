"""Regenerate the vertical close-up unstable fixture from known inputs.

The aligned/unaligned alternating frame sequence keeps the V2 acquisition in
the single-person aligned but unstable state, so the installed Machine UI
manual-capture control remains available and deterministic.  Run from the
vending-vision repository with the locked environment:

    PYTHONPATH=. python fixtures/recorded-video/generate-front-vertical-unstable.py
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


def build_close_up_canvas() -> np.ndarray:
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
    return canvas


def build_unaligned_canvas(close_up: np.ndarray) -> np.ndarray:
    """Shift the close-up so the shoulders leave the centered alignment band."""
    shifted = np.zeros_like(close_up)
    shift = FRAME_SIZE[0] * 300 // 720
    shifted[:, : FRAME_SIZE[0] - shift] = close_up[:, shift:]
    return shifted


def write_recording(output: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, FRAME_SIZE
    )
    if not writer.isOpened():
        raise SystemExit("could not open recorded-video writer")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    ok, _ = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"generated recording cannot be decoded: {output}")


def main() -> None:
    close_up = build_close_up_canvas()
    unaligned = build_unaligned_canvas(close_up)
    # aligned, aligned, unaligned, aligned, aligned, unaligned: the stability
    # counter never reaches three consecutive aligned frames.
    frames = [
        close_up,
        close_up,
        unaligned,
        close_up,
        close_up,
        unaligned,
    ]
    output = Path(__file__).with_name("front-vertical-unstable.mp4")
    write_recording(output, frames)
    print(f"{output.name} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
