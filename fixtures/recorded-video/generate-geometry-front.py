"""Generate deterministic far/mid/near acquisition recordings from one fixture source.

The clips share a fixed person pose.  Only four background-corner grayscale
markers change, which proves decoded preview liveness without changing the
detector or acquisition outcome.  This generator is fixture-only and is never
included in the Windows runtime archive.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import FPS, FRONT_FRAME_SIZE, PERSON_SOURCE

ROOT = Path(__file__).parent
SOURCE = PERSON_SOURCE
FRAME_SIZE = FRONT_FRAME_SIZE
FRAME_COUNT = 36
SCALES = {"geometry-far.mp4": 0.62, "geometry-mid.mp4": 0.86, "geometry-near.mp4": 1.05}


def _center_crop_or_pad(image: np.ndarray) -> np.ndarray:
    width, height = FRAME_SIZE
    source_height, source_width = image.shape[:2]
    canvas = np.full((height, width, 3), 118, dtype=np.uint8)
    source_left = max(0, (source_width - width) // 2)
    source_top = max(0, (source_height - height) // 2)
    target_left = max(0, (width - source_width) // 2)
    target_top = max(0, (height - source_height) // 2)
    copy_width = min(width, source_width)
    copy_height = min(height, source_height)
    canvas[target_top : target_top + copy_height, target_left : target_left + copy_width] = image[
        source_top : source_top + copy_height, source_left : source_left + copy_width
    ]
    return canvas


def _frame_with_background_marker(base: np.ndarray, index: int) -> np.ndarray:
    frame = base.copy()
    shade = 25 + (index * 19) % 180
    for left, top in ((8, 8), (FRAME_SIZE[0] - 24, 8), (8, FRAME_SIZE[1] - 24), (FRAME_SIZE[0] - 24, FRAME_SIZE[1] - 24)):
        frame[top : top + 16, left : left + 16] = shade
    return frame


def _write(output: Path, base: np.ndarray) -> None:
    writer = cv2.VideoWriter(
        str(output), getattr(cv2, "VideoWriter_fourcc")(*"mp4v"), FPS, FRAME_SIZE
    )
    if not writer.isOpened():
        raise SystemExit("could not open recorded-video writer")
    try:
        for index in range(FRAME_COUNT):
            writer.write(_frame_with_background_marker(base, index))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    try:
        decoded = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded != FRAME_COUNT:
        raise SystemExit(f"generated recording cannot be fully decoded: {output}")


def main() -> None:
    image = cv2.imread(str(SOURCE))
    if image is None:
        raise SystemExit(f"missing source image: {SOURCE}")
    for filename, scale in SCALES.items():
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output = ROOT / filename
        _write(output, _center_crop_or_pad(scaled))
        print(f"{output.name} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
