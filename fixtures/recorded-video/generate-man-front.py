"""Regenerate the front-acquisition recordings from known fixture inputs.

This fixture is deliberately small and is not part of the Windows runtime.
It contains six identical source frames so recorded-video looping exercises
the production adapter without changing the real detector or pose outcome.
Run from the vending-vision repository with the locked environment:

    PYTHONPATH=. python fixtures/recorded-video/generate-man-front.py
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
    root = Path(__file__).parent
    recordings = {
        "man-front.mp4": cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_AREA),
        # This deterministic left crop remains one YOLO person but moves the
        # torso beyond the permitted centered-pose acquisition geometry.
        "man-unaligned-front.mp4": cv2.resize(image[:, :700], FRAME_SIZE, interpolation=cv2.INTER_AREA),
        "empty-front.mp4": np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8),
    }
    for filename, frame in recordings.items():
        output = root / filename
        write_recording(output, frame)
        print(f"{output.name} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
