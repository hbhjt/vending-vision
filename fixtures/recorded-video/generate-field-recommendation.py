"""Generate near/far recommendation recordings from authorized field captures.

The source JPEGs are unmodified captures from the physical top and front
cameras.  The generator only corrects the front-camera orientation, applies a
center crop to the managed 16:9/9:16 camera aspect, and resizes into the shared
1080p fixture contract.  Run from the repository root with the locked
environment:

    PYTHONPATH=. python fixtures/recorded-video/generate-field-recommendation.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from common import FPS, FRONT_FRAME_SIZE, TOP_FRAME_SIZE


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "sources"
FRAME_COUNT = FPS * 6
DISTANCES = ("near", "far")


def center_crop(image, target_size: tuple[int, int]):
    target_width, target_height = target_size
    source_height, source_width = image.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height
    if source_aspect > target_aspect:
        crop_width = round(source_height * target_aspect)
        left = (source_width - crop_width) // 2
        return image[:, left : left + crop_width]
    crop_height = round(source_width / target_aspect)
    top = (source_height - crop_height) // 2
    return image[top : top + crop_height, :]


def prepare_frame(source: Path, role: str):
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"missing field recommendation source: {source}")
    if role == "front":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_size = FRONT_FRAME_SIZE
    else:
        frame_size = TOP_FRAME_SIZE
    cropped = center_crop(image, frame_size)
    return cv2.resize(cropped, frame_size, interpolation=cv2.INTER_CUBIC)


def write_recording(output: Path, frame, frame_size: tuple[int, int]) -> None:
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, frame_size
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open recorded-video writer: {output}")
    try:
        for _ in range(FRAME_COUNT):
            writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(output))
    ok, decoded = capture.read()
    capture.release()
    if not ok or decoded.shape[:2] != (frame_size[1], frame_size[0]):
        raise SystemExit(f"generated recording cannot be decoded: {output}")


def main() -> None:
    for distance in DISTANCES:
        for role, frame_size in (
            ("top", TOP_FRAME_SIZE),
            ("front", FRONT_FRAME_SIZE),
        ):
            source = SOURCE_ROOT / f"field-recommendation-{distance}-{role}.jpg"
            output = ROOT / f"field-recommendation-{distance}-{role}.mp4"
            write_recording(output, prepare_frame(source, role), frame_size)
            print(
                f"{output.name} sha256="
                f"{hashlib.sha256(output.read_bytes()).hexdigest()}"
            )


if __name__ == "__main__":
    main()
