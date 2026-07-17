"""Vision-owned decoded-frame sources.

Every frame source returns BGR OpenCV frames.  The pipeline above this
boundary does not distinguish physical DirectShow cameras from recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2

from vision.frame_transform import camera_rotation, rotate_frame


class FrameSource(Protocol):
    """Supplies decoded frames for one logical Vision camera role."""

    def read(self, warmup_frames: int | None = None): ...

    def status(self) -> dict: ...

    def release(self) -> None: ...

    def reset(self) -> None: ...


class RecordedVideoFrameSource:
    """Decode one deterministic recording without changing pipeline behavior."""

    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = dict(config)
        self.path = Path(str(self.config.get("video_path") or ""))
        self.loop = bool(self.config.get("loop", False))
        self.capture = None
        self.frame_count = 0

    def _ensure_open(self):
        if self.capture is not None and self.capture.isOpened():
            return
        if not self.path.is_file():
            raise RuntimeError(f"recorded video does not exist: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"recorded video cannot be decoded: {self.path}")

    def _read_once(self):
        self._ensure_open()
        ok, frame = self.capture.read()
        if ok:
            return frame
        if not self.loop:
            raise RuntimeError(f"recorded video exhausted: {self.path}")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"recorded video contains no decodable frames: {self.path}")
        return frame

    def read(self, warmup_frames: int | None = None):
        frame = None
        for _ in range(max(1, int(warmup_frames or 1))):
            frame = self._read_once()
        self.frame_count += 1
        return rotate_frame(frame, camera_rotation(self.config))

    def status(self) -> dict:
        self._ensure_open()
        return {
            "ok": True,
            "role": self.role,
            "source": "recorded_video",
            "videoPath": str(self.path),
            "loop": self.loop,
            "frameCount": self.frame_count,
            "actual": {
                "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(float(self.capture.get(cv2.CAP_PROP_FPS)), 2),
                "frameTotal": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            },
        }

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def reset(self) -> None:
        self.release()
