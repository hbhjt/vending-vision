"""Vision-owned decoded-frame sources.

Every frame source returns BGR OpenCV frames.  The pipeline above this
boundary does not distinguish physical DirectShow cameras from recordings.
"""

from __future__ import annotations

import hashlib
import json
import os
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
    """Decode one deterministic recording without changing pipeline behavior.

    ``loop=True`` rewinds only after EOF. ``loop=False`` remains exhausted
    until the caller explicitly resets the source.
    """

    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = dict(config)
        self.path = Path(str(self.config.get("video_path") or ""))
        self.loop = bool(self.config.get("loop", False))
        self.capture = None
        self.frame_count = 0
        self.exhausted = False
        self.last_error = None
        self.actual = None
        self._last_frame_index = None
        self._decoded_frame_count = None
        self._last_source_frame = None
        self._fixture_sha256 = None
        self._config_sha256 = None

    @staticmethod
    def _sha256(path: Path | str) -> str | None:
        candidate = Path(path)
        if not candidate.is_file():
            return None
        hasher = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def _managed_config_sha256(self) -> str | None:
        config_path_value = os.getenv("VISION_CONFIG_FILE")
        if not config_path_value:
            return None
        config_path = Path(config_path_value)
        try:
            config_bytes = config_path.read_bytes()
            camera = json.loads(config_bytes)["cameras"][self.role]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(camera, dict):
            return None

        managed_video_path = Path(str(camera.get("video_path") or ""))
        if not managed_video_path.is_absolute():
            managed_video_path = config_path.parent / managed_video_path
        expected = {
            "source": str(camera.get("source", "dshow")).lower(),
            "video_path": str(managed_video_path.resolve()),
            "loop": bool(camera.get("loop", False)),
            "rotate": int(camera.get("rotate", 0)),
            "role": camera.get("role"),
        }
        actual = {
            "source": str(self.config.get("source", "dshow")).lower(),
            "video_path": str(self.path.resolve()),
            "loop": self.loop,
            "rotate": int(self.config.get("rotate", 0)),
            "role": self.config.get("role"),
        }
        if expected != actual:
            return None
        return hashlib.sha256(config_bytes).hexdigest()

    def _ensure_open(self):
        if self.capture is not None and self.capture.isOpened():
            return
        if self.exhausted and not self.loop:
            raise RuntimeError(self.last_error or f"recorded video exhausted: {self.path}")
        if not self.path.is_file():
            self._raise_error(f"recorded video does not exist: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            self._raise_error(f"recorded video cannot be decoded: {self.path}")

        # A container may open while containing no usable frame.  Probe once
        # and reset before handing the capture to the shared decode pipeline.
        ok, _ = capture.read()
        if not ok:
            capture.release()
            self._raise_error(f"recorded video contains no decodable frames: {self.path}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            capture.release()
            self._raise_error(f"recorded video cannot reset after probe: {self.path}")

        self.capture = capture
        self.exhausted = False
        self.last_error = None
        self.actual = self._actual_status()
        self._decoded_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fixture_sha256 = self._sha256(self.path)
        self._config_sha256 = self._managed_config_sha256()

    def _actual_status(self) -> dict:
        return {
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": round(float(self.capture.get(cv2.CAP_PROP_FPS)), 2),
            "frameTotal": int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def _raise_error(self, message: str, *, exhausted: bool = False):
        self.last_error = message
        self.exhausted = exhausted
        raise RuntimeError(message)

    def _rewind(self) -> bool:
        return bool(self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0))

    def _read_once(self):
        self._ensure_open()
        ok, frame = self.capture.read()
        if ok:
            self._last_frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            return frame
        if not self.loop:
            self._raise_error(f"recorded video exhausted: {self.path}", exhausted=True)
        if not self._rewind():
            self._raise_error(f"recorded video cannot rewind after exhaustion: {self.path}")
        ok, frame = self.capture.read()
        if not ok:
            self._raise_error(f"recorded video cannot decode after rewind: {self.path}")
        self.exhausted = False
        self.last_error = None
        self._last_frame_index = 0 if self._decoded_frame_count else 0
        return frame

    def read(self, warmup_frames: int | None = None):
        frame = None
        self._last_source_frame = None
        for _ in range(max(1, int(warmup_frames or 1))):
            frame = self._read_once()
        self.frame_count += 1
        self._last_source_frame = self._build_frame_metadata()
        return rotate_frame(frame, camera_rotation(self.config))

    def _build_frame_metadata(self) -> dict:
        frame_index = self._last_frame_index if self._last_frame_index is not None else 0
        decoded_frame_count = (
            self._decoded_frame_count
            if self._decoded_frame_count is not None
            else int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        )

        if not self._fixture_sha256 or not self._config_sha256:
            return None
        if (
            frame_index is None
            or decoded_frame_count <= 0
            or frame_index < 0
            or frame_index >= decoded_frame_count
        ):
            return None

        return {
            "adapter": "recorded_video",
            "role": self.role,
            "frameIndex": frame_index,
            "decodedFrameCount": decoded_frame_count,
            "synthetic": False,
            "relabeled": False,
            "configSha256": self._config_sha256,
            "fixtureSha256": self._fixture_sha256,
        }

    def last_frame(self) -> dict | None:
        return self._last_source_frame

    def status(self) -> dict:
        if self.capture is None and self.last_error is None and not self.exhausted:
            try:
                self._ensure_open()
            except RuntimeError:
                pass
        ready = bool(
            self.capture is not None
            and self.capture.isOpened()
            and self.last_error is None
            and not self.exhausted
        )
        return {
            "ok": ready,
            "ready": ready,
            "role": self.role,
            "source": "recorded_video",
            "videoPath": str(self.path),
            "loop": self.loop,
            "frameCount": self.frame_count,
            "exhausted": self.exhausted,
            "lastError": self.last_error,
            "actual": self.actual,
        }

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def reset(self) -> None:
        self.release()
        self.exhausted = False
        self.last_error = None
        self.actual = None
