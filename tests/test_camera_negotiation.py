"""DirectShow 分辨率协商守卫：显式请求 1080p 时必须 fail closed。"""

from __future__ import annotations

import cv2
import pytest

import vision.camera as camera_module


class FakeCapture:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, *args, **kwargs) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        return 0.0

    def release(self) -> None:
        self.released = True


def test_open_camera_fails_closed_on_resolution_mismatch(monkeypatch):
    cap = FakeCapture(1280, 720)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: cap,
    )
    with pytest.raises(RuntimeError, match="resolution negotiation failed"):
        camera_module.open_camera(camera_index=0, width=1920, height=1080)
    assert cap.released is True


def test_open_camera_accepts_the_requested_resolution(monkeypatch):
    cap = FakeCapture(1920, 1080)
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: cap,
    )
    opened = camera_module.open_camera(camera_index=0, width=1920, height=1080)
    assert opened is cap
    assert cap.released is False
