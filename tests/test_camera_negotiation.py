"""DirectShow MediaType selection must be exact and fail closed."""

from __future__ import annotations

import cv2
import pytest

import vision.camera as camera_module


def _fourcc_value(value: str) -> int:
    return cv2.VideoWriter_fourcc(*value)


class FakeCapture:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps: float = 30.0,
        fourcc: str | None = "MJPG",
        rejected_property: int | None = None,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._rejected_property = rejected_property
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, _value: float) -> bool:
        return prop != self._rejected_property

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_FOURCC:
            return 0.0 if self._fourcc is None else float(_fourcc_value(self._fourcc))
        return 0.0

    def release(self) -> None:
        self.released = True


class EnumeratingFakeCapture(FakeCapture):
    """Model a backend that selects from several advertised MediaTypes."""

    def __init__(self, modes):
        self._modes = modes
        self._requested = {}
        first = modes[0]
        super().__init__(*first[:2], fps=first[2], fourcc=first[3])

    def set(self, prop: int, value: float) -> bool:
        self._requested[prop] = value
        matches = []
        for width, height, fps, fourcc in self._modes:
            if cv2.CAP_PROP_FRAME_WIDTH in self._requested and width != round(
                self._requested[cv2.CAP_PROP_FRAME_WIDTH]
            ):
                continue
            if cv2.CAP_PROP_FRAME_HEIGHT in self._requested and height != round(
                self._requested[cv2.CAP_PROP_FRAME_HEIGHT]
            ):
                continue
            if cv2.CAP_PROP_FPS in self._requested and abs(
                fps - self._requested[cv2.CAP_PROP_FPS]
            ) > 0.5:
                continue
            if cv2.CAP_PROP_FOURCC in self._requested and _fourcc_value(
                fourcc
            ) != round(self._requested[cv2.CAP_PROP_FOURCC]):
                continue
            matches.append((width, height, fps, fourcc))
        if not matches:
            return False
        self._width, self._height, self._fps, self._fourcc = matches[0]
        return True


def _install_capture(monkeypatch, capture):
    monkeypatch.setattr(
        camera_module.cv2,
        "VideoCapture",
        lambda index, backend: capture,
    )


def test_open_camera_uses_configured_defaults_and_rejects_silent_fallback(monkeypatch):
    cap = FakeCapture(640, 480)
    _install_capture(monkeypatch, cap)

    with pytest.raises(RuntimeError) as error:
        camera_module.open_camera(camera_index=0)

    assert "requested=1920x1080@30 MJPG" in str(error.value)
    assert "actual=640x480@30 MJPG" in str(error.value)
    assert cap.released is True


def test_open_camera_selects_exact_media_type_among_multiple_candidates(monkeypatch):
    cap = EnumeratingFakeCapture(
        [
            (640, 480, 30.0, "YUY2"),
            (1920, 1080, 15.0, "MJPG"),
            (1920, 1080, 30.0, "MJPG"),
        ]
    )
    _install_capture(monkeypatch, cap)

    opened = camera_module.open_camera(
        camera_index=0,
        width=1920,
        height=1080,
        fps=30,
        fourcc="MJPG",
    )

    assert opened is cap
    assert camera_module.describe_capture(opened) == {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "fourcc": "MJPG",
    }
    assert cap.released is False


@pytest.mark.parametrize(
    ("capture", "actual"),
    (
        (FakeCapture(1280, 720), "1280x720@30 MJPG"),
        (FakeCapture(1920, 1080, fps=15.0), "1920x1080@15 MJPG"),
        (FakeCapture(1920, 1080, fourcc="YUY2"), "1920x1080@30 YUY2"),
        (
            FakeCapture(1920, 1080, fps=0.0, fourcc=None),
            "1920x1080@unreported unreported",
        ),
    ),
    ids=("resolution", "fps", "fourcc", "unreported"),
)
def test_open_camera_rejects_unsupported_or_silently_mismatched_media_type(
    monkeypatch, capture, actual
):
    _install_capture(monkeypatch, capture)

    with pytest.raises(RuntimeError) as error:
        camera_module.open_camera(
            camera_index=0,
            width=1920,
            height=1080,
            fps=30,
            fourcc="MJPG",
        )

    assert "camera media type negotiation failed" in str(error.value)
    assert "requested=1920x1080@30 MJPG" in str(error.value)
    assert f"actual={actual}" in str(error.value)
    assert capture.released is True


def test_open_camera_rejects_a_backend_that_refuses_a_requested_property(monkeypatch):
    cap = FakeCapture(
        1920,
        1080,
        rejected_property=cv2.CAP_PROP_FOURCC,
    )
    _install_capture(monkeypatch, cap)

    with pytest.raises(RuntimeError) as error:
        camera_module.open_camera(
            camera_index=0,
            width=1920,
            height=1080,
            fps=30,
            fourcc="MJPG",
        )

    assert "camera media type selection failed" in str(error.value)
    assert "property=fourcc" in str(error.value)
    assert cap.released is True
