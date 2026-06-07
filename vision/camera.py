import cv2
import threading
import time
from datetime import datetime

from vision.config import settings
from vision.logger import logger


def get_camera_backend(backend_name: str | None = None):
    name = (backend_name or settings.CAMERA_BACKEND or "dshow").lower()

    mapping = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    return mapping.get(name, cv2.CAP_DSHOW)


def apply_camera_settings(
    cap,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    width = settings.CAMERA_WIDTH if width is None else width
    height = settings.CAMERA_HEIGHT if height is None else height
    fps = settings.CAMERA_FPS if fps is None else fps
    fourcc = settings.CAMERA_FOURCC if fourcc is None else fourcc

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))

    if width and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def open_camera(
    camera_index: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    if camera_index is None:
        camera_index = settings.CAMERA_INDEX

    backend = get_camera_backend(backend_name)
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"camera unavailable, index={camera_index}, backend={backend_name or settings.CAMERA_BACKEND}"
        )

    apply_camera_settings(cap, width=width, height=height, fps=fps, fourcc=fourcc)
    return cap


def read_warmup_frame(cap, warmup_frames: int):
    image = None

    for _ in range(max(1, warmup_frames)):
        ret, frame = cap.read()
        if ret:
            image = frame

    if image is None:
        raise RuntimeError("camera opened but failed to read a valid frame")

    return image


def _time_iso(value):
    if value is None:
        return None

    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _uses_default_camera(
    camera_index: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    return all(
        value is None
        for value in [camera_index, backend_name, width, height, fps, fourcc]
    )


class CameraStream:
    def __init__(self):
        self.lock = threading.RLock()
        self.cap = None
        self.opened_at = None
        self.last_frame_at = None
        self.frame_count = 0
        self.reconnect_count = 0
        self.last_error = None

    def _release_locked(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Failed to release camera: {e}")

        self.cap = None
        self.opened_at = None

    def release(self):
        with self.lock:
            self._release_locked()

    def reset(self):
        with self.lock:
            self._release_locked()
            self.last_error = None

    def _ensure_open_locked(self):
        if self.cap is not None and self.cap.isOpened():
            return

        self._release_locked()
        self.cap = open_camera()
        self.opened_at = time.time()
        self.reconnect_count += 1
        logger.info(
            "Camera stream opened "
            f"index={settings.CAMERA_INDEX}, backend={settings.CAMERA_BACKEND}"
        )

    def read(self, warmup_frames: int | None = None):
        if warmup_frames is None:
            warmup_frames = settings.CAMERA_WARMUP_FRAMES

        attempts = max(1, int(settings.CAMERA_READ_RETRY_COUNT) + 1)
        last_error = None

        for attempt in range(1, attempts + 1):
            try:
                with self.lock:
                    self._ensure_open_locked()
                    image = read_warmup_frame(self.cap, warmup_frames)
                    self.frame_count += 1
                    self.last_frame_at = time.time()
                    self.last_error = None
                    return image

            except Exception as e:
                last_error = e
                with self.lock:
                    self.last_error = str(e)
                    self._release_locked()

                logger.warning(
                    f"Camera read failed attempt={attempt}/{attempts}: {e}"
                )

                if attempt < attempts:
                    time.sleep(settings.CAMERA_RECONNECT_DELAY_MS / 1000.0)

        raise RuntimeError(f"camera read failed after reconnect: {last_error}")

    def status(self):
        with self.lock:
            self._ensure_open_locked()
            image = read_warmup_frame(self.cap, settings.CAMERA_WARMUP_FRAMES)
            capture = describe_capture(self.cap)
            h, w = image.shape[:2]

            return {
                "ok": True,
                "index": settings.CAMERA_INDEX,
                "backend": settings.CAMERA_BACKEND,
                "mode": "persistent",
                "requested": {
                    "width": settings.CAMERA_WIDTH,
                    "height": settings.CAMERA_HEIGHT,
                    "fps": settings.CAMERA_FPS,
                    "fourcc": settings.CAMERA_FOURCC,
                },
                "actual": capture,
                "frame": {
                    "width": w,
                    "height": h,
                    "channels": image.shape[2] if len(image.shape) == 3 else 1,
                },
                "stream": {
                    "keepOpen": settings.CAMERA_KEEP_OPEN,
                    "opened": self.cap is not None and self.cap.isOpened(),
                    "openedAt": _time_iso(self.opened_at),
                    "lastFrameAt": _time_iso(self.last_frame_at),
                    "frameCount": self.frame_count,
                    "reconnectCount": self.reconnect_count,
                    "lastError": self.last_error,
                    "readRetryCount": settings.CAMERA_READ_RETRY_COUNT,
                    "reconnectDelayMs": settings.CAMERA_RECONNECT_DELAY_MS,
                },
            }


_camera_stream = CameraStream()


def get_camera_stream():
    return _camera_stream


def release_camera_stream():
    _camera_stream.release()


def reset_camera_stream():
    _camera_stream.reset()


def capture_image(
    camera_index: int | None = None,
    warmup_frames: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    if warmup_frames is None:
        warmup_frames = settings.CAMERA_WARMUP_FRAMES

    if settings.CAMERA_KEEP_OPEN and _uses_default_camera(
        camera_index=camera_index,
        backend_name=backend_name,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    ):
        return get_camera_stream().read(warmup_frames=warmup_frames)

    cap = open_camera(
        camera_index=camera_index,
        backend_name=backend_name,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )

    try:
        return read_warmup_frame(cap, warmup_frames)
    finally:
        cap.release()


def get_configured_camera_status():
    if settings.CAMERA_KEEP_OPEN:
        return get_camera_stream().status()

    cap = open_camera()

    try:
        image = read_warmup_frame(cap, settings.CAMERA_WARMUP_FRAMES)
        capture = describe_capture(cap)
        h, w = image.shape[:2]

        return {
            "ok": True,
            "index": settings.CAMERA_INDEX,
            "backend": settings.CAMERA_BACKEND,
            "mode": "single_capture",
            "requested": {
                "width": settings.CAMERA_WIDTH,
                "height": settings.CAMERA_HEIGHT,
                "fps": settings.CAMERA_FPS,
                "fourcc": settings.CAMERA_FOURCC,
            },
            "actual": capture,
            "frame": {
                "width": w,
                "height": h,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
            },
        }
    finally:
        cap.release()


def describe_capture(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()

    return {
        "width": width,
        "height": height,
        "fps": round(float(fps), 2),
        "fourcc": fourcc,
    }


def probe_cameras(max_index: int = 8, backend_name: str | None = None):
    results = []

    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, get_camera_backend(backend_name))
        opened = cap.isOpened()
        frame_ok = False
        shape = None
        capture = {}

        if opened:
            ret, frame = cap.read()
            frame_ok = bool(ret)
            capture = describe_capture(cap)

            if ret and frame is not None:
                shape = list(frame.shape)

        cap.release()

        results.append(
            {
                "index": index,
                "opened": opened,
                "frameOk": frame_ok,
                "frameShape": shape,
                "capture": capture,
            }
        )

    return results
