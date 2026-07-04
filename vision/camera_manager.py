import threading
import time
from datetime import datetime

from vision.camera import describe_capture, open_camera, read_warmup_frame
from vision.config import settings
from vision.logger import logger


CAMERA_ROLES = {"top", "front"}


def _time_iso(value):
    if value is None:
        return None

    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _camera_config(role: str) -> dict:
    if role == "top":
        return dict(settings.TOP_CAMERA_CONFIG)

    if role == "front":
        return dict(settings.FRONT_CAMERA_CONFIG)

    raise ValueError(f"unknown camera role: {role}")


def _keep_open(config: dict) -> bool:
    value = config.get("keep_open", False)

    if isinstance(value, bool):
        return value

    return str(value).lower() == "true"


def _open_role_camera(config: dict):
    return open_camera(
        camera_index=int(config.get("index", settings.CAMERA_INDEX)),
        backend_name=config.get("backend", settings.CAMERA_BACKEND),
        width=int(config.get("width", settings.CAMERA_WIDTH) or 0),
        height=int(config.get("height", settings.CAMERA_HEIGHT) or 0),
        fps=int(config.get("fps", settings.CAMERA_FPS) or 0),
        fourcc=config.get("fourcc", settings.CAMERA_FOURCC),
    )


def _requested_config(config: dict) -> dict:
    return {
        "index": int(config.get("index", settings.CAMERA_INDEX)),
        "backend": config.get("backend", settings.CAMERA_BACKEND),
        "width": int(config.get("width", settings.CAMERA_WIDTH) or 0),
        "height": int(config.get("height", settings.CAMERA_HEIGHT) or 0),
        "fps": int(config.get("fps", settings.CAMERA_FPS) or 0),
        "fourcc": config.get("fourcc", settings.CAMERA_FOURCC),
        "keepOpen": _keep_open(config),
        "role": config.get("role"),
    }


def _frame_status(role: str, config: dict, cap, image) -> dict:
    height, width = image.shape[:2]
    return {
        "ok": True,
        "role": role,
        "index": int(config.get("index", settings.CAMERA_INDEX)),
        "backend": config.get("backend", settings.CAMERA_BACKEND),
        "requested": _requested_config(config),
        "actual": describe_capture(cap),
        "frame": {
            "width": width,
            "height": height,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
        },
    }


class ManagedCameraStream:
    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = dict(config)
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
            except Exception as exc:
                logger.warning(f"Failed to release {self.role} camera: {exc}")

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
        self.cap = _open_role_camera(self.config)
        self.opened_at = time.time()
        self.reconnect_count += 1
        logger.info(
            f"{self.role} camera stream opened "
            f"index={self.config.get('index')}, "
            f"backend={self.config.get('backend')}"
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

            except Exception as exc:
                last_error = exc
                with self.lock:
                    self.last_error = str(exc)
                    self._release_locked()

                logger.warning(
                    f"{self.role} camera read failed "
                    f"attempt={attempt}/{attempts}: {exc}"
                )

                if attempt < attempts:
                    time.sleep(settings.CAMERA_RECONNECT_DELAY_MS / 1000.0)

        raise RuntimeError(f"{self.role} camera read failed after reconnect: {last_error}")

    def status(self):
        with self.lock:
            self._ensure_open_locked()
            image = read_warmup_frame(self.cap, settings.CAMERA_WARMUP_FRAMES)
            status = _frame_status(self.role, self.config, self.cap, image)
            status["mode"] = "persistent"
            status["stream"] = {
                "opened": self.cap is not None and self.cap.isOpened(),
                "openedAt": _time_iso(self.opened_at),
                "lastFrameAt": _time_iso(self.last_frame_at),
                "frameCount": self.frame_count,
                "reconnectCount": self.reconnect_count,
                "lastError": self.last_error,
                "readRetryCount": settings.CAMERA_READ_RETRY_COUNT,
                "reconnectDelayMs": settings.CAMERA_RECONNECT_DELAY_MS,
            }
            return status


_streams: dict[str, ManagedCameraStream] = {}
_streams_lock = threading.RLock()


def get_camera_config(role: str) -> dict:
    return _camera_config(role)


def get_camera_stream(role: str) -> ManagedCameraStream:
    config = _camera_config(role)

    with _streams_lock:
        stream = _streams.get(role)
        if stream is None:
            stream = ManagedCameraStream(role, config)
            _streams[role] = stream
        return stream


def read_camera(role: str, warmup_frames: int | None = None):
    config = _camera_config(role)

    if _keep_open(config):
        return get_camera_stream(role).read(warmup_frames=warmup_frames)

    cap = _open_role_camera(config)
    try:
        return read_warmup_frame(
            cap,
            settings.CAMERA_WARMUP_FRAMES if warmup_frames is None else warmup_frames,
        )
    finally:
        cap.release()


def get_camera_status(role: str) -> dict:
    config = _camera_config(role)

    if _keep_open(config):
        return get_camera_stream(role).status()

    cap = _open_role_camera(config)
    try:
        image = read_warmup_frame(cap, settings.CAMERA_WARMUP_FRAMES)
        status = _frame_status(role, config, cap, image)
        status["mode"] = "single_capture"
        return status
    finally:
        cap.release()


def get_all_camera_statuses() -> dict:
    statuses = {}

    for role in sorted(CAMERA_ROLES):
        try:
            statuses[role] = get_camera_status(role)
        except Exception as exc:
            statuses[role] = {
                "ok": False,
                "role": role,
                "error": str(exc),
                "requested": _requested_config(_camera_config(role)),
            }

    return statuses


def release_camera(role: str):
    with _streams_lock:
        stream = _streams.get(role)

    if stream is not None:
        stream.release()


def reset_camera(role: str):
    with _streams_lock:
        stream = _streams.get(role)

    if stream is not None:
        stream.reset()


def release_all_cameras():
    with _streams_lock:
        streams = list(_streams.values())

    for stream in streams:
        stream.release()
