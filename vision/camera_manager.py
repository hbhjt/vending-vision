"""
摄像头流管理模块

提供按角色（top/front）管理的持久化摄像头流。
- 支持自动重连：读取失败时自动重新打开摄像头
- 线程安全：所有操作使用 RLock 保护
- 状态监控：记录帧数、重连次数、最后活跃时间等指标
- 支持图像旋转和 ROI 裁剪变换
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from vision.camera import describe_capture, open_camera, read_warmup_frame
from vision.camera_binding import acquire_runtime_camera_lease, get_camera_maintenance
from vision.config import settings
from vision.frame_source import FrameSource, RecordedVideoFrameSource
from vision.frame_transform import camera_rotation, rotate_frame
from vision.logger import logger
from vision.metrics import metrics


# 支持的摄像头角色
CAMERA_ROLES = {"top", "front"}


def _time_iso(value):
    """将 Unix 时间戳转为 ISO 8601 格式字符串。"""
    if value is None:
        return None

    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _camera_config(role: str) -> dict:
    """根据角色获取摄像头配置字典。"""
    if role == "top":
        config = dict(settings.TOP_CAMERA_CONFIG)
    elif role == "front":
        config = dict(settings.FRONT_CAMERA_CONFIG)
    else:
        raise ValueError(f"unknown camera role: {role}")

    if str(config.get("source", "dshow")).lower() == "recorded_video":
        return config

    candidate = get_camera_maintenance().resolve(role)
    config.update(
        {
            "index": candidate.index,
            "backend": candidate.backend,
            "stableId": candidate.stable_id,
        }
    )
    return config


def _keep_open(config: dict) -> bool:
    """判断摄像头是否应保持常开模式（而非每次读取时重新打开）。"""
    value = config.get("keep_open", False)

    if isinstance(value, bool):
        return value

    return str(value).lower() == "true"


def _open_role_camera(config: dict):
    """根据角色配置打开对应的摄像头。"""
    candidate_id = config.get("stableId")
    role = config.get("role")
    if not isinstance(candidate_id, str) or not isinstance(role, str):
        raise RuntimeError("camera runtime requires a resolved stable role binding")
    lease = acquire_runtime_camera_lease(candidate_id, role)
    try:
        capture = open_camera(
        camera_index=int(config["index"]),
        backend_name=config.get("backend", settings.CAMERA_BACKEND),
        width=int(config.get("width", settings.CAMERA_WIDTH) or 0),
        height=int(config.get("height", settings.CAMERA_HEIGHT) or 0),
        fps=int(config.get("fps", settings.CAMERA_FPS) or 0),
        fourcc=config.get("fourcc", settings.CAMERA_FOURCC),
        )
        return _LeaseBoundCapture(capture, lease)
    except Exception:
        lease.release()
        raise


class _LeaseBoundCapture:
    """Makes runtime acquisition share one ownership lifecycle with maintenance."""
    def __init__(self, capture, lease):
        self._capture, self._lease = capture, lease

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def release(self):
        try:
            self._capture.release()
        finally:
            self._lease.release()


def _requested_config(config: dict) -> dict:
    """构建请求配置的摘要信息，包含旋转、ROI 等变换参数。"""
    return {
        "backend": config.get("backend", settings.CAMERA_BACKEND),
        "width": int(config.get("width", settings.CAMERA_WIDTH) or 0),
        "height": int(config.get("height", settings.CAMERA_HEIGHT) or 0),
        "fps": int(config.get("fps", settings.CAMERA_FPS) or 0),
        "fourcc": config.get("fourcc", settings.CAMERA_FOURCC),
        "keepOpen": _keep_open(config),
        "role": config.get("role"),
        "rotate": config.get("rotate", config.get("rotation")),
        "roi": config.get("roi"),
    }


def _apply_camera_transform(config: dict, image):
    """根据配置对图像应用旋转变换。"""
    return rotate_frame(image, camera_rotation(config))


def _frame_status(role: str, config: dict, cap, image, raw_image=None) -> dict:
    """构建单帧状态报告，包含实际分辨率和变换信息。"""
    height, width = image.shape[:2]
    raw_height, raw_width = raw_image.shape[:2] if raw_image is not None else (height, width)
    return {
        "ok": True,
        "role": role,
        "backend": config.get("backend", settings.CAMERA_BACKEND),
        "requested": _requested_config(config),
        "actual": describe_capture(cap),
        "transform": {
            "rotate": camera_rotation(config),
            "rawWidth": raw_width,
            "rawHeight": raw_height,
        },
        "frame": {
            "width": width,
            "height": height,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
        },
    }


class ManagedCameraStream:
    """持久化摄像头流管理器。

    维护一个长期打开的摄像头连接，支持：
    - 自动重连：read() 失败后自动重新打开摄像头
    - 状态追踪：记录打开时间、帧数、重连次数、最近错误
    - 线程安全：所有操作使用 RLock 保护
    """

    def __init__(self, role: str, config: dict):
        self.role = role                      # 摄像头角色: top / front
        self.config = dict(config)            # 摄像头配置
        self.lock = threading.RLock()         # 线程锁
        self._maintenance_condition = threading.Condition(self.lock)
        self.cap = None                       # OpenCV VideoCapture 对象
        self.opened_at = None                 # 最近一次打开的时间戳
        self.last_frame_at = None             # 最近一次成功读取帧的时间戳
        self.frame_count = 0                  # 累计读取帧数
        self.reconnect_count = 0              # 累计重连次数
        self.last_error = None                # 最近一次错误信息
        self._maintenance_handoffs = 0

    def _release_locked(self):
        """释放摄像头资源（需在持有锁时调用）。"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                logger.warning(f"Failed to release {self.role} camera: {exc}")

        self.cap = None
        self.opened_at = None

    def release(self):
        """线程安全地释放摄像头资源。"""
        with self.lock:
            self._release_locked()

    def reset(self):
        """重置摄像头流：释放资源并清除错误状态。"""
        with self.lock:
            self._release_locked()
            self.last_error = None

    def quiesce_for_maintenance(self):
        """Yield a persistent runtime capture to one protected maintenance action."""
        with self._maintenance_condition:
            self._maintenance_handoffs += 1
            self._release_locked()
        return _RuntimeMaintenanceHandoff(self)

    def _resume_after_maintenance(self):
        with self._maintenance_condition:
            self._maintenance_handoffs = max(0, self._maintenance_handoffs - 1)
            self._maintenance_condition.notify_all()

    def _ensure_open_locked(self):
        """确保摄像头已打开（需在持有锁时调用）。

        如果摄像头未打开或已关闭，则重新打开。
        """
        while self._maintenance_handoffs:
            if not self._maintenance_condition.wait(timeout=10.0):
                raise RuntimeError(f"{self.role} camera maintenance handoff timed out")
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
        """读取一帧图像，支持自动重连。

        读取失败时自动重试（次数由 CAMERA_READ_RETRY_COUNT 配置），
        每次重试之间等待 CAMERA_RECONNECT_DELAY_MS 毫秒。
        所有重试失败后抛出 RuntimeError。
        """
        if warmup_frames is None:
            warmup_frames = settings.CAMERA_WARMUP_FRAMES

        attempts = max(1, int(settings.CAMERA_READ_RETRY_COUNT) + 1)
        last_error = None
        started = time.time()

        for attempt in range(1, attempts + 1):
            try:
                with self.lock:
                    self._ensure_open_locked()
                    image = _apply_camera_transform(
                        self.config,
                        read_warmup_frame(self.cap, warmup_frames),
                    )
                    self.frame_count += 1
                    self.last_frame_at = time.time()
                    self.last_error = None
                    # 更新监控指标
                    metrics.increment("camera_read_success_total", role=self.role)
                    metrics.observe_ms(
                        "camera_read_duration_ms",
                        (time.time() - started) * 1000,
                        role=self.role,
                    )
                    metrics.set_gauge(
                        "camera_frame_count",
                        self.frame_count,
                        role=self.role,
                    )
                    metrics.set_gauge(
                        "camera_reconnect_count",
                        self.reconnect_count,
                        role=self.role,
                    )
                    return image

            except Exception as exc:
                last_error = exc
                with self.lock:
                    self.last_error = str(exc)
                    self._release_locked()
                    # Device/read failure is an explicit re-enumeration trigger;
                    # normal per-frame reads only use the cached generation.
                    get_camera_maintenance().refresh_after_read_failure()
                    self.config = _camera_config(self.role)

                metrics.increment("camera_read_failure_total", role=self.role)
                logger.warning(
                    f"{self.role} camera read failed "
                    f"attempt={attempt}/{attempts}: {exc}"
                )

                if attempt < attempts:
                    time.sleep(settings.CAMERA_RECONNECT_DELAY_MS / 1000.0)

        metrics.observe_ms(
            "camera_read_duration_ms",
            (time.time() - started) * 1000,
            role=self.role,
        )
        raise RuntimeError(f"{self.role} camera read failed after reconnect: {last_error}")

    def status(self):
        """获取摄像头流的详细状态报告。

        包括：打开状态、时间线、帧数、重连次数、错误信息等。
        """
        with self.lock:
            self._ensure_open_locked()
            raw_image = read_warmup_frame(self.cap, settings.CAMERA_WARMUP_FRAMES)
            image = _apply_camera_transform(self.config, raw_image)
            status = _frame_status(self.role, self.config, self.cap, image, raw_image=raw_image)
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


class _RuntimeMaintenanceHandoff:
    def __init__(self, stream: ManagedCameraStream):
        self._stream = stream
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            self._stream._resume_after_maintenance()


class _NoopRuntimeMaintenanceHandoff:
    def release(self):
        return None


# ---------------------------------------------------------------------------
# 全局摄像头流注册表
# ---------------------------------------------------------------------------

# 按角色存储的持久化摄像头流
_streams: dict[str, ManagedCameraStream] = {}
_recorded_sources: dict[str, RecordedVideoFrameSource] = {}
_streams_lock = threading.RLock()
_last_frame_metadata: dict[str, dict] = {}
_role_read_locks = {role: threading.RLock() for role in CAMERA_ROLES}


class DirectShowFrameSource:
    """FrameSource adapter over the existing bound DirectShow stream lifecycle."""

    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = dict(config)

    def read(self, warmup_frames: int | None = None):
        if _keep_open(self.config):
            stream = get_camera_stream(self.role)
            return stream.read(warmup_frames=warmup_frames)
        cap = _open_role_camera(self.config)
        try:
            return _apply_camera_transform(
                self.config,
                read_warmup_frame(
                    cap,
                    settings.CAMERA_WARMUP_FRAMES if warmup_frames is None else warmup_frames,
                ),
            )
        finally:
            cap.release()

    def status(self) -> dict:
        if _keep_open(self.config):
            return get_camera_stream(self.role).status()
        cap = _open_role_camera(self.config)
        try:
            raw_image = read_warmup_frame(cap, settings.CAMERA_WARMUP_FRAMES)
            image = _apply_camera_transform(self.config, raw_image)
            status = _frame_status(self.role, self.config, cap, image, raw_image=raw_image)
            status["mode"] = "single_capture"
            return status
        finally:
            cap.release()

    def release(self) -> None:
        release_camera(self.role)

    def reset(self) -> None:
        reset_camera(self.role)

    def last_frame(self) -> dict | None:
        return None


def get_camera_config(role: str) -> dict:
    """获取指定角色的摄像头配置。"""
    return _camera_config(role)


def get_camera_stream(role: str) -> ManagedCameraStream:
    """获取或创建指定角色的持久化摄像头流（懒初始化）。"""
    config = _camera_config(role)

    with _streams_lock:
        stream = _streams.get(role)
        if stream is None:
            stream = ManagedCameraStream(role, config)
            _streams[role] = stream
        elif stream.config != config:
            stream.release()
            stream = ManagedCameraStream(role, config)
            _streams[role] = stream
        return stream


def get_frame_source(role: str) -> FrameSource:
    """Return the configured Vision-owned frame source for one logical role."""
    config = _camera_config(role)
    source_kind = str(config.get("source", "dshow")).lower()
    if source_kind == "recorded_video":
        with _streams_lock:
            source = _recorded_sources.get(role)
            if source is None or source.config != config:
                if source is not None:
                    source.release()
                source = RecordedVideoFrameSource(role, config)
                _recorded_sources[role] = source
            return source
    if source_kind != "dshow":
        raise ValueError(f"unsupported Vision frame source: {source_kind}")
    return DirectShowFrameSource(role, config)


def get_last_frame_source(role: str):
    """Return cached metadata for the last frame produced for a role."""
    try:
        role_lock = _role_read_locks[role]
    except KeyError as exc:
        raise ValueError(f"unknown camera role: {role}") from exc
    with role_lock:
        source = _last_frame_metadata.get(role)
        return dict(source) if source is not None else None


def quiesce_runtime_camera(candidate_id: str):
    """Temporarily release the persistent stream bound to one stable identity.

    The caller owns the returned handoff until its preview/test capture closes.
    Runtime reads then resume lazily through the normal single owner pipeline.
    """
    with _streams_lock:
        streams = list(_streams.values())
    matching = [
        stream for stream in streams
        if stream.config.get("stableId") == candidate_id
    ]
    if not matching:
        return _NoopRuntimeMaintenanceHandoff()
    if len(matching) != 1:
        raise RuntimeError("camera runtime ownership is ambiguous")
    return matching[0].quiesce_for_maintenance()


def read_camera_with_source(role: str, warmup_frames: int | None = None):
    """Read one image and its source evidence as one role-local operation."""
    try:
        role_lock = _role_read_locks[role]
    except KeyError as exc:
        raise ValueError(f"unknown camera role: {role}") from exc
    with role_lock:
        source = get_frame_source(role)
        image = source.read(warmup_frames=warmup_frames)
        frame = source.last_frame() if hasattr(source, "last_frame") else None
        if frame is not None:
            frame = dict(frame)
            _last_frame_metadata[role] = frame
        else:
            _last_frame_metadata.pop(role, None)
        return image, frame


def read_camera(role: str, warmup_frames: int | None = None):
    """读取指定角色摄像头的一帧。

    如果配置为常开模式，使用持久化流；
    否则每次打开新连接，读取后立即释放。
    """
    image, _ = read_camera_with_source(role, warmup_frames=warmup_frames)
    return image


def get_camera_status(role: str) -> dict:
    """获取指定角色摄像头的状态报告。

    包含请求配置、实际输出参数、变换信息和流状态。
    """
    return get_frame_source(role).status()


def get_all_camera_statuses() -> dict:
    """获取所有摄像头角色的状态报告。

    对每个角色调用 get_camera_status，失败时返回错误信息。
    """
    statuses = {}

    for role in sorted(CAMERA_ROLES):
        try:
            statuses[role] = get_camera_status(role)
        except Exception as exc:
            try:
                requested = _requested_config(_camera_config(role))
            except Exception:
                requested = None
            statuses[role] = {
                "ok": False,
                "role": role,
                "error": str(exc),
                "requested": requested,
            }

    return statuses


def release_camera(role: str):
    """释放指定角色的摄像头资源。"""
    with _role_read_locks[role]:
        with _streams_lock:
            stream = _streams.get(role)
            source = _recorded_sources.get(role)
        if stream is not None:
            stream.release()
        if source is not None:
            source.release()
        _last_frame_metadata.pop(role, None)


def reset_camera(role: str):
    """重置指定角色的摄像头（释放并清除错误状态）。"""
    with _role_read_locks[role]:
        with _streams_lock:
            stream = _streams.get(role)
            source = _recorded_sources.get(role)
        if stream is not None:
            stream.reset()
        if source is not None:
            source.reset()
        _last_frame_metadata.pop(role, None)


def release_all_cameras():
    """释放所有摄像头资源。"""
    for role in CAMERA_ROLES:
        release_camera(role)
