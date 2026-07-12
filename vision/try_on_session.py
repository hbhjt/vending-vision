"""
试衣会话管理模块

管理虚拟试衣（Try-On）会话的完整生命周期：
- 启动会话（获取前置摄像头所有权，优先级高于 vision）
- MJPEG 视频流生成（用于前端实时显示）
- 人物离开追踪（试衣期间人物离开时标记）
- 会话停止与资源释放

Session ID 验证规则：只允许字母、数字和 . _ : - 字符，最长 96 字符。
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from datetime import datetime, timezone

import cv2

from vision.camera_manager import read_camera
from vision.camera_owner import (
    acquire_front_camera,
    front_camera_io_lock,
    release_front_camera,
    renew_front_camera,
)
from vision.config import settings


# Session ID 格式验证：字母数字 + . _ : -，最长96字符
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def _now_iso():
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _preview_url(session_id: str, stream_token: str) -> str:
    """生成 MJPEG 预览流的 URL。"""
    host = str(settings.HOST)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return (
        f"http://{host}:{settings.PORT}/try-on/{session_id}.mjpeg"
        f"?token={stream_token}"
    )


def _validate_session_id(session_id: str | None):
    """验证 Session ID 格式。

    Raises:
        ValueError: Session ID 为空或包含不支持的字符
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("payload.sessionId must be a non-empty string")

    session_id = session_id.strip()
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("payload.sessionId contains unsupported characters")

    return session_id


class TryOnSessionManager:
    """试衣会话管理器。

    管理前置摄像头的试衣流会话。
    - 同时只能有一个活跃试衣会话
    - 新会话会自动替换旧会话
    - 支持人物离开标记（试衣期间人物离开）
    - 线程安全
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.active_session_id = None          # 当前活跃试衣会话ID
        self.sessions = {}                      # 所有会话记录

    def start(
        self,
        session_id: str,
        catalog_key: str | None = None,
        variant_id: str | None = None,
        owner_id: str | None = None,
    ):
        """启动一个新的试衣会话。

        流程：
        1. 验证 Session ID
        2. 获取前置摄像头所有权（tryon_frontend 优先级）
        3. 如果已有活跃会话，自动替换

        Raises:
            ValueError: Session ID 格式无效
            RuntimeError: 前置摄像头被占用
        """
        session_id = _validate_session_id(session_id)
        with self.lock:
            self._prune_locked()
            active = self.sessions.get(self.active_session_id)
            if active is not None and active.get("ownerId") != owner_id:
                raise PermissionError("try_on_session_owned_by_another_client")
            if (
                active is not None
                and active.get("sessionId") == session_id
                and active.get("ownerId") == owner_id
            ):
                active["updatedAt"] = _now_iso()
                active["updatedMonotonic"] = time.monotonic()
                renewed = renew_front_camera(
                    "tryon_frontend",
                    reason=f"try_on_start_idempotent:{session_id}",
                )
                if not renewed.get("ok"):
                    reacquired = acquire_front_camera(
                        "tryon_frontend",
                        reason=f"try_on_start_idempotent:{session_id}",
                    )
                    if not reacquired.get("ok"):
                        raise RuntimeError(
                            reacquired.get("error") or "try_on_unavailable"
                        )
                return dict(active)
        acquired = acquire_front_camera(
            "tryon_frontend",
            reason=f"try_on_start:{session_id}",
        )

        if not acquired.get("ok"):
            raise RuntimeError(acquired.get("error") or "try_on_unavailable")

        with self.lock:
            self._prune_locked()
            # 替换旧会话
            if self.active_session_id and self.active_session_id != session_id:
                previous = self.sessions.get(self.active_session_id)
                if previous is not None:
                    if previous.get("ownerId") != owner_id:
                        raise PermissionError("try_on_session_owned_by_another_client")
                    previous["active"] = False
                    previous["stoppedAt"] = _now_iso()
                    previous["stopReason"] = "session_replaced"

            session = {
                "sessionId": session_id,
                "catalogKey": catalog_key,
                "variantId": variant_id,
                "streamToken": secrets.token_urlsafe(24),
                "streamType": "mjpeg",
                "active": True,
                "departedDuringTryon": False,
                "departureEvent": None,
                "startedAt": _now_iso(),
                "updatedAt": _now_iso(),
                "updatedMonotonic": time.monotonic(),
                "ownerId": owner_id,
                "streamClientCount": 0,
            }
            session["previewUrl"] = _preview_url(session_id, session["streamToken"])
            self.sessions[session_id] = session
            self.active_session_id = session_id
            return dict(session)

    def mark_departed(self, departure_event: dict | None = None):
        """标记当前试衣会话中的人物已离开。"""
        with self.lock:
            if not self.active_session_id:
                return None

            session = self.sessions.get(self.active_session_id)
            if session is None or not session.get("active"):
                return None

            session["departedDuringTryon"] = True
            session["departureEvent"] = departure_event
            session["updatedAt"] = _now_iso()
            session["updatedMonotonic"] = time.monotonic()
            return dict(session)

    def stop(self, session_id: str, reason: str | None = None, owner_id: str | None = None):
        """停止试衣会话。

        释放前置摄像头所有权，更新会话状态。
        返回 stopped 信息，包含 shouldRefreshProfile 标志
        （人物在试衣期间离开时需要重新采集画像）。
        """
        session_id = _validate_session_id(session_id)
        reason = reason or "client_stop"
        release_owner = False
        departed_during_tryon = False

        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError("try_on session does not exist")

            if session.get("ownerId") != owner_id:
                raise PermissionError("try_on_session_owned_by_another_client")

            release_owner = self.active_session_id == session_id and bool(
                session.get("active")
            )
            departed_during_tryon = bool(session.get("departedDuringTryon"))
            session["active"] = False
            session["stoppedAt"] = _now_iso()
            session["updatedAt"] = session["stoppedAt"]
            session["updatedMonotonic"] = time.monotonic()
            session["stopReason"] = reason

            if self.active_session_id == session_id:
                self.active_session_id = None

        # 释放前置摄像头所有权
        if release_owner:
            release_front_camera(
                "tryon_frontend",
                reason=f"try_on_stop:{session_id}:{reason}",
            )

        return {
            "sessionId": session_id,
            "reason": reason,
            "departedDuringTryon": departed_during_tryon,
            "shouldRefreshProfile": departed_during_tryon,
        }

    def is_active(self, session_id: str):
        """检查指定 Session ID 的试衣会话是否活跃。"""
        with self.lock:
            session = self.sessions.get(session_id)
            return bool(session and session.get("active"))

    def can_stream(self, session_id: str, stream_token: str | None):
        with self.lock:
            self._prune_locked()
            session = self.sessions.get(session_id)
            allowed = bool(
                session
                and session.get("active")
                and stream_token
                and secrets.compare_digest(str(session.get("streamToken", "")), str(stream_token))
            )
            if allowed:
                session["updatedAt"] = _now_iso()
                session["updatedMonotonic"] = time.monotonic()
            return allowed

    def begin_stream(self, session_id: str, stream_token: str):
        with self.lock:
            if not self.can_stream(session_id, stream_token):
                raise PermissionError("try_on stream is not authorized")
            session = self.sessions[session_id]
            limit = max(int(settings.TRY_ON_MAX_STREAM_CLIENTS), 1)
            if int(session.get("streamClientCount", 0)) >= limit:
                raise RuntimeError("try_on stream client limit reached")
            session["streamClientCount"] = int(session.get("streamClientCount", 0)) + 1

    def end_stream(self, session_id: str):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is not None:
                session["streamClientCount"] = max(
                    int(session.get("streamClientCount", 0)) - 1,
                    0,
                )

    def _prune_locked(self):
        """Bound inactive history so arbitrary session IDs cannot grow memory."""
        limit = max(int(settings.TRY_ON_SESSION_HISTORY_LIMIT), 1)
        ttl_seconds = max(int(settings.TRY_ON_SESSION_TTL_MS), 0) / 1000.0
        now = time.monotonic()
        if ttl_seconds > 0:
            active = self.sessions.get(self.active_session_id)
            if (
                active is not None
                and now - float(active.get("updatedMonotonic") or now) > ttl_seconds
            ):
                active["active"] = False
                active["stoppedAt"] = _now_iso()
                active["updatedAt"] = active["stoppedAt"]
                active["updatedMonotonic"] = now
                active["stopReason"] = "session_ttl_expired"
                self.active_session_id = None
                release_front_camera(
                    "tryon_frontend",
                    reason=f"try_on_ttl_expired:{active['sessionId']}",
                )
            for session_id, item in list(self.sessions.items()):
                if (
                    not item.get("active")
                    and now - float(item.get("updatedMonotonic") or now) > ttl_seconds
                ):
                    self.sessions.pop(session_id, None)
        inactive = [
            item for item in self.sessions.values()
            if not item.get("active")
        ]
        inactive.sort(key=lambda item: item.get("updatedAt") or "")
        for item in inactive[:-limit]:
            self.sessions.pop(item["sessionId"], None)

    def status(self):
        """获取试衣会话的当前状态。"""
        with self.lock:
            self._prune_locked()
            active = self.sessions.get(self.active_session_id)
            public_active = dict(active) if active else None
            if public_active is not None:
                # Capability and connection ownership are transport secrets,
                # not diagnostics exposed by /version.
                public_active.pop("streamToken", None)
                public_active.pop("ownerId", None)
                public_active.pop("updatedMonotonic", None)
                public_active.pop("previewUrl", None)
            return {
                "activeSessionId": self.active_session_id,
                "activeSession": public_active,
                "sessionCount": len(self.sessions),
            }


# ---------------------------------------------------------------------------
# 全局试衣会话实例及便捷函数
# ---------------------------------------------------------------------------

_try_on_sessions = TryOnSessionManager()


def start_try_on_session(
    session_id: str,
    catalog_key: str | None = None,
    variant_id: str | None = None,
    owner_id: str | None = None,
):
    """启动试衣会话。"""
    return _try_on_sessions.start(
        session_id,
        catalog_key=catalog_key,
        variant_id=variant_id,
        owner_id=owner_id,
    )


def stop_try_on_session(session_id: str, reason: str | None = None, owner_id: str | None = None):
    """停止试衣会话。"""
    return _try_on_sessions.stop(session_id, reason=reason, owner_id=owner_id)


def mark_active_try_on_departed(departure_event: dict | None = None):
    """标记当前试衣会话的人物已离开。"""
    return _try_on_sessions.mark_departed(departure_event=departure_event)


def is_try_on_session_active(session_id: str, stream_token: str | None = None):
    """检查试衣会话是否活跃。"""
    try:
        session_id = _validate_session_id(session_id)
    except ValueError:
        return False

    if stream_token is not None:
        return _try_on_sessions.can_stream(session_id, stream_token)
    return _try_on_sessions.is_active(session_id)


def get_try_on_status():
    """获取试衣状态。"""
    return _try_on_sessions.status()


def iter_try_on_mjpeg(
    session_id: str,
    stream_token: str,
    fps: float = 10.0,
    jpeg_quality: int = 80,
):
    """MJPEG 流生成器。

    持续从前置摄像头读取帧并以 MJPEG 格式 yield。
    当会话变为非活跃时自动停止。

    Args:
        session_id: 试衣会话ID
        fps: 目标帧率（默认 10）
        jpeg_quality: JPEG 压缩质量（0-100，默认 80）

    Yields:
        MJPEG 格式的帧数据（含 multipart 边界）
    """
    session_id = _validate_session_id(session_id)
    _try_on_sessions.begin_stream(session_id, stream_token)
    delay = 1.0 / max(float(fps), 1.0)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while is_try_on_session_active(session_id, stream_token=stream_token):
            renew_front_camera("tryon_frontend", reason=f"try_on_stream:{session_id}")
            with front_camera_io_lock():
                frame = read_camera("front", warmup_frames=1)

            ok, encoded = cv2.imencode(".jpg", frame, encode_params)

            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

            time.sleep(delay)
    finally:
        _try_on_sessions.end_stream(session_id)
