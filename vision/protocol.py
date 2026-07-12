"""
WebSocket 协议消息构建模块

提供标准化的消息封包（envelope）格式和错误消息构建。
所有 WebSocket 消息都遵循统一的结构：

{
    "protocol": "vem.vision.v1",
    "type": "vision.xxx",
    "messageId": "<uuid>",
    "timestamp": "<ISO 8601 UTC>",
    "payload": { ... }
}
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from vision.config import settings


# 协议版本标识
PROTOCOL = settings.PROTOCOL
# 应用版本号
APP_VERSION = settings.APP_VERSION


def now_iso() -> str:
    """生成当前 UTC 时间的 ISO 8601 格式字符串（带 Z 后缀）。"""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def envelope(message_type: str, message_id: str | None = None, payload: dict | None = None):
    """构建标准协议消息封包。

    Args:
        message_type: 消息类型，如 "vision.profile_result"
        message_id: 消息唯一 ID，不传则自动生成 UUID
        payload: 消息负载数据

    Returns:
        标准格式的消息字典
    """
    return {
        "protocol": PROTOCOL,
        "type": message_type,
        "messageId": message_id or str(uuid4()),
        "timestamp": now_iso(),
        "payload": payload or {}
    }


def error_envelope(
    code: str,
    message: str,
    session_id: str | None = None,
    retryable: bool = True,
    detail: dict | None = None,
    message_id: str | None = None
):
    """构建错误消息封包。

    Args:
        code: 错误码，如 "camera_unavailable"
        message: 人类可读的错误描述
        session_id: 关联的视觉会话 ID（可选）
        retryable: 是否为可重试错误
        detail: 额外的错误详情
        message_id: 消息唯一 ID

    Returns:
        标准格式的错误消息字典
    """
    payload = {
        "code": code,
        "message": message,
        "retryable": retryable,
    }

    if session_id is not None:
        payload["sessionId"] = session_id

    if detail:
        payload["detail"] = detail

    return envelope(
        message_type="vision.error",
        message_id=message_id,
        payload=payload
    )
