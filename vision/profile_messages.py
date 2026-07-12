"""
画像消息构建模块

提供 WebSocket 协议消息的标准化构建函数：
- build_presence_status: 构建 presence_status 消息 payload
- profile_update: 封装消息类型和 payload 为统一格式
"""

from __future__ import annotations

from vision.profile_state import (
    normalize_protocol_occupancy,
    protocol_occupancy_snapshot,
)
from vision.protocol import now_iso


def build_presence_status(
    event_id,
    state,
    reason,
    proximity=None,
    tracking=None,
    occupancy=None,
    sample=None,
    detail=None,
    ambient_light=None,
):
    """构建 vision.presence_status 消息的 payload。

    聚合 proximity 检测结果、occupancy 状态、环境光照等信息，
    生成标准化的 presence_status 消息体。

    Args:
        event_id: 事件唯一ID
        state: 状态 (approach/waiting/occupied/unusable/empty)
        reason: 状态原因描述
        proximity: 靠近检测结果
        tracking: 人物跟踪状态
        occupancy: 占用状态
        sample: 样本信息（可选）
        detail: 额外详情（可选）
        ambient_light: 环境光照信息（可选）

    Returns:
        标准化的 presence_status payload 字典
    """
    proximity = proximity or {}
    payload = {
        "eventId": event_id,
        "detectedAt": now_iso(),
        "state": state,
        "reason": reason,
        "personPresent": bool(proximity.get("present")),
        "closeNow": bool(proximity.get("closeNow")),
        "close": bool(proximity.get("close")),
        "closeTrigger": proximity.get("closeTrigger"),
        "proximity": proximity,
    }

    payload["occupancy"] = (
        normalize_protocol_occupancy(occupancy, proximity)
        if occupancy is not None
        else protocol_occupancy_snapshot(proximity)
    )

    if ambient_light is not None:
        payload["ambientLight"] = ambient_light

    return payload


def profile_update(message_type, payload):
    """封装消息类型和 payload 为统一的返回格式。

    Args:
        message_type: 消息类型字符串（如 "vision.profile_result"）
        payload: 消息负载字典

    Returns:
        {"message_type": ..., "payload": ...}
    """
    return {
        "message_type": message_type,
        "payload": dict(payload),
    }
