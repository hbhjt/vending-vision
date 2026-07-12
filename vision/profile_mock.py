"""
画像模拟（Mock）模块

用于在没有真实摄像头或模型的情况下进行功能测试。

支持的 Mock 场景（通过 VISION_MOCK_SCENARIO 环境变量配置）：
- off: 正常模式（默认）
- success: 模拟成功检测到单人并推送画像
- no_person: 模拟无人场景
- timeout: 模拟超时场景
- camera_unavailable: 模拟摄像头不可用
"""

from __future__ import annotations

import time
from uuid import uuid4

from vision.config import settings
from vision.profile_mapper import vision_profile_to_protocol
from vision.profile_messages import build_presence_status
from vision.profile_state import protocol_occupancy_snapshot
from vision.protocol import now_iso
from vision.schema import VisionProfile


def mock_profile_event():
    """根据配置的场景生成模拟的画像事件 payload。

    Returns:
        profile_result payload 字典，或 None（无画像可推送时）
    """
    scenario = settings.MOCK_SCENARIO

    if scenario == "success":
        profile = VisionProfile(
            age=None,
            gender="unknown",
            height_cm=172.0,
            shoulder_width_cm=43.0,
            body_type="medium",
            upper_color="dark",
            presence=True,
        )
        protocol_profile = vision_profile_to_protocol(profile)
        return {
            "eventId": f"vision-event-{uuid4()}",
            "detectedAt": now_iso(),
            "occupancy": protocol_occupancy_snapshot(
                {"present": True},
                state_hint="single",
            ),
            "profile": protocol_profile,
            "quality": {
                "overall": "fair",
                "warnings": ["mock scenario enabled: success"],
                "profileUsable": True,
                "sampleCount": 1,
                "validFrameCount": 1,
            },
        }

    if scenario == "no_person":
        return None

    if scenario == "camera_unavailable":
        raise RuntimeError("camera unavailable")

    if scenario == "timeout":
        time.sleep(max(settings.MOCK_PUSH_INTERVAL_MS / 1000.0, 1.0))
        return None

    return None


def mock_ambient_light(enabled: bool):
    """生成模拟的环境光照数据。"""
    if not enabled:
        return None

    return {
        "level": "dim",
        "measuredAt": now_iso(),
        "source": "camera",
        "confidence": 0.5,
        "sample": {"lumaMean": 80.0},
    }


def mock_presence_status(scenario: str, include_ambient_light: bool):
    """根据场景生成模拟的 presence_status 消息。"""
    proximity_by_scenario = {
        "success": {
            "present": True,
            "close": True,
            "closeNow": True,
            "closeTrigger": "close_now",
            "personReady": True,
            "personPresent": True,
            "largestPersonRatio": 0.21,
            "method": "mock",
        },
        "no_person": {
            "present": False,
            "close": False,
            "closeNow": False,
            "closeTrigger": None,
            "personReady": True,
            "personPresent": False,
            "largestPersonRatio": 0.0,
            "method": "mock",
        },
        "timeout": {
            "present": False,
            "close": False,
            "closeNow": False,
            "closeTrigger": None,
            "personReady": True,
            "personPresent": False,
            "largestPersonRatio": 0.0,
            "method": "mock",
        },
    }
    state_by_scenario = {
        "success": "approach",
        "no_person": "empty",
        "timeout": "waiting",
    }
    reason_by_scenario = {
        "success": "mock_person_close_profile_pending",
        "no_person": "mock_no_person",
        "timeout": "mock_timeout_waiting",
    }
    proximity = proximity_by_scenario.get(scenario, {})

    return build_presence_status(
        event_id=f"vision-status-{uuid4()}",
        state=state_by_scenario.get(scenario, "waiting"),
        reason=reason_by_scenario.get(scenario, f"mock_{scenario}"),
        proximity=proximity,
        occupancy=protocol_occupancy_snapshot(proximity),
        ambient_light=mock_ambient_light(include_ambient_light),
    )


def mock_departure_event(include_ambient_light: bool):
    """生成模拟的人物离开事件 payload。"""
    now_monotonic = time.time()
    last_seen_at = now_iso()
    time.sleep(0.001)

    payload = {
        "eventId": f"vision-departure-{uuid4()}",
        "detectedAt": now_iso(),
        "lastSeenAt": last_seen_at,
        "reason": "left_frame",
        "absenceDurationMs": int((time.time() - now_monotonic) * 1000),
    }

    ambient_light = mock_ambient_light(include_ambient_light)
    if ambient_light is not None:
        payload["ambientLight"] = ambient_light

    return payload
