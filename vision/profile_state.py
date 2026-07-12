"""
画像状态管理模块

管理画像采集过程中的各种状态：
- TemporaryProfileTrack: 跨帧临时跟踪目标人物
- ProfileOccupancyGate: 防止同一人物重复推送画像的门控
- PersonDepartureTracker: 检测人物离开事件

同时提供基于 proximity 检测结果的目标签名（signature）计算和匹配功能。
"""

from __future__ import annotations

import math
import time
from collections import deque
from uuid import uuid4

from vision.config import settings
from vision.protocol import now_iso


class TemporaryProfileTrack:
    """跨帧临时人物跟踪器。

    在画像采集窗口中跟踪目标人物的位置变化。
    使用基于边界框中心/面积的签名匹配来关联帧间检测结果。
    同时维护 body_samples 缓冲区用于身体测量数据的跨帧累积。
    """

    def __init__(self, signature):
        self.track_id = f"profile-track-{uuid4()}"       # 唯一跟踪ID
        self.state = "present"                             # 当前状态: present/leaving/pushed
        self.started_at = time.time()                      # 创建时间
        self.updated_at = self.started_at                  # 最近更新时间
        self.missing_count = 0                             # 连续丢失帧数
        self.match_score = 1.0                             # 最近匹配分数
        self.signature = signature                         # 目标签名（位置+面积）
        # 身体测量样本缓冲区（固定长度队列）
        self.body_samples = deque(
            maxlen=max(settings.PROFILE_BODY_BUFFER_MAX_FRAMES, 1)
        )
        # 已广播过的 presence 状态（防止重复广播）
        self.announced_presence_states = set()

    def update(self, signature, state, match_score=1.0):
        """用新的检测结果更新跟踪状态。"""
        self.signature = signature
        self.state = state
        self.match_score = round(float(match_score), 4)
        self.updated_at = time.time()
        self.missing_count = 0

    def mark_missing(self):
        """标记当前帧未检测到目标。"""
        self.state = "leaving"
        self.missing_count += 1
        self.updated_at = time.time()

    def is_lost(self):
        """判断是否已丢失（连续丢失帧数超过阈值）。"""
        return self.missing_count > settings.PROFILE_TRACK_MAX_MISSING_FRAMES

    def prune_body_samples(self):
        """清理超过 TTL 的 body 样本。"""
        now = time.time()
        ttl_seconds = max(settings.PROFILE_BODY_BUFFER_TTL_MS, 0) / 1000.0

        while (
            self.body_samples
            and now - self.body_samples[0]["capturedAt"] > ttl_seconds
        ):
            self.body_samples.popleft()

    def append_body_sample(self, sample):
        """添加一个 body 测量样本到缓冲区。"""
        self.body_samples.append(sample)

    def announce_presence_once(self, state):
        """检查该 presence 状态是否已广播过，避免重复。

        Returns:
            True 如果还未广播过（本次应广播），False 如果已广播过。
        """
        if state in self.announced_presence_states:
            return False

        self.announced_presence_states.add(state)
        return True

    def public_state(self):
        """返回跟踪器的公开状态。"""
        return {
            "trackId": self.track_id,
            "state": self.state,
            "ageMs": int((time.time() - self.started_at) * 1000),
            "missingCount": self.missing_count,
            "matchScore": self.match_score,
            "bodyBufferFrameCount": len(self.body_samples),
            "target": self.signature,
        }


class ProfileOccupancyGate:
    """画像推送门控。

    防止同一人物在短时间内被重复采集画像。
    状态流转: empty -> tracking -> occupied -> empty

    - tracking: 检测到人但尚未推送画像
    - occupied: 已推送画像，阻止再次触发（直到人物离开）
    - empty: 无人物或人物已离开足够久
    """

    def __init__(self):
        self.state = "empty"                # empty -> tracking -> occupied
        self.absent_count = 0               # 连续无人帧计数
        self.last_event_id = None           # 最近推送的事件ID
        self.last_pushed_at = None          # 最近推送时间
        self.updated_at = time.time()

    def can_trigger(self):
        """检查是否允许触发新的画像采集。

        仅在非 occupied 状态或门控禁用时返回 True。
        """
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return True

        return self.state != "occupied"

    def mark_present(self):
        """标记检测到人。从 empty 转为 tracking。"""
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        if self.state == "empty":
            self.state = "tracking"

        self.absent_count = 0
        self.updated_at = time.time()

    def mark_pushed(self, event_id):
        """标记画像已推送。转为 occupied 状态，锁定门控。"""
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        self.state = "occupied"
        self.absent_count = 0
        self.last_event_id = event_id
        self.last_pushed_at = time.time()
        self.updated_at = self.last_pushed_at

    def mark_absent(self):
        """标记未检测到人。累计 absent 计数，达标后重置为 empty。"""
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        self.absent_count += 1
        self.updated_at = time.time()

        if self.absent_count >= settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES:
            self.state = "empty"
            self.absent_count = 0
            self.last_event_id = None
            self.last_pushed_at = None

    def mark_target_changed(self):
        """Unlock a pushed profile after a stable, materially different target."""
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return
        self.state = "tracking"
        self.absent_count = 0
        self.last_event_id = None
        self.last_pushed_at = None
        self.updated_at = time.time()

    def public_state(self):
        """返回门控的公开状态。"""
        age_ms = None

        if self.last_pushed_at is not None:
            age_ms = int((time.time() - self.last_pushed_at) * 1000)

        return {
            "enabled": settings.PROFILE_OCCUPANCY_GATE_ENABLED,
            "state": self.state,
            "canTrigger": self.can_trigger(),
            "absentCount": self.absent_count,
            "resetAbsentFrames": settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES,
            "lastEventId": self.last_event_id,
            "lastPushedAgeMs": age_ms,
        }


class PersonDepartureTracker:
    """人物离开检测器。

    跟踪人物是否已离开摄像头视野。
    当连续 absent 帧数达到阈值时，生成离开事件 payload。
    离开事件只触发一次（departed_announced 标志）。
    """

    def __init__(self):
        self.active = False                  # 是否有活跃的人物跟踪
        self.absent_count = 0                # 连续无人帧计数
        self.last_seen_at = None             # 最后看到人的时间（ISO格式）
        self.last_seen_monotonic = None      # 最后看到人的时间（单调时钟）
        self.departed_announced = False      # 是否已广播离开事件

    def mark_present(self):
        """标记有人存在，重置离开检测状态。"""
        self.active = True
        self.absent_count = 0
        self.last_seen_at = now_iso()
        self.last_seen_monotonic = time.time()
        self.departed_announced = False

    def mark_absent(self, reason="no_person", ambient_light=None):
        """标记无人帧。如果连续 absent 达标，生成离开事件。

        Returns:
            离开事件 payload 字典，或 None（未触发离开事件时）。
        """
        if not self.active or self.departed_announced:
            return None

        self.absent_count += 1

        if self.absent_count < settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES:
            return None

        detected_monotonic = time.time()
        absence_duration_ms = None

        if self.last_seen_monotonic is not None:
            absence_duration_ms = int(
                max(detected_monotonic - self.last_seen_monotonic, 0.0) * 1000
            )

        self.active = False
        self.departed_announced = True

        payload = {
            "eventId": f"vision-departure-{uuid4()}",
            "detectedAt": now_iso(),
            "lastSeenAt": self.last_seen_at,
            "reason": reason,
        }

        if absence_duration_ms is not None:
            payload["absenceDurationMs"] = absence_duration_ms

        if ambient_light is not None:
            payload["ambientLight"] = ambient_light

        return payload


# ---------------------------------------------------------------------------
# 全局状态实例
# ---------------------------------------------------------------------------

_active_track = None                        # 当前活跃的人物跟踪器
_occupancy_gate = ProfileOccupancyGate()    # 画像推送门控
_departure_tracker = PersonDepartureTracker() # 人物离开检测器


def target_signature_from_proximity(proximity):
    """从 proximity 检测结果构建目标签名。

    签名用于跨帧匹配同一人物。优先使用人体检测框，
    其次人脸检测框，最后姿态关键点包围框。

    签名包含: source（来源类型）、centerX、centerY、areaRatio、count
    """
    if not proximity or not proximity.get("present"):
        return None

    candidates = [
        (
            "person",
            proximity.get("personPresent"),
            proximity.get("largestPersonBox"),
            proximity.get("largestPersonRatio", 0.0),
            proximity.get("personCount", 0),
        ),
        (
            "face",
            proximity.get("facePresent"),
            proximity.get("largestFaceBox"),
            proximity.get("largestFaceRatio", 0.0),
            proximity.get("faceCount", 0),
        ),
        (
            "body",
            proximity.get("bodyPresent"),
            proximity.get("bodyBox"),
            proximity.get("bodyBoxRatio", 0.0),
            1 if proximity.get("bodyPresent") else 0,
        ),
    ]

    for source, present, box, area_ratio, count in candidates:
        if present and box:
            return {
                "source": source,
                "centerX": box["centerX"],
                "centerY": box["centerY"],
                "areaRatio": round(float(area_ratio), 5),
                "count": count,
            }

    return {
        "source": "unknown",
        "centerX": 0.5,
        "centerY": 0.5,
        "areaRatio": 0.0,
        "count": 0,
    }


def signature_match_score(previous, current):
    """计算两个目标签名的匹配分数。

    评分维度：
    - 中心距离 (65%): 越近越高
    - 来源一致性 (20%): person/body 之间兼容，face 略有折扣
    - 面积变化 (15%): 变化越小越高
    - 多人惩罚: 多人场景降低分数

    Returns:
        0~1 的匹配分数
    """
    if not previous or not current:
        return 0.0

    dx = float(previous["centerX"]) - float(current["centerX"])
    dy = float(previous["centerY"]) - float(current["centerY"])
    distance = math.sqrt(dx * dx + dy * dy)
    max_shift = max(settings.PROFILE_TRACK_MAX_CENTER_SHIFT, 0.01)
    center_score = max(0.0, 1.0 - distance / max_shift)

    previous_area = max(float(previous.get("areaRatio", 0.0)), 0.0001)
    current_area = max(float(current.get("areaRatio", 0.0)), 0.0001)
    ratio_change = max(previous_area, current_area) / min(previous_area, current_area)
    ratio_score = max(0.0, 1.0 - min(ratio_change - 1.0, 4.0) / 4.0)

    if previous.get("source") == current.get("source"):
        source_score = 1.0
    elif {previous.get("source"), current.get("source")} <= {"person", "body"}:
        source_score = 0.75
    else:
        source_score = 0.6

    crowd_penalty = 0.85 if int(current.get("count") or 0) > 1 else 1.0

    score = (
        center_score * 0.65
        + source_score * 0.2
        + ratio_score * 0.15
    ) * crowd_penalty

    return round(score, 4)


def ensure_active_track(signature, state):
    """确保存在活跃的人物跟踪器，必要时创建新的。

    匹配策略：
    - 如果跟踪未启用，总是创建新跟踪器（向后兼容）
    - 如果已有跟踪器，计算签名匹配分数
    - 匹配分数低于阈值则创建新跟踪器（视为新人物）
    - 否则更新现有跟踪器
    """
    global _active_track

    if not settings.PROFILE_TRACK_ENABLED:
        if _active_track is None:
            _active_track = TemporaryProfileTrack(signature)
        _active_track.update(signature, state, match_score=1.0)
        return _active_track

    if _active_track is None:
        _active_track = TemporaryProfileTrack(signature)
        _active_track.update(signature, state, match_score=1.0)
        return _active_track

    score = signature_match_score(_active_track.signature, signature)

    if score < settings.PROFILE_TRACK_MIN_MATCH_SCORE:
        _active_track = TemporaryProfileTrack(signature)
        _active_track.update(signature, state, match_score=1.0)
        return _active_track

    _active_track.update(signature, state, match_score=score)
    return _active_track


def mark_active_track_missing():
    """标记当前跟踪器丢失一帧，连续丢失过多时清除。"""
    global _active_track

    if _active_track is None:
        return

    _active_track.mark_missing()

    if _active_track.is_lost():
        _active_track = None


def reset_active_track():
    """强制重置当前跟踪器。"""
    global _active_track
    _active_track = None


def get_occupancy_gate():
    """获取全局画像推送门控实例。"""
    return _occupancy_gate


def get_departure_tracker():
    """获取全局人物离开检测器实例。"""
    return _departure_tracker


def protocol_occupancy_snapshot(proximity=None, state_hint: str | None = None):
    """生成协议占用状态快照。

    根据多种信息源综合判断当前占用状态（none/single/multiple/unknown）。

    灵敏模式下，一帧人体、人脸或姿态证据即可确认 single；明确的多人
    证据仍拥有最高优先级。顶部 YOLO 返回 none 时，不得覆盖其他检测器
    已经确认的人员证据。
    """
    proximity = proximity or {}
    top_occupancy = proximity.get("topOccupancy")

    top_state = (
        top_occupancy.get("occupancy")
        if isinstance(top_occupancy, dict)
        else None
    )
    detected_count = max(
        int(proximity.get("personCount") or 0),
        int(proximity.get("faceCount") or 0),
    )
    source = "fallback"

    if state_hint in {"none", "single", "multiple", "unknown"}:
        state = state_hint
        source = "hint"
    elif top_state == "multiple" or detected_count > 1:
        state = "multiple"
        source = "top" if top_state == "multiple" else "fallback"
    elif top_state == "single":
        state = "single"
        source = "top"
    elif proximity.get("present"):
        state = "single"
    elif top_state == "unknown":
        # A missing person model must not be mistaken for an empty scene.
        state = "unknown"
        source = "top"
    else:
        state = "none"
        source = "top" if top_state == "none" else "fallback"

    confidence = 0.5
    if state == "single":
        confidence = 0.82
    elif state == "multiple":
        confidence = 0.78
    elif state == "none":
        confidence = 0.8

    if source == "top" and isinstance(top_occupancy, dict):
        confidence = float(top_occupancy.get("confidence", confidence))
    elif state == "single":
        # Face/pose-only evidence is intentionally actionable but lower
        # confidence than a direct top-person detection.
        confidence = 0.62
    elif state == "multiple":
        confidence = 0.7

    return {
        "state": state,
        "confidence": round(confidence, 2),
    }


class ResponsiveOccupancyFilter:
    """Apply asymmetric debounce to the final multi-detector occupancy.

    Any current person/face/pose evidence is actionable immediately.  Empty is
    confirmed only after the configured number of consecutive evidence-free
    polls, so a one-frame miss does not clear the vending-machine session.
    """

    def __init__(self, absent_min_frames=None):
        self.absent_min_frames = max(
            int(
                settings.TOP_OCCUPANCY_CONFIG.get("present_min_frames", 2)
                if absent_min_frames is None
                else absent_min_frames
            ),
            1,
        )
        self.absent_streak = 0
        self.last_occupied = None

    @staticmethod
    def has_current_evidence(proximity):
        proximity = proximity or {}
        return bool(
            int(proximity.get("rawCount") or 0) > 0
            or int(proximity.get("personCount") or 0) > 0
            or int(proximity.get("faceCount") or 0) > 0
            or proximity.get("facePresent")
            or proximity.get("bodyPresent")
        )

    def update(self, proximity, occupancy=None):
        occupancy = occupancy or protocol_occupancy_snapshot(proximity)
        state = occupancy.get("state")

        if self.has_current_evidence(proximity):
            self.absent_streak = 0
            if state in {"single", "multiple"}:
                self.last_occupied = dict(occupancy)
            return occupancy

        self.absent_streak += 1
        if (
            self.last_occupied is not None
            and self.absent_streak < self.absent_min_frames
        ):
            retained = dict(self.last_occupied)
            retained["confidence"] = min(
                float(retained.get("confidence", 0.55)), 0.55,
            )
            return retained

        if self.absent_streak >= self.absent_min_frames:
            self.last_occupied = None
            return {"state": "none", "confidence": 0.8}

        return occupancy


def normalize_protocol_occupancy(occupancy=None, proximity=None):
    """标准化 occupancy 数据，确保返回有效的协议占用快照。"""
    if isinstance(occupancy, dict):
        state = occupancy.get("state")
        if state in {"none", "single", "multiple", "unknown"}:
            return occupancy

    return protocol_occupancy_snapshot(proximity)
