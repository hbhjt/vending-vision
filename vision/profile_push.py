"""
画像推送编排模块

画像采集的顶层状态机编排器，是连接所有子系统的核心枢纽。

主要功能：
1. collect_profile_update() — 主循环入口，每 ~300ms 调用一次
2. collect_front_profile_update() — 前置摄像头画像采集的完整流程
3. Mock 模式支持 — 无需真实摄像头即可测试

工作流程概述：
1. 从顶部摄像头检测靠近状态 (proximity)
2. 判断 occupancy 状态 (none/single/multiple/unknown)
3. 单人时：获取前置摄像头所有权 -> 多帧采样 -> 聚合 -> 推送画像
4. 多人时：仅广播 presence_status
5. 无人时：检测离开事件 -> 广播 person_departed
"""

from __future__ import annotations

import time
from uuid import uuid4

from vision.camera_manager import get_last_frame_source
from vision.camera_owner import acquire_front_camera, release_front_camera
from vision.config import settings
from vision.metrics import metrics
from vision.profile_aggregation import aggregate_samples, build_quality
from vision.profile_messages import build_presence_status, profile_update
from vision.profile_mock import (
    mock_departure_event,
    mock_presence_status,
    mock_profile_event,
)
from vision.profile_sampling import (
    FrontCameraBusy,
    ProfileSamplingCancelled,
    collect_best_profile_samples,
    collect_face_vote_samples,
    estimate_ambient_light,
    sample_frame,
    to_public_sample,
)
from vision.profile_state import (
    ensure_active_track,
    get_departure_tracker,
    get_occupancy_gate,
    mark_active_track_missing,
    normalize_protocol_occupancy,
    protocol_occupancy_snapshot,
    reset_active_track,
    target_signature_from_proximity,
)
from vision.proximity import check_proximity_once_with_image
from vision.protocol import now_iso
from vision.session_state import (
    mark_vision_session_departed,
    mark_vision_session_presence,
    mark_vision_session_profile_pushed,
    mark_vision_session_profiling,
    mark_vision_session_tryon_departed,
    mark_vision_session_unusable,
    mark_vision_session_waiting_front_camera,
)
from vision.try_on_session import get_try_on_status, mark_active_try_on_departed


# Mock 模式下的待推送画像和离开事件（跨轮次延迟推送）
_mock_pending_profile_payload = None
_mock_pending_departure_payload = None


def active_try_on_status():
    """获取当前活跃的试衣会话状态。"""
    status = get_try_on_status()
    return status if status.get("activeSessionId") else None


def wait_for_front_camera_owner(event_id):
    """等待获取前置摄像头所有权。

    在超时时间内轮询尝试获取所有权。
    超时时间由 FRONT_CAMERA_PROFILE_MAX_WAIT_MS 配置（默认 3000ms）。

    Returns:
        acquire 的结果字典（ok=True 表示成功获取）
    """
    deadline = time.time() + max(settings.FRONT_CAMERA_PROFILE_MAX_WAIT_MS, 0) / 1000.0
    result = None

    while True:
        result = acquire_front_camera("vision", reason=f"profile:{event_id}")
        if result.get("ok"):
            return result

        if result.get("error") != "front_camera_busy" or time.time() >= deadline:
            return result

        time.sleep(min(settings.FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS, 100) / 1000.0)


def build_front_camera_waiting_update(
    event_id, reason, proximity=None, tracking=None, occupancy=None,
    ambient_light=None, owner_status=None,
):
    """构建"等待前置摄像头"的 presence_status 更新消息。"""
    return profile_update(
        "vision.presence_status",
        build_presence_status(
            event_id=event_id, state="waiting", reason=reason,
            proximity=proximity, tracking=tracking, occupancy=occupancy,
            ambient_light=ambient_light,
            source="front",
            source_frame=get_last_frame_source("front"),
        ),
    )


def _best_front_profile_source_frame(samples):
    """Select the most representative front profile frame metadata."""
    valid_samples = [
        sample for sample in samples if sample.get("sourceFrame") and sample.get("valid")
    ]
    if not valid_samples:
        valid_samples = [sample for sample in samples if sample.get("sourceFrame")]
    if not valid_samples:
        return None

    return max(
        valid_samples,
        key=lambda sample: float(sample.get("quality", {}).get("qualityScore", 0.0)),
    ).get("sourceFrame")


def collect_front_profile_update(
    event_id, proximity, track, close_enough, ambient_light, include_status,
    cancel_event=None, completion_validator=None, completion_occupancy=None,
    close_validator=None,
):
    """前置摄像头画像采集的完整流程。

    这是采集推送的核心函数，在持有前置摄像头 I/O 锁和所有权后执行。

    流程：
    1. 等待获取前置摄像头所有权
    2. 清理过期的 body 缓冲区样本
    3. 单人出现后立即预采样；近距离可提前结束，未靠近时至少采样一秒
    4. 采样完成后：
       a. 采集多帧最佳样本 (collect_best_profile_samples)
       b. 补充面部投票帧 (collect_face_vote_samples)
       c. 过滤有效样本，检查数量是否达标
       d. 加权聚合 (aggregate_samples)
       e. 检查置信度是否达标
       f. 构建 profile_result payload 并返回
    5. finally: 释放前置摄像头所有权

    Returns:
        profile_update 字典，或 None（不发送消息时）
    """
    started = time.time()
    owner_result = wait_for_front_camera_owner(event_id)

    if not owner_result.get("ok"):
        # 获取失败：可能被 tryon 占用或超时
        reason = owner_result.get("error") or "front_camera_busy"
        mark_vision_session_waiting_front_camera(
            reason=reason, owner_status=owner_result,
        )

        if include_status:
            return build_front_camera_waiting_update(
                event_id=event_id, reason=reason, proximity=proximity,
                tracking=track.public_state(),
                occupancy=get_occupancy_gate().public_state(),
                ambient_light=ambient_light, owner_status=owner_result,
            )

        return None

    try:
        mark_vision_session_profiling(reason="front_profile_sampling")
        # 清理过期的 body 样本
        track.prune_body_samples()
        samples = list(track.body_samples)

        for index, sample in enumerate(samples, start=1):
            sample["summary"]["index"] = index

        # ---- 主采集流程 ----
        # 1. 采集最佳帧批次
        best_samples = collect_best_profile_samples(
            proximity=proximity,
            track=track,
            cancel_event=cancel_event,
            close_enough=close_enough,
            close_validator=close_validator,
        )
        samples.extend(best_samples)

        for sample in best_samples:
            if sample["protocolProfile"]["personPresent"]:
                track.append_body_sample(sample)

        # 2. 补充面部投票帧
        collect_face_vote_samples(
            samples, proximity, track, cancel_event=cancel_event,
        )

        if completion_validator is not None and not completion_validator():
            return None

        # 3. 过滤有效样本
        valid_samples = [sample for sample in samples if sample["valid"]]
        public_samples = [to_public_sample(sample) for sample in samples]
        required_valid_frames = max(
            int(settings.PROFILE_MIN_VALID_FRAMES),
            int(settings.PROFILE_SAMPLING_CONFIG.get("min_good_frames", 2)),
        )

        if len(valid_samples) < required_valid_frames:
            mark_vision_session_unusable(reason="not_enough_valid_frames")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="unusable",
                        reason="not_enough_valid_frames",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        source="front",
                        source_frame=_best_front_profile_source_frame(samples),
                        detail={
                            "validFrameCount": len(valid_samples),
                            "minValidFrames": required_valid_frames,
                        },
                    ),
                )
            return None

        # 4. 聚合样本
        aggregated = aggregate_samples(samples)

        if aggregated is None:
            mark_vision_session_unusable(reason="not_enough_valid_frames")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="unusable",
                        reason="not_enough_valid_frames",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        source="front",
                        source_frame=_best_front_profile_source_frame(samples),
                        detail={
                            "validFrameCount": len(valid_samples),
                            "minValidFrames": required_valid_frames,
                        },
                    ),
                )
            return None

        _, protocol_profile = aggregated
        quality = build_quality(
            protocol_profile, public_samples, len(valid_samples),
            proximity=proximity, min_valid_frames=required_valid_frames,
            sampling_mode="top_presence_front_profile",
        )

        if not quality.get("profileUsable"):
            reason = quality.get("notUsableReason") or "insufficient_quality"
            mark_vision_session_unusable(reason=reason)
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="unusable",
                        reason=reason,
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=normalize_protocol_occupancy(None, proximity),
                        ambient_light=ambient_light,
                        source="front",
                        source_frame=_best_front_profile_source_frame(samples),
                        detail={
                            "validFrameCount": len(valid_samples),
                            "minValidFrames": required_valid_frames,
                        },
                    ),
                )
            return None

        # 5. 置信度检查
        if protocol_profile["confidence"] < settings.PROFILE_MIN_CONFIDENCE:
            mark_vision_session_unusable(reason="confidence_below_threshold")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="unusable",
                        reason="confidence_below_threshold",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        source="front",
                        source_frame=_best_front_profile_source_frame(samples),
                        detail={
                            "confidence": protocol_profile["confidence"],
                            "minConfidence": settings.PROFILE_MIN_CONFIDENCE,
                        },
                    ),
                )
            return None

        # 6. 构建最终的 profile_result payload
        if completion_validator is not None and not completion_validator():
            return None
        final_occupancy = (
            completion_occupancy()
            if completion_occupancy is not None
            else normalize_protocol_occupancy(
                get_occupancy_gate().public_state(), proximity,
            )
        )
        if final_occupancy.get("state") != "single":
            return None

        payload = {
            "eventId": event_id,
            "detectedAt": now_iso(),
            "occupancy": final_occupancy,
            "profile": protocol_profile,
            "quality": quality,
            "source": "front",
        }
        best_profile_frame = _best_front_profile_source_frame(samples)
        if best_profile_frame is not None:
            payload["sourceFrame"] = best_profile_frame

        # 更新状态：track -> pushed, gate -> occupied, reset track
        track.update(track.signature, "pushed", match_score=track.match_score)
        get_occupancy_gate().mark_pushed(event_id)
        reset_active_track()
        mark_vision_session_profile_pushed(payload)
        metrics.increment("profile_result_total", result="pushed")

        return profile_update("vision.profile_result", payload)

    except FrontCameraBusy as exc:
        mark_vision_session_waiting_front_camera(
            reason=exc.reason, owner_status=exc.owner_status,
        )

        if include_status:
            return build_front_camera_waiting_update(
                event_id=event_id, reason=exc.reason, proximity=proximity,
                tracking=track.public_state(),
                occupancy=get_occupancy_gate().public_state(),
                ambient_light=ambient_light, owner_status=exc.owner_status,
            )

        return None

    except ProfileSamplingCancelled:
        return None

    finally:
        metrics.observe_ms(
            "profile_collect_duration_ms",
            (time.time() - started) * 1000,
            result="finished",
        )
        release_front_camera("vision", reason=f"profile_done:{event_id}")


def collect_profile_update(
    include_status: bool = False,
    include_ambient_light: bool = False,
    include_departure: bool = False,
    skip_collection: bool = False,
):
    """主循环入口：执行一轮画像采集检查。

    这是 profile_broadcast_loop 的核心函数，每 ~300ms 调用一次。

    决策逻辑：
    1. Mock 模式：按场景生成模拟数据
    2. 真实模式：
       a. 顶部摄像头检测靠近状态
       b. occupancy=none -> 门控标记 absent、离开检测
       c. occupancy=multiple -> 广播多人状态
       d. occupancy=unknown -> 广播等待状态
       e. occupancy=single + gate open + no tryon -> 画像采集流程

    Args:
        include_status: 是否返回 presence_status 消息（非画像推送状态）
        include_ambient_light: 是否包含环境光照信息
        include_departure: 是否返回离开事件消息
        skip_collection: 跳过画像采集（仅更新 gate 状态 + 检测离开），
                         用于冷却期内保持 gate 状态同步

    Returns:
        profile_update 字典（含 message_type 和 payload），或 None
    """
    global _mock_pending_profile_payload, _mock_pending_departure_payload

    # ---- Mock 模式 ----
    if settings.MOCK_SCENARIO != "off":
        scenario = settings.MOCK_SCENARIO

        # 处理待推送的延迟 mock 画像
        if _mock_pending_profile_payload is not None:
            event_payload = _mock_pending_profile_payload
            _mock_pending_profile_payload = None
            mark_vision_session_profile_pushed(event_payload)
            return profile_update("vision.profile_result", event_payload)

        # 处理待推送的延迟 mock 离开事件
        if _mock_pending_departure_payload is not None:
            pending_departure = _mock_pending_departure_payload
            _mock_pending_departure_payload = None
            event_payload = mock_departure_event(
                pending_departure.get("includeAmbientLight", False)
            )
            mark_vision_session_departed(event_payload)
            return profile_update("vision.person_departed", event_payload)

        if scenario in {"success", "no_person", "timeout"} and include_status:
            if scenario == "success":
                mock_proximity = {"present": True}
                mark_vision_session_presence(
                    "approach_detected", reason="mock_person_close_profile_pending",
                    proximity=mock_proximity,
                    occupancy=protocol_occupancy_snapshot(mock_proximity),
                )
                _mock_pending_profile_payload = mock_profile_event()
                if include_departure:
                    _mock_pending_departure_payload = {
                        "includeAmbientLight": include_ambient_light,
                    }
            else:
                mark_vision_session_departed(
                    {
                        "eventId": f"vision-departure-{uuid4()}",
                        "detectedAt": now_iso(),
                        "reason": f"mock_{scenario}",
                    }
                )
            return profile_update(
                "vision.presence_status",
                mock_presence_status(scenario, include_ambient_light),
            )

        event_payload = mock_profile_event()

        if event_payload is not None:
            mark_vision_session_profile_pushed(event_payload)
            return profile_update("vision.profile_result", event_payload)

        return None

    # ---- 真实模式 ----
    event_id = f"vision-event-{uuid4()}"
    proximity = None

    if settings.PROXIMITY_ENABLED:
        proximity, proximity_image = check_proximity_once_with_image()
        ambient_light = (
            estimate_ambient_light(proximity_image)
            if include_ambient_light
            else None
        )
        signature = target_signature_from_proximity(proximity)
        occupancy_gate = get_occupancy_gate()
        departure_tracker = get_departure_tracker()
        occupancy_snapshot = protocol_occupancy_snapshot(proximity)

        # ----- 无人场景 -----
        if occupancy_snapshot["state"] == "none":
            occupancy_gate.mark_absent()
            mark_active_track_missing()
            departure_payload = departure_tracker.mark_absent(
                reason="no_person", ambient_light=ambient_light,
            )

            try_on_status = active_try_on_status()

            # 试衣期间人物离开
            if departure_payload is not None and try_on_status:
                mark_active_try_on_departed(departure_payload)
                mark_vision_session_tryon_departed(departure_payload)

            # 正常离开
            if departure_payload is not None and not try_on_status:
                mark_vision_session_departed(departure_payload)

            if departure_payload is not None and include_departure:
                departure_payload = dict(departure_payload)
                departure_payload["source"] = "top"
                source_frame = get_last_frame_source("top")
                if source_frame is not None:
                    departure_payload["sourceFrame"] = source_frame
                return profile_update("vision.person_departed", departure_payload)

            return None

        # ----- 有人场景 -----
        occupancy_gate.mark_present()
        departure_tracker.mark_present()
        single_person = occupancy_snapshot["state"] == "single"
        close_enough = single_person and proximity.get("close", False)
        proximity["profileTrigger"] = "single_person_top_occupancy"
        proximity["closeTrigger"] = (
            "single_person_top_occupancy" if single_person else None
        )
        track_state = "single" if single_person else "approach"
        track = ensure_active_track(signature, track_state)

        # ----- 多人场景 -----
        if occupancy_snapshot["state"] == "multiple":
            mark_vision_session_presence(
                "multiple", reason="multiple_people_detected",
                proximity=proximity, occupancy=occupancy_snapshot,
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="occupied",
                        reason="multiple_people_detected",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=occupancy_snapshot, ambient_light=ambient_light,
                        source="top",
                        source_frame=get_last_frame_source("top"),
                    ),
                )

            return None

        # ----- 不确定场景 -----
        if occupancy_snapshot["state"] == "unknown":
            mark_vision_session_presence(
                "waiting_front_camera", reason="top_occupancy_unknown",
                proximity=proximity, occupancy=occupancy_snapshot,
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="waiting",
                        reason="top_occupancy_unknown",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=occupancy_snapshot, ambient_light=ambient_light,
                        source="top",
                        source_frame=get_last_frame_source("top"),
                    ),
                )

            return None

        # ----- 试衣活跃检查 -----
        try_on_status = active_try_on_status()
        if try_on_status is not None:
            mark_vision_session_presence(
                "tryon_active", reason="front_camera_reserved_by_tryon",
                proximity=proximity, occupancy=occupancy_snapshot,
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="waiting",
                        reason="front_camera_reserved_by_tryon",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=occupancy_snapshot, ambient_light=ambient_light,
                        source="top",
                        source_frame=get_last_frame_source("top"),
                    ),
                )

            return None

        # ----- 门控检查 -----
        if not occupancy_gate.can_trigger():
            mark_vision_session_presence(
                "profile_pushed", reason="occupancy_gate_locked",
                proximity=proximity, occupancy=occupancy_gate.public_state(),
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id, state="occupied",
                        reason="occupancy_gate_locked",
                        proximity=proximity, tracking=track.public_state(),
                        occupancy=occupancy_gate.public_state(),
                        source="top",
                        source_frame=get_last_frame_source("top"),
                        ambient_light=ambient_light,
                    ),
                )

            return None

        # ----- 首次单人检测：广播 approach 状态 -----
        if include_status and track.announce_presence_once(track_state):
            mark_vision_session_presence(
                "approach_detected", reason="single_person_profile_pending",
                proximity=proximity, occupancy=occupancy_gate.public_state(),
            )

            return profile_update(
                "vision.presence_status",
                build_presence_status(
                    event_id=event_id, state="approach",
                    reason="single_person_profile_pending",
                    proximity=proximity, tracking=track.public_state(),
                    occupancy=occupancy_gate.public_state(),
                    source="top",
                    source_frame=get_last_frame_source("top"),
                    ambient_light=ambient_light,
                ),
            )

        # ----- 执行画像采集 -----
        if skip_collection:
            return None

        return collect_front_profile_update(
            event_id=event_id, proximity=proximity, track=track,
            close_enough=close_enough, ambient_light=ambient_light,
            include_status=include_status,
        )

    # ---- PROXIMITY 禁用时的后备逻辑 ----
    ambient_light = None
    signature = {
        "source": "disabled", "centerX": 0.5, "centerY": 0.5,
        "areaRatio": 0.0, "count": 1,
    }
    track = ensure_active_track(signature, "close")
    occupancy_gate = get_occupancy_gate()
    occupancy_gate.mark_present()

    if not occupancy_gate.can_trigger():
        if include_status:
            return profile_update(
                "vision.presence_status",
                build_presence_status(
                    event_id=event_id, state="occupied",
                    reason="occupancy_gate_locked",
                    tracking=track.public_state(),
                    occupancy=occupancy_gate.public_state(),
                    source="top",
                    source_frame=get_last_frame_source("top"),
                    ambient_light=ambient_light,
                ),
            )

        return None

    if skip_collection:
        return None

    return collect_front_profile_update(
        event_id=event_id, proximity=proximity, track=track,
        close_enough=True, ambient_light=ambient_light,
        include_status=include_status,
    )


def collect_profile_event():
    """便捷函数：只收集画像事件（不包含状态更新）。

    Returns:
        profile_result payload 字典，或 None
    """
    update = collect_profile_update(include_status=False)

    if update and update["message_type"] == "vision.profile_result":
        return update["payload"]

    return None
