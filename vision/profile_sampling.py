"""
画像采样模块

基于中部（前置）摄像头进行多帧画像采集。
负责：
- 帧质量评分（亮度、清晰度、人体检测、人脸检测、位置评分）
- 按质量排序选择最佳帧批次
- 环境光照估计
- 人脸投票采样（补充更多的年龄/性别推理帧）
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import cv2

from vision.camera_manager import read_camera_with_source
from vision.camera_owner import front_camera_io_lock, get_front_camera_owner
from vision.config import settings
from vision.face_detector import FaceDetector
from vision.metrics import metrics
from vision.person_detector import PersonDetector
from vision.pipeline import infer_image
from vision.profile_aggregation import profile_has_detected_field, sample_weight
from vision.profile_mapper import vision_profile_to_protocol
from vision.protocol import now_iso


class FrontCameraBusy(RuntimeError):
    """前置摄像头被占用的异常。

    当采集过程中前置摄像头所有权发生变化（如被 tryon 抢占）时抛出。
    """
    def __init__(self, owner_status=None, reason="front_camera_busy"):
        super().__init__(reason)
        self.owner_status = owner_status or {}
        self.reason = reason


class ProfileSamplingCancelled(RuntimeError):
    """Raised when top-camera state or try-on preempts profile sampling."""


# 画像采集专用的检测器懒加载单例
_person_detector = None
_face_detector = None


def get_profile_person_detector():
    """获取画像采集专用的人体检测器单例。"""
    global _person_detector

    if _person_detector is None:
        _person_detector = PersonDetector()

    return _person_detector


def get_profile_face_detector():
    """获取画像采集专用的人脸检测器单例。"""
    global _face_detector

    if _face_detector is None:
        _face_detector = FaceDetector()

    return _face_detector


def image_quality(image):
    """计算图像的基础质量指标：亮度和清晰度。

    Args:
        image: BGR 图像

    Returns:
        {"brightness": 平均灰度值, "sharpness": Laplacian 方差}
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


def center_score_for_box(box, image_shape):
    """计算边界框相对于图像中心的位置得分。

    越靠近图像中心得分越高（0~1）。
    """
    height, width = image_shape[:2]
    x, y, box_w, box_h = box
    center_x = (x + box_w / 2.0) / float(width)
    center_y = (y + box_h / 2.0) / float(height)
    distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
    return round(max(0.0, 1.0 - distance / 0.75), 4)


def score_frame_quality(image, profile=None):
    """对一帧图像进行综合质量评分。

    评分维度（满分约 1.0）：
    - 人体检测 (25%+15%+12%+8%): 人体存在 + 置信度 + 面积比 + 中心位置
    - 人脸检测 (18%+8%+6%+3%): 人脸存在 + 置信度 + 面积比 + 正面度
    - 清晰度 (3%): Laplacian 方差归一化
    - 亮度 (2%): 偏离中间亮度的程度

    Returns:
        包含所有质量维度的详细字典
    """
    quality = image_quality(image)
    height, width = image.shape[:2]
    person_detections = (
        [] if profile is not None else get_profile_person_detector().detect(image)
    )
    face_detections = (
        [] if profile is not None else get_profile_face_detector().detect_faces(image)
    )

    # 最佳人体检测结果
    best_person = (
        max(person_detections, key=lambda item: item["score"])
        if person_detections
        else None
    )
    # 最佳人脸检测结果
    best_face = (
        max(face_detections, key=lambda item: item.get("score", 0.0))
        if face_detections
        else None
    )
    person_area_ratio = 0.0
    person_score = 0.0
    person_center_score = 0.0

    if best_person:
        _, _, box_w, box_h = best_person["box"]
        person_area_ratio = (box_w * box_h) / float(width * height)
        person_score = float(best_person.get("score") or 0.0)
        person_center_score = center_score_for_box(best_person["box"], image.shape)

    face_area_ratio = 0.0
    face_score = 0.0
    face_frontal_score = 0.0

    if best_face:
        _, _, box_w, box_h = best_face["bbox"]
        face_area_ratio = (box_w * box_h) / float(width * height)
        face_score = float(best_face.get("score") or 0.0)
        # 有关键点信息时认为更可能是正面
        face_frontal_score = 0.5 if best_face.get("landmarks") is None else 0.8

    # 亮度评分：越靠近 (min+max)/2 中间值越好
    brightness = quality["brightness"]
    brightness_min = settings.PROFILE_SAMPLING_CONFIG.get("brightness_min", 35)
    brightness_max = settings.PROFILE_SAMPLING_CONFIG.get("brightness_max", 230)
    brightness_mid = (brightness_min + brightness_max) / 2.0
    brightness_span = max((brightness_max - brightness_min) / 2.0, 1.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - brightness_mid) / brightness_span)
    # 清晰度评分
    blur_score = min(max(quality["sharpness"], 0.0) / 200.0, 1.0)

    if profile is not None:
        person_detected = bool(profile.presence)
        person_score = 1.0 if person_detected else 0.0
        face_detected = bool(
            profile.age is not None or profile.gender != "unknown"
        )
        face_score = 1.0 if face_detected else 0.0
    else:
        person_detected = bool(
            best_person
            and person_score >= settings.PROFILE_SAMPLING_CONFIG.get("min_person_score", 0.35)
        )
        face_detected = bool(
            best_face
            and face_score >= settings.PROFILE_SAMPLING_CONFIG.get("min_face_score", 0.45)
            and face_area_ratio >= settings.PROFILE_SAMPLING_CONFIG.get("min_face_area_ratio", 0.01)
        )

    # 综合质量评分（各维度加权求和）
    score = (
        (0.25 if person_detected else 0.0)
        + min(person_score, 1.0) * 0.15
        + min(person_area_ratio / 0.25, 1.0) * 0.12
        + person_center_score * 0.08
        + (0.18 if face_detected else 0.0)
        + min(face_score, 1.0) * 0.08
        + min(face_area_ratio / 0.04, 1.0) * 0.06
        + face_frontal_score * 0.03
        + blur_score * 0.03
        + brightness_score * 0.02
    )

    return {
        **quality,
        "qualityScore": round(score, 4),
        "personDetected": person_detected,
        "personScore": round(person_score, 4),
        "personAreaRatio": round(person_area_ratio, 5),
        "personCenterScore": person_center_score,
        "faceDetected": face_detected,
        "faceScore": round(face_score, 4),
        "faceAreaRatio": round(face_area_ratio, 5),
        "faceFrontalScore": round(face_frontal_score, 4),
        "bestPersonBox": best_person["box"] if best_person else None,
        "bestFaceBox": best_face["bbox"] if best_face else None,
    }


def estimate_ambient_light(image):
    """从图像估算环境光照水平。

    Returns:
        {"level": "dark"/"dim"/"bright", "measuredAt": ..., "source": "camera", ...}
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    luma_mean = float(gray.mean())

    if luma_mean < settings.AMBIENT_LIGHT_DARK_LUMA:
        level = "dark"
    elif luma_mean < settings.AMBIENT_LIGHT_DIM_LUMA:
        level = "dim"
    else:
        level = "bright"

    return {
        "level": level,
        "measuredAt": now_iso(),
        "source": "camera",
        "confidence": 0.82,
        "sample": {
            "lumaMean": round(luma_mean, 2),
        },
    }


def resize_for_profile_inference(image):
    """将图像缩放到画像推理所需的分辨率（减少计算量）。"""
    width = settings.PROFILE_DETECTION_WIDTH
    height = settings.PROFILE_DETECTION_HEIGHT

    if not width or not height or width <= 0 or height <= 0:
        return image

    return cv2.resize(image, (width, height))


def sample_frame(
    source,
    index,
    proximity=None,
    track=None,
    cancel_event=None,
    *,
    _io_lock_held=False,
):
    """采集一帧并进行完整推理。

    这是画像采集的核心函数。每帧执行：
    1. 检查前置摄像头所有权（必须是 vision）
    2. 读取前置摄像头图像
    3. 缩放 + 质量评分 + 视觉推理
    4. 人脸不可见时清除年龄/性别字段
    5. 构建包含原始图像和推理图像的完整样本

    Raises:
        FrontCameraBusy: 前置摄像头被其他使用者抢占时
    """
    if cancel_event is not None and cancel_event.is_set():
        raise ProfileSamplingCancelled("profile_sampling_cancelled")

    if not _io_lock_held:
        with front_camera_io_lock():
            return sample_frame(
                source,
                index,
                proximity=proximity,
                track=track,
                cancel_event=cancel_event,
                _io_lock_held=True,
            )

    started = time.time()
    # 检查前置摄像头所有权（快速失败）
    owner_status = get_front_camera_owner()
    if owner_status.get("owner") != "vision":
        metrics.increment("profile_sample_owner_busy_total", owner=owner_status.get("owner"))
        raise FrontCameraBusy(
            owner_status=owner_status,
            reason="front_camera_owner_changed",
        )

    # 二次确认：加锁后重新检查，防止 TOCTOU 竞态
    owner_status = get_front_camera_owner()
    if owner_status.get("owner") != "vision":
        metrics.increment("profile_sample_owner_busy_total", owner=owner_status.get("owner"))
        raise FrontCameraBusy(
            owner_status=owner_status,
            reason="front_camera_owner_changed_after_lock",
        )
    image, source_frame = read_camera_with_source("front", warmup_frames=1)

    inference_image = resize_for_profile_inference(image)
    profile = infer_image(inference_image)
    # The full profile pipeline already performs person, pose, and face
    # inference. Reuse its result instead of running a second YOLO/face pass.
    quality = score_frame_quality(inference_image, profile=profile)

    # 低质量或不可见人脸不强制输出年龄/性别
    if not quality["faceDetected"]:
        profile.age = None
        profile.gender = "unknown"

    protocol_profile = vision_profile_to_protocol(profile)
    confidence = protocol_profile["confidence"]
    has_profile_field = profile_has_detected_field(profile)
    is_valid = bool(
        protocol_profile["personPresent"]
        and has_profile_field
        and confidence >= settings.PROFILE_MIN_CONFIDENCE
    )

    sample = {
        "index": index,
        "source": source,
        "capturedAt": time.time(),
        "sourceFrame": source_frame,
        "profile": profile,
        "protocolProfile": protocol_profile,
        "quality": quality,
        "proximity": proximity,
        "trackId": track.track_id if track else None,
        "valid": is_valid,
        "summary": {
            "index": index,
            "source": source,
            "trackId": track.track_id if track else None,
            "trackState": track.state if track else None,
            "trackMatchScore": track.match_score if track else None,
            "personPresent": protocol_profile["personPresent"],
            "confidence": confidence,
            "brightness": quality["brightness"],
            "sharpness": quality["sharpness"],
            "qualityScore": quality["qualityScore"],
            "inferenceWidth": inference_image.shape[1],
            "inferenceHeight": inference_image.shape[0],
            "valid": is_valid,
            "hasProfileField": has_profile_field,
            "hasBodyMeasure": bool(
                profile.height_cm is not None
                or profile.shoulder_width_cm is not None
                or profile.body_type != "unknown"
            ),
            "hasFaceAttribute": bool(
                profile.age is not None or profile.gender != "unknown"
            ),
        },
        "rawImage": image,               # 原始图像（用于后续可能的重新推理）
        "inferenceImage": inference_image, # 推理用图像
    }

    sample["summary"]["bodyWeight"] = sample_weight(sample, "body")
    sample["summary"]["faceWeight"] = sample_weight(sample, "face")
    metrics.increment("profile_sample_total", source=source, valid=is_valid)
    metrics.observe_ms(
        "profile_sample_duration_ms",
        (time.time() - started) * 1000,
        source=source,
    )

    return sample


def collect_best_profile_samples(
    proximity=None,
    track=None,
    cancel_event=None,
    close_enough=False,
    close_validator=None,
    *,
    _io_lock_held=False,
):
    """按时间窗口采集多帧，并选择质量最高的帧。

    在配置的 duration_sec 时间窗口内按 target_fps 帧率采样，
    然后按质量评分排序，选取前 max_good_frames 帧。

    Returns:
        按质量评分降序排列的最佳帧列表
    """
    config = settings.PROFILE_SAMPLING_CONFIG
    duration_sec = float(config.get("duration_sec", 3.0))
    early_finish_after_sec = max(
        float(config.get("early_finish_after_sec", 1.0)), 0.0,
    )
    min_good_frames = max(int(config.get("min_good_frames", 2)), 1)
    target_fps = max(float(config.get("target_fps", 6)), 1.0)
    configured_target = max(int(settings.FRONT_CAMERA_PROFILE_SAMPLE_COUNT), 1)
    max_good_frames = min(
        max(int(config.get("max_good_frames", 10)), 1),
        configured_target,
    )
    started_at = time.monotonic()
    deadline = time.monotonic() + max(duration_sec, 0.1)
    configured_interval = max(
        int(settings.FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS), 1,
    ) / 1000.0
    interval = max(1.0 / target_fps, configured_interval)
    samples = []
    index = 1

    # One physical-camera sequence owns the lane across every frame and every
    # inter-frame gap. Try-On may preempt only after this sequence releases it.
    sequence_lock = nullcontext() if _io_lock_held else front_camera_io_lock()
    with sequence_lock:
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise ProfileSamplingCancelled("profile_sampling_cancelled")
            sample = sample_frame(
                source="candidate",
                index=index,
                proximity=proximity,
                track=track,
                cancel_event=cancel_event,
                _io_lock_held=True,
            )
            samples.append(sample)
            index += 1
            valid_count = sum(1 for item in samples if item.get("valid"))
            latest_close = bool(close_enough)
            if close_validator is not None:
                latest_close = latest_close or bool(close_validator())
            if valid_count >= min_good_frames and (
                latest_close
                or time.monotonic() - started_at >= early_finish_after_sec
            ):
                break
            time.sleep(interval)

    # 按质量评分排序，选最优帧
    samples.sort(
        key=lambda item: item["quality"].get("qualityScore", 0.0),
        reverse=True,
    )
    selected = samples[:max_good_frames]

    for index, sample in enumerate(selected, start=1):
        sample["summary"]["index"] = index
        sample["summary"]["selectedRank"] = index
        sample["source"] = "best_frame"
        sample["summary"]["source"] = "best_frame"

    return selected


def to_public_sample(sample):
    """将内部样本转为公开摘要格式（隐藏原始图像等敏感数据）。"""
    return dict(sample["summary"])


def is_face_vote_candidate(sample):
    """判断样本是否适合用于年龄/性别的面部投票。"""
    summary = sample.get("summary") or {}
    quality = sample.get("quality") or {}

    return bool(
        summary.get("hasFaceAttribute")
        and sample.get("valid")
        and quality.get("sharpness", 0.0)
        >= settings.PROFILE_FACE_VOTE_MIN_SHARPNESS
    )


def collect_face_vote_samples(
    samples, proximity, track, cancel_event=None, *, _io_lock_held=False
):
    """补充采集更多的面部投票帧。

    年龄/性别识别可能需要多帧投票以提高置信度。
    在已有帧数不足时，按配置间隔继续采样直到达到目标数量。
    """
    if not settings.PROFILE_FACE_VOTE_ENABLED:
        return

    target_count = max(settings.PROFILE_FACE_VOTE_SAMPLE_COUNT, 0)
    if target_count <= 0:
        return

    qualified_count = len(
        [sample for sample in samples if is_face_vote_candidate(sample)]
    )

    sequence_lock = nullcontext() if _io_lock_held else front_camera_io_lock()
    with sequence_lock:
        while qualified_count < target_count:
            if cancel_event is not None and cancel_event.is_set():
                raise ProfileSamplingCancelled("profile_sampling_cancelled")
            if settings.PROFILE_FACE_VOTE_INTERVAL_MS > 0:
                time.sleep(settings.PROFILE_FACE_VOTE_INTERVAL_MS / 1000.0)

            sample = sample_frame(
                source="face_vote",
                index=len(samples) + 1,
                proximity=proximity,
                track=track,
                cancel_event=cancel_event,
                _io_lock_held=True,
            )
            samples.append(sample)

            if is_face_vote_candidate(sample):
                qualified_count += 1

            if (
                len([item for item in samples if item.get("source") == "face_vote"])
                >= target_count
            ):
                break
