"""
画像聚合模块

将多帧采样结果通过加权统计方法聚合为单一画像：
- 加权中位数（身高、肩宽等连续值）
- 加权众数（体型、颜色、性别等离散值）
- 质量评估（综合置信度 + 可用性判断）

支持按 body 和 face 两个维度分别加权。
"""

from __future__ import annotations

import statistics
from collections import Counter

from vision.config import settings
from vision.profile_mapper import calculate_confidence, vision_profile_to_protocol
from vision.schema import VisionProfile


def profile_has_detected_field(profile):
    """检查画像中是否包含至少一个有效字段（非 None/unknown）。"""
    return bool(
        profile.height_cm is not None
        or profile.shoulder_width_cm is not None
        or profile.body_type != "unknown"
        or profile.upper_color != "unknown"
        or profile.age is not None
        or profile.gender != "unknown"
    )


def sample_weight(sample, purpose="general"):
    """计算样本的加权权重。

    权重受以下因素影响：
    - 清晰度（sharpness）
    - 亮度适宜度（60~190 之间加分）
    - 协议画像置信度
    - 专用维度加分：
      - body: 身高/肩宽/体型存在、人体检测、body_buffer 来源
      - face: 年龄/性别存在、人脸检测、近距离人脸

    Args:
        sample: 样本字典
        purpose: "general" / "body" / "face"
    """
    quality = sample["quality"]
    profile = sample["profile"]
    protocol_profile = sample["protocolProfile"]
    proximity = sample.get("proximity") or {}

    weight = 1.0
    weight += min(max(quality["sharpness"], 0.0), 300.0) / 300.0

    brightness = quality["brightness"]
    if 60 <= brightness <= 190:
        weight += 0.5

    weight += protocol_profile["confidence"]

    if purpose == "body":
        if profile.height_cm is not None:
            weight += 0.8
        if profile.shoulder_width_cm is not None:
            weight += 0.5
        if profile.body_type != "unknown":
            weight += 0.4
        if proximity.get("bodyPresent") or proximity.get("personPresent"):
            weight += 0.5
        if sample["source"] == "body_buffer":
            weight += 0.4

    if purpose == "face":
        if profile.age is not None:
            weight += 0.4
        if profile.gender != "unknown":
            weight += 0.4
        if proximity.get("facePresent"):
            weight += 0.5
        if proximity.get("largestFaceRatio", 0.0) >= settings.PROXIMITY_CLOSE_FACE_RATIO:
            weight += 0.4
        if sample["source"] == "close_sample":
            weight += 0.4

    return round(weight, 4)


def weighted_mode_or_unknown(items):
    """加权众数：返回加权票数最多的值。

    Args:
        items: [(value, weight), ...] 列表
    """
    scores = Counter()

    for value, weight in items:
        if value in (None, "unknown"):
            continue
        scores[value] += weight

    if not scores:
        return "unknown"

    return scores.most_common(1)[0][0]


def weighted_median_or_none(items):
    """加权中位数：按值排序后，找到累计权重过半的值。

    Args:
        items: [(value, weight), ...] 列表
    """
    items = [(value, weight) for value, weight in items if value is not None]

    if not items:
        return None

    items.sort(key=lambda item: item[0])
    total_weight = sum(weight for _, weight in items)
    midpoint = total_weight / 2.0
    running = 0.0

    for value, weight in items:
        running += weight
        if running >= midpoint:
            return round(float(value), 1)

    return round(float(items[-1][0]), 1)


def mode_or_unknown(values):
    """简单众数（等权重）。过滤 None/unknown 后取最常见的值。"""
    values = [value for value in values if value not in (None, "unknown")]

    if not values:
        return "unknown"

    return Counter(values).most_common(1)[0][0]


def median_or_none(values):
    """简单中位数（等权重）。过滤 None 后取中位数。"""
    values = [value for value in values if value is not None]

    if not values:
        return None

    return round(float(statistics.median(values)), 1)


def age_from_range(age_range: str):
    """将协议年龄范围映射为近似数值。

    child->10, teen->16, adult->30, senior->65
    """
    mapping = {
        "child": 10,
        "teen": 16,
        "adult": 30,
        "senior": 65,
    }
    return mapping.get(age_range)


def aggregate_profiles(profiles):
    """聚合多个 VisionProfile（等权重，用于简单场景）。

    Returns:
        (VisionProfile, protocol_profile_dict) 或 None
    """
    if not profiles:
        return None

    protocol_profiles = [vision_profile_to_protocol(profile) for profile in profiles]

    age_range = mode_or_unknown([item["ageRange"] for item in protocol_profiles])
    body_type_protocol = mode_or_unknown([item["bodyType"] for item in protocol_profiles])

    # 协议体型 -> 内部体型映射
    reverse_body_type = {
        "slim": "thin",
        "regular": "medium",
        "strong": "fat",
        "unknown": "unknown",
    }

    profile = VisionProfile(
        age=age_from_range(age_range),
        gender=mode_or_unknown([profile.gender for profile in profiles]),
        height_cm=median_or_none([profile.height_cm for profile in profiles]),
        shoulder_width_cm=median_or_none(
            [profile.shoulder_width_cm for profile in profiles]
        ),
        body_type=reverse_body_type.get(body_type_protocol, "unknown"),
        upper_color=mode_or_unknown([profile.upper_color for profile in profiles]),
        presence=True,
    )

    confidence_values = [item["confidence"] for item in protocol_profiles]
    confidence = round(
        max(calculate_confidence(profile), statistics.mean(confidence_values)),
        2,
    )

    protocol_profile = vision_profile_to_protocol(profile)
    protocol_profile["confidence"] = min(confidence, 0.95)

    return profile, protocol_profile


def aggregate_samples(samples):
    """聚合多个采样帧为单一画像（加权方式）。

    这是画像推送前的主要聚合函数。
    分 body 和 face 两个维度独立加权聚合。

    流程：
    1. 过滤无效样本（无人、无字段、低置信度）
    2. 分离 body 样本和 face 样本
    3. 加权中位数/众数聚合各项
    4. 合成最终画像和协议输出

    Returns:
        (VisionProfile, protocol_profile_dict) 或 None
    """
    if not samples:
        return None

    # 过滤：保留有人 + 有字段 + 置信度达标的样本
    valid_samples = [
        sample
        for sample in samples
        if sample["protocolProfile"]["personPresent"]
        and profile_has_detected_field(sample["profile"])
        and sample["protocolProfile"]["confidence"] >= settings.PROFILE_MIN_CONFIDENCE
    ]

    if not valid_samples:
        return None

    # 按维度分组
    body_samples = [
        sample
        for sample in valid_samples
        if (
            sample["profile"].height_cm is not None
            or sample["profile"].shoulder_width_cm is not None
            or sample["profile"].body_type != "unknown"
            or sample["profile"].upper_color != "unknown"
        )
    ]
    face_samples = [
        sample
        for sample in valid_samples
        if sample["profile"].age is not None or sample["profile"].gender != "unknown"
    ]

    body_pool = body_samples or valid_samples
    face_pool = face_samples or valid_samples

    # 加权聚合各字段
    height_cm = weighted_median_or_none(
        [
            (sample["profile"].height_cm, sample_weight(sample, "body"))
            for sample in body_pool
        ]
    )
    shoulder_width_cm = weighted_median_or_none(
        [
            (sample["profile"].shoulder_width_cm, sample_weight(sample, "body"))
            for sample in body_pool
        ]
    )
    body_type = weighted_mode_or_unknown(
        [
            (sample["profile"].body_type, sample_weight(sample, "body"))
            for sample in body_pool
        ]
    )
    upper_color = weighted_mode_or_unknown(
        [
            (sample["profile"].upper_color, sample_weight(sample, "body"))
            for sample in body_pool
        ]
    )

    protocol_face_profiles = [
        vision_profile_to_protocol(sample["profile"]) for sample in face_pool
    ]
    age_range = weighted_mode_or_unknown(
        [
            (item["ageRange"], sample_weight(sample, "face"))
            for item, sample in zip(protocol_face_profiles, face_pool)
        ]
    )

    profile = VisionProfile(
        age=age_from_range(age_range),
        gender=weighted_mode_or_unknown(
            [
                (sample["profile"].gender, sample_weight(sample, "face"))
                for sample in face_pool
            ]
        ),
        height_cm=height_cm,
        shoulder_width_cm=shoulder_width_cm,
        body_type=body_type,
        upper_color=upper_color,
        presence=True,
    )

    confidence_values = [sample["protocolProfile"]["confidence"] for sample in valid_samples]
    quality_scores = [
        float(sample.get("quality", {}).get("qualityScore", 0.7))
        for sample in valid_samples
    ]
    # Confidence is deliberately based on consistency and observed frame
    # quality, not simply on how many fields happened to be populated.
    def _agreement(values, value, numeric_tolerance=None):
        usable = [item for item in values if item not in (None, "unknown")]
        if not usable or value in (None, "unknown"):
            return 0.0
        if numeric_tolerance is not None:
            return sum(abs(float(item) - float(value)) <= numeric_tolerance for item in usable) / len(usable)
        return sum(item == value for item in usable) / len(usable)

    agreements = [
        _agreement([item["profile"].height_cm for item in valid_samples], height_cm, 8.0),
        _agreement([item["profile"].shoulder_width_cm for item in valid_samples], shoulder_width_cm, 4.0),
        _agreement([item["profile"].body_type for item in valid_samples], body_type),
        _agreement([item["profile"].upper_color for item in valid_samples], upper_color),
        _agreement([item["profile"].gender for item in valid_samples], profile.gender),
    ]
    nonzero_agreements = [item for item in agreements if item > 0]
    consistency = statistics.mean(nonzero_agreements) if nonzero_agreements else 0.0
    coverage = min(len(valid_samples) / max(settings.PROFILE_MIN_VALID_FRAMES, 1), 1.0)
    confidence = round(
        min(
            0.95,
            statistics.mean(confidence_values) * 0.45
            + statistics.mean(quality_scores) * 0.2
            + consistency * 0.25
            + coverage * 0.1,
        ),
        2,
    )

    protocol_profile = vision_profile_to_protocol(profile)
    protocol_profile["confidence"] = min(confidence, 0.95)

    return profile, protocol_profile


def build_quality(
    protocol_profile,
    samples,
    valid_count,
    proximity=None,
    min_valid_frames=None,
    sampling_mode=None,
):
    """构建画像质量报告。

    评估维度：
    - overall: good (>=0.75) / fair (>=0.45) / poor (<0.45)
    - profileUsable: overall != poor 且有效帧数达标
    - warnings: 各字段不可用的警告列表
    """
    warnings = []
    min_valid_frames = (
        settings.PROFILE_MIN_VALID_FRAMES
        if min_valid_frames is None
        else min_valid_frames
    )

    if valid_count < min_valid_frames:
        warnings.append(f"valid frames {valid_count}/{min_valid_frames}")

    if protocol_profile["heightCm"] is None:
        warnings.append("height is unavailable")

    if protocol_profile["bodyType"] == "unknown":
        warnings.append("body type is unknown")

    if protocol_profile["ageRange"] == "unknown":
        warnings.append("age range is unknown")

    if protocol_profile["gender"] == "unknown":
        warnings.append("gender is unknown")

    confidence = protocol_profile["confidence"]

    if confidence >= 0.75:
        overall = "good"
    elif confidence >= settings.PROFILE_MIN_CONFIDENCE:
        overall = "fair"
    else:
        overall = "poor"

    enough_valid_frames = valid_count >= min_valid_frames
    quality = {
        "overall": overall,
        "warnings": warnings,
        "profileUsable": overall != "poor" and enough_valid_frames,
        "sampleCount": len(samples),
        "validFrameCount": valid_count,
        "minValidFrames": min_valid_frames,
        "targetSampleCount": min(
            max(int(settings.FRONT_CAMERA_PROFILE_SAMPLE_COUNT), 1),
            int(settings.PROFILE_SAMPLING_CONFIG.get("max_good_frames", 10)),
        ),
    }

    if not quality["profileUsable"]:
        quality["notUsableReason"] = (
            "low_confidence" if overall == "poor" else "insufficient_quality"
        )

    if proximity is not None:
        quality["proximity"] = proximity

    if sampling_mode is not None:
        quality["samplingMode"] = sampling_mode

    return quality
