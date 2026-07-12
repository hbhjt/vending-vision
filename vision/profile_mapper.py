"""
画像映射模块

负责 VisionProfile（内部数据模型）与协议格式之间的转换：
- vision_profile_to_protocol: 内部模型 -> 协议 JSON 格式
- calculate_confidence: 计算画像的整体置信度
- 体型/年龄范围的映射转换
"""


def age_to_age_range(age):
    """将年龄数值映射为协议定义的年龄段。

    映射关系：
    - None -> "unknown"
    - 0-12 -> "child"
    - 13-17 -> "teen"
    - 18-59 -> "adult"
    - 60+ -> "senior"
    """
    if age is None:
        return "unknown"

    if age < 13:
        return "child"
    elif age < 18:
        return "teen"
    elif age < 60:
        return "adult"
    else:
        return "senior"


def body_type_to_protocol(body_type):
    """将内部体型标识映射为协议定义的体型标识。

    映射关系：
    - thin -> "slim"（偏瘦）
    - medium -> "regular"（标准）
    - fat -> "strong"（偏胖）
    - unknown -> "unknown"
    """
    mapping = {
        "thin": "slim",
        "medium": "regular",
        "fat": "strong",
        "unknown": "unknown"
    }

    return mapping.get(body_type, "unknown")


def calculate_confidence(profile):
    """计算画像的整体置信度（0.1 ~ 0.95）。

    评分规则：
    - 基础分: 0.3
    - presence=True: +0.2
    - 身高在合理范围 (140~200): +0.15，否则 -0.15
    - 肩宽在合理范围 (32~55): +0.1，否则 -0.1
    - 体型已知: +0.1
    - 上衣颜色已知: +0.05
    - 性别已知: +0.02（低权重，当前不稳定）
    - 年龄已知: +0.02（低权重，当前不稳定）

    最后限制在 [0.1, 0.95] 范围内。
    """
    score = 0.3

    if profile.presence:
        score += 0.2

    # 身高合理范围
    if profile.height_cm is not None:
        if 140 <= profile.height_cm <= 200:
            score += 0.15
        else:
            score -= 0.15

    # 肩宽合理范围
    if profile.shoulder_width_cm is not None:
        if 32 <= profile.shoulder_width_cm <= 55:
            score += 0.1
        else:
            score -= 0.1

    if profile.body_type != "unknown":
        score += 0.1

    if profile.upper_color != "unknown":
        score += 0.05

    # 年龄性别目前不稳定，只给很低权重
    if profile.gender != "unknown":
        score += 0.02

    if profile.age is not None:
        score += 0.02

    if score < 0.1:
        score = 0.1

    return round(min(score, 0.95), 2)


def vision_profile_to_protocol(profile):
    """将 VisionProfile 内部模型转换为协议格式的 JSON 字典。

    转换包括：
    - 年龄 -> 年龄段 (child/teen/adult/senior)
    - 体型 -> 协议体型 (slim/regular/strong)
    - 计算置信度
    """
    return {
        "personPresent": profile.presence,
        "heightCm": profile.height_cm,
        "shoulderWidthCm": profile.shoulder_width_cm,
        "ageRange": age_to_age_range(profile.age),
        "gender": profile.gender,
        "bodyType": body_type_to_protocol(profile.body_type),
        "upperColor": profile.upper_color or "unknown",
        "confidence": calculate_confidence(profile)
    }
