def age_to_age_range(age):
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
    mapping = {
        "thin": "slim",
        "medium": "regular",
        "fat": "strong",
        "unknown": "unknown"
    }

    return mapping.get(body_type, "unknown")


def calculate_confidence(profile):
    """
    原型置信度估算。

    不仅看字段是否存在，也要检查数值是否合理。
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