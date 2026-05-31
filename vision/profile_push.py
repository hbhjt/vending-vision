import statistics
import time
from collections import Counter
from uuid import uuid4

import cv2

from vision.camera import capture_image
from vision.config import settings
from vision.pipeline import infer_image
from vision.profile_mapper import (
    age_to_age_range,
    body_type_to_protocol,
    calculate_confidence,
    vision_profile_to_protocol,
)
from vision.proximity import check_proximity_once
from vision.protocol import now_iso
from vision.schema import VisionProfile


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


def resize_for_profile_inference(image):
    width = settings.PROFILE_DETECTION_WIDTH
    height = settings.PROFILE_DETECTION_HEIGHT

    if not width or not height or width <= 0 or height <= 0:
        return image

    return cv2.resize(image, (width, height))


def mode_or_unknown(values):
    values = [value for value in values if value not in (None, "unknown")]

    if not values:
        return "unknown"

    return Counter(values).most_common(1)[0][0]


def median_or_none(values):
    values = [value for value in values if value is not None]

    if not values:
        return None

    return round(float(statistics.median(values)), 1)


def age_from_range(age_range: str):
    mapping = {
        "child": 10,
        "teen": 16,
        "adult": 30,
        "senior": 65,
    }
    return mapping.get(age_range)


def aggregate_profiles(profiles):
    if not profiles:
        return None

    protocol_profiles = [vision_profile_to_protocol(profile) for profile in profiles]

    age_range = mode_or_unknown([item["ageRange"] for item in protocol_profiles])
    body_type_protocol = mode_or_unknown([item["bodyType"] for item in protocol_profiles])

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


def build_quality(protocol_profile, samples, valid_count, proximity=None):
    warnings = []

    if valid_count < settings.PROFILE_SAMPLE_COUNT:
        warnings.append(
            f"valid frames {valid_count}/{settings.PROFILE_SAMPLE_COUNT}"
        )

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
    elif confidence >= 0.45:
        overall = "fair"
    else:
        overall = "poor"

    quality = {
        "overall": overall,
        "warnings": warnings,
        "sampleCount": len(samples),
        "validFrameCount": valid_count,
        "minValidFrames": settings.PROFILE_MIN_VALID_FRAMES,
        "samples": samples,
    }

    if proximity is not None:
        quality["proximity"] = proximity

    return quality


def mock_profile_event():
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
            "profile": protocol_profile,
            "quality": {
                "overall": "fair",
                "warnings": ["mock scenario enabled: success"],
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


def collect_profile_event():
    if settings.MOCK_SCENARIO != "off":
        return mock_profile_event()

    proximity = None

    if settings.PROXIMITY_ENABLED:
        proximity = check_proximity_once()

        if not proximity["close"]:
            return None

    samples = []
    valid_profiles = []

    for index in range(settings.PROFILE_SAMPLE_COUNT):
        image = capture_image()
        inference_image = resize_for_profile_inference(image)
        quality = image_quality(image)
        profile = infer_image(inference_image)
        protocol_profile = vision_profile_to_protocol(profile)
        confidence = protocol_profile["confidence"]

        is_valid = bool(
            protocol_profile["personPresent"]
            and confidence >= settings.PROFILE_MIN_CONFIDENCE
        )

        samples.append(
            {
                "index": index + 1,
                "personPresent": protocol_profile["personPresent"],
                "confidence": confidence,
                "brightness": quality["brightness"],
                "sharpness": quality["sharpness"],
                "inferenceWidth": inference_image.shape[1],
                "inferenceHeight": inference_image.shape[0],
                "valid": is_valid,
            }
        )

        if is_valid:
            valid_profiles.append(profile)

        if index < settings.PROFILE_SAMPLE_COUNT - 1:
            time.sleep(settings.PROFILE_SAMPLE_INTERVAL_MS / 1000.0)

    if len(valid_profiles) < settings.PROFILE_MIN_VALID_FRAMES:
        return None

    _, protocol_profile = aggregate_profiles(valid_profiles)
    quality = build_quality(
        protocol_profile,
        samples,
        len(valid_profiles),
        proximity=proximity,
    )

    if protocol_profile["confidence"] < settings.PROFILE_MIN_CONFIDENCE:
        return None

    return {
        "eventId": f"vision-event-{uuid4()}",
        "detectedAt": now_iso(),
        "profile": protocol_profile,
        "quality": quality,
    }
