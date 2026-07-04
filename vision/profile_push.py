import math
import statistics
import time
from collections import Counter, deque
from uuid import uuid4

import cv2

from vision.camera_manager import read_camera
from vision.camera_owner import (
    acquire_front_camera,
    front_camera_io_lock,
    get_front_camera_owner,
    release_front_camera,
)
from vision.config import settings
from vision.pipeline import infer_image
from vision.profile_mapper import (
    age_to_age_range,
    body_type_to_protocol,
    calculate_confidence,
    vision_profile_to_protocol,
)
from vision.proximity import check_proximity_once_with_image
from vision.protocol import now_iso
from vision.schema import VisionProfile
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


class TemporaryProfileTrack:
    def __init__(self, signature):
        self.track_id = f"profile-track-{uuid4()}"
        self.state = "present"
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.missing_count = 0
        self.match_score = 1.0
        self.signature = signature
        self.body_samples = deque(
            maxlen=max(settings.PROFILE_BODY_BUFFER_MAX_FRAMES, 1)
        )
        self.announced_presence_states = set()

    def update(self, signature, state, match_score=1.0):
        self.signature = signature
        self.state = state
        self.match_score = round(float(match_score), 4)
        self.updated_at = time.time()
        self.missing_count = 0

    def mark_missing(self):
        self.state = "leaving"
        self.missing_count += 1
        self.updated_at = time.time()

    def is_lost(self):
        return self.missing_count > settings.PROFILE_TRACK_MAX_MISSING_FRAMES

    def prune_body_samples(self):
        now = time.time()
        ttl_seconds = max(settings.PROFILE_BODY_BUFFER_TTL_MS, 0) / 1000.0

        while self.body_samples and now - self.body_samples[0]["capturedAt"] > ttl_seconds:
            self.body_samples.popleft()

    def append_body_sample(self, sample):
        self.body_samples.append(sample)

    def announce_presence_once(self, state):
        if state in self.announced_presence_states:
            return False

        self.announced_presence_states.add(state)
        return True

    def public_state(self):
        return {
            "trackId": self.track_id,
            "state": self.state,
            "ageMs": int((time.time() - self.started_at) * 1000),
            "missingCount": self.missing_count,
            "matchScore": self.match_score,
            "bodyBufferFrameCount": len(self.body_samples),
            "target": self.signature,
        }


_active_track = None
_mock_pending_profile_payload = None
_mock_pending_departure_payload = None


class ProfileOccupancyGate:
    def __init__(self):
        self.state = "empty"
        self.absent_count = 0
        self.last_event_id = None
        self.last_pushed_at = None
        self.updated_at = time.time()

    def can_trigger(self):
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return True

        return self.state != "occupied"

    def mark_present(self):
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        if self.state == "empty":
            self.state = "tracking"

        self.absent_count = 0
        self.updated_at = time.time()

    def mark_pushed(self, event_id):
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        self.state = "occupied"
        self.absent_count = 0
        self.last_event_id = event_id
        self.last_pushed_at = time.time()
        self.updated_at = self.last_pushed_at

    def mark_absent(self):
        if not settings.PROFILE_OCCUPANCY_GATE_ENABLED:
            return

        self.absent_count += 1
        self.updated_at = time.time()

        if self.absent_count >= settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES:
            self.state = "empty"
            self.absent_count = 0
            self.last_event_id = None
            self.last_pushed_at = None

    def public_state(self):
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


_occupancy_gate = ProfileOccupancyGate()


class PersonDepartureTracker:
    def __init__(self):
        self.active = False
        self.absent_count = 0
        self.last_seen_at = None
        self.last_seen_monotonic = None
        self.departed_announced = False

    def mark_present(self):
        self.active = True
        self.absent_count = 0
        self.last_seen_at = now_iso()
        self.last_seen_monotonic = time.time()
        self.departed_announced = False

    def mark_absent(self, reason="no_person", ambient_light=None):
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


_departure_tracker = PersonDepartureTracker()


class FrontCameraBusy(RuntimeError):
    def __init__(self, owner_status=None, reason="front_camera_busy"):
        super().__init__(reason)
        self.owner_status = owner_status or {}
        self.reason = reason


def active_try_on_status():
    status = get_try_on_status()
    return status if status.get("activeSessionId") else None


def wait_for_front_camera_owner(event_id):
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
    event_id,
    reason,
    proximity=None,
    tracking=None,
    occupancy=None,
    ambient_light=None,
    owner_status=None,
):
    return profile_update(
        "vision.presence_status",
        build_presence_status(
            event_id=event_id,
            state="waiting",
            reason=reason,
            proximity=proximity,
            tracking=tracking,
            occupancy=occupancy,
            ambient_light=ambient_light,
        ),
    )


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


def estimate_ambient_light(image):
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
    width = settings.PROFILE_DETECTION_WIDTH
    height = settings.PROFILE_DETECTION_HEIGHT

    if not width or not height or width <= 0 or height <= 0:
        return image

    return cv2.resize(image, (width, height))


def target_signature_from_proximity(proximity):
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
    global _active_track

    if _active_track is None:
        return

    _active_track.mark_missing()

    if _active_track.is_lost():
        _active_track = None


def reset_active_track():
    global _active_track
    _active_track = None


def get_occupancy_gate():
    return _occupancy_gate


def protocol_occupancy_snapshot(proximity=None, state_hint: str | None = None):
    proximity = proximity or {}

    if state_hint in {"none", "single", "multiple", "unknown"}:
        state = state_hint
    elif max(
        int(proximity.get("personCount") or 0),
        int(proximity.get("faceCount") or 0),
    ) > 1:
        state = "multiple"
    elif proximity.get("present"):
        state = "single"
    else:
        state = "none"

    confidence = 0.5
    if state == "single":
        confidence = 0.82
    elif state == "multiple":
        confidence = 0.78
    elif state == "none":
        confidence = 0.8

    return {
        "state": state,
        "confidence": round(confidence, 2),
    }


def normalize_protocol_occupancy(occupancy=None, proximity=None):
    if isinstance(occupancy, dict):
        state = occupancy.get("state")
        if state in {"none", "single", "multiple", "unknown"}:
            return occupancy

    return protocol_occupancy_snapshot(proximity)


def sample_weight(sample, purpose="general"):
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
    scores = Counter()

    for value, weight in items:
        if value in (None, "unknown"):
            continue
        scores[value] += weight

    if not scores:
        return "unknown"

    return scores.most_common(1)[0][0]


def weighted_median_or_none(items):
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


def aggregate_samples(samples):
    if not samples:
        return None

    valid_samples = [
        sample
        for sample in samples
        if sample["protocolProfile"]["personPresent"]
        and sample["protocolProfile"]["confidence"] >= settings.PROFILE_MIN_CONFIDENCE
    ]

    if not valid_samples:
        return None

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

    height_cm = weighted_median_or_none(
        [(sample["profile"].height_cm, sample_weight(sample, "body")) for sample in body_pool]
    )
    shoulder_width_cm = weighted_median_or_none(
        [
            (sample["profile"].shoulder_width_cm, sample_weight(sample, "body"))
            for sample in body_pool
        ]
    )
    body_type = weighted_mode_or_unknown(
        [(sample["profile"].body_type, sample_weight(sample, "body")) for sample in body_pool]
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
            [(sample["profile"].gender, sample_weight(sample, "face")) for sample in face_pool]
        ),
        height_cm=height_cm,
        shoulder_width_cm=shoulder_width_cm,
        body_type=body_type,
        upper_color=upper_color,
        presence=True,
    )

    confidence_values = [sample["protocolProfile"]["confidence"] for sample in valid_samples]
    confidence = round(
        max(calculate_confidence(profile), statistics.mean(confidence_values)),
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
    warnings = []
    min_valid_frames = (
        settings.PROFILE_MIN_VALID_FRAMES
        if min_valid_frames is None
        else min_valid_frames
    )

    if valid_count < min_valid_frames:
        warnings.append(
            f"valid frames {valid_count}/{min_valid_frames}"
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
        "profileUsable": overall != "poor",
        "sampleCount": len(samples),
        "validFrameCount": valid_count,
        "minValidFrames": min_valid_frames,
        "targetSampleCount": settings.PROFILE_SAMPLE_COUNT,
    }

    if not quality["profileUsable"]:
        quality["notUsableReason"] = "low_confidence"

    if proximity is not None:
        quality["proximity"] = proximity

    if sampling_mode is not None:
        quality["samplingMode"] = sampling_mode

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


def sample_frame(source, index, proximity=None, track=None):
    owner_status = get_front_camera_owner()
    if owner_status.get("owner") != "vision":
        raise FrontCameraBusy(
            owner_status=owner_status,
            reason="front_camera_owner_changed",
        )

    with front_camera_io_lock():
        image = read_camera("front", warmup_frames=1)

    inference_image = resize_for_profile_inference(image)
    quality = image_quality(image)
    profile = infer_image(inference_image)
    protocol_profile = vision_profile_to_protocol(profile)
    confidence = protocol_profile["confidence"]
    is_valid = bool(
        protocol_profile["personPresent"]
        and confidence >= settings.PROFILE_MIN_CONFIDENCE
    )

    sample = {
        "index": index,
        "source": source,
        "capturedAt": time.time(),
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
            "inferenceWidth": inference_image.shape[1],
            "inferenceHeight": inference_image.shape[0],
            "valid": is_valid,
            "hasBodyMeasure": bool(
                profile.height_cm is not None
                or profile.shoulder_width_cm is not None
                or profile.body_type != "unknown"
            ),
            "hasFaceAttribute": bool(
                profile.age is not None or profile.gender != "unknown"
            ),
        },
        "rawImage": image,
        "inferenceImage": inference_image,
    }

    sample["summary"]["bodyWeight"] = sample_weight(sample, "body")
    sample["summary"]["faceWeight"] = sample_weight(sample, "face")

    return sample

def to_public_sample(sample):
    return dict(sample["summary"])


def is_face_vote_candidate(sample):
    summary = sample.get("summary") or {}
    quality = sample.get("quality") or {}

    return bool(
        summary.get("hasFaceAttribute")
        and sample.get("valid")
        and quality.get("sharpness", 0.0)
        >= settings.PROFILE_FACE_VOTE_MIN_SHARPNESS
    )


def collect_face_vote_samples(samples, proximity, track):
    if not settings.PROFILE_FACE_VOTE_ENABLED:
        return

    target_count = max(settings.PROFILE_FACE_VOTE_SAMPLE_COUNT, 0)
    if target_count <= 0:
        return

    qualified_count = len([sample for sample in samples if is_face_vote_candidate(sample)])

    while qualified_count < target_count:
        if settings.PROFILE_FACE_VOTE_INTERVAL_MS > 0:
            time.sleep(settings.PROFILE_FACE_VOTE_INTERVAL_MS / 1000.0)

        sample = sample_frame(
            source="face_vote",
            index=len(samples) + 1,
            proximity=proximity,
            track=track,
        )
        samples.append(sample)

        if is_face_vote_candidate(sample):
            qualified_count += 1

        if len([item for item in samples if item.get("source") == "face_vote"]) >= target_count:
            break


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
    return {
        "message_type": message_type,
        "payload": dict(payload),
    }


def collect_front_profile_update(
    event_id,
    proximity,
    track,
    close_enough,
    ambient_light,
    include_status,
):
    owner_result = wait_for_front_camera_owner(event_id)

    if not owner_result.get("ok"):
        reason = owner_result.get("error") or "front_camera_busy"
        mark_vision_session_waiting_front_camera(
            reason=reason,
            owner_status=owner_result,
        )

        if include_status:
            return build_front_camera_waiting_update(
                event_id=event_id,
                reason=reason,
                proximity=proximity,
                tracking=track.public_state(),
                occupancy=get_occupancy_gate().public_state(),
                ambient_light=ambient_light,
                owner_status=owner_result,
            )

        return None

    try:
        mark_vision_session_profiling(reason="front_profile_sampling")
        track.prune_body_samples()
        samples = list(track.body_samples)

        for index, sample in enumerate(samples, start=1):
            sample["summary"]["index"] = index

        if not close_enough:
            body_sample = sample_frame(
                source="body_buffer",
                index=len(track.body_samples) + 1,
                proximity=proximity,
                track=track,
            )
            if body_sample["protocolProfile"]["personPresent"]:
                track.append_body_sample(body_sample)

            mark_vision_session_presence(
                "approach_detected",
                reason="person_present_but_not_close",
                proximity=proximity,
                occupancy=get_occupancy_gate().public_state(),
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="approach",
                        reason="person_present_but_not_close",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        sample=to_public_sample(body_sample),
                        ambient_light=ambient_light,
                    ),
                )

            return None

        close_sample = sample_frame(
            source="close_sample",
            index=len(samples) + 1,
            proximity=proximity,
            track=track,
        )
        samples.append(close_sample)

        if close_sample["protocolProfile"]["personPresent"]:
            track.append_body_sample(close_sample)

        collect_face_vote_samples(samples, proximity, track)

        valid_samples = [sample for sample in samples if sample["valid"]]
        public_samples = [to_public_sample(sample) for sample in samples]
        required_valid_frames = 1

        if len(valid_samples) < required_valid_frames:
            mark_vision_session_unusable(reason="not_enough_valid_frames")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="unusable",
                        reason="not_enough_valid_frames",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        detail={
                            "validFrameCount": len(valid_samples),
                            "minValidFrames": required_valid_frames,
                        },
                    ),
                )
            return None

        aggregated = aggregate_samples(samples)

        if aggregated is None:
            mark_vision_session_unusable(reason="not_enough_valid_frames")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="unusable",
                        reason="not_enough_valid_frames",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        detail={
                            "validFrameCount": len(valid_samples),
                            "minValidFrames": required_valid_frames,
                        },
                    ),
                )
            return None

        _, protocol_profile = aggregated
        quality = build_quality(
            protocol_profile,
            public_samples,
            len(valid_samples),
            proximity=proximity,
            min_valid_frames=required_valid_frames,
            sampling_mode="top_presence_front_profile",
        )

        if protocol_profile["confidence"] < settings.PROFILE_MIN_CONFIDENCE:
            mark_vision_session_unusable(reason="confidence_below_threshold")
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="unusable",
                        reason="confidence_below_threshold",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=get_occupancy_gate().public_state(),
                        ambient_light=ambient_light,
                        detail={
                            "confidence": protocol_profile["confidence"],
                            "minConfidence": settings.PROFILE_MIN_CONFIDENCE,
                        },
                    ),
                )
            return None

        payload = {
            "eventId": event_id,
            "detectedAt": now_iso(),
            "occupancy": normalize_protocol_occupancy(
                get_occupancy_gate().public_state(),
                proximity,
            ),
            "profile": protocol_profile,
            "quality": quality,
        }

        track.update(track.signature, "pushed", match_score=track.match_score)
        get_occupancy_gate().mark_pushed(event_id)
        reset_active_track()
        mark_vision_session_profile_pushed(payload)

        return profile_update("vision.profile_result", payload)

    except FrontCameraBusy as exc:
        mark_vision_session_waiting_front_camera(
            reason=exc.reason,
            owner_status=exc.owner_status,
        )

        if include_status:
            return build_front_camera_waiting_update(
                event_id=event_id,
                reason=exc.reason,
                proximity=proximity,
                tracking=track.public_state(),
                occupancy=get_occupancy_gate().public_state(),
                ambient_light=ambient_light,
                owner_status=exc.owner_status,
            )

        return None

    finally:
        release_front_camera("vision", reason=f"profile_done:{event_id}")


def collect_profile_update(
    include_status: bool = False,
    include_ambient_light: bool = False,
    include_departure: bool = False,
):
    global _mock_pending_profile_payload, _mock_pending_departure_payload

    if settings.MOCK_SCENARIO != "off":
        scenario = settings.MOCK_SCENARIO

        if _mock_pending_profile_payload is not None:
            event_payload = _mock_pending_profile_payload
            _mock_pending_profile_payload = None
            mark_vision_session_profile_pushed(event_payload)
            return profile_update("vision.profile_result", event_payload)

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
                    "approach_detected",
                    reason="mock_person_close_profile_pending",
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

        if not proximity["present"]:
            occupancy_gate.mark_absent()
            mark_active_track_missing()
            departure_payload = _departure_tracker.mark_absent(
                reason="no_person",
                ambient_light=ambient_light,
            )

            try_on_status = active_try_on_status()

            if departure_payload is not None and try_on_status:
                mark_active_try_on_departed(departure_payload)
                mark_vision_session_tryon_departed(departure_payload)

            if departure_payload is not None and not try_on_status:
                mark_vision_session_departed(departure_payload)

            if departure_payload is not None and include_departure:
                return profile_update("vision.person_departed", departure_payload)

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="empty",
                        reason="no_person",
                        proximity=proximity,
                        occupancy=occupancy_gate.public_state(),
                        ambient_light=ambient_light,
                    ),
                )
            return None

        occupancy_gate.mark_present()
        _departure_tracker.mark_present()
        close_enough = bool(proximity.get("close") or proximity.get("closeNow"))
        proximity["closeTrigger"] = (
            "close_streak"
            if proximity.get("close")
            else "close_now"
            if proximity.get("closeNow")
            else None
        )
        track_state = "close" if close_enough else "approach"
        track = ensure_active_track(signature, track_state)
        occupancy_snapshot = protocol_occupancy_snapshot(proximity)

        if occupancy_snapshot["state"] == "multiple":
            mark_vision_session_presence(
                "multiple",
                reason="multiple_people_detected",
                proximity=proximity,
                occupancy=occupancy_snapshot,
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="occupied",
                        reason="multiple_people_detected",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=occupancy_snapshot,
                        ambient_light=ambient_light,
                    ),
                )

            return None

        try_on_status = active_try_on_status()
        if try_on_status is not None:
            mark_vision_session_presence(
                "tryon_active",
                reason="front_camera_reserved_by_tryon",
                proximity=proximity,
                occupancy=occupancy_snapshot,
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="waiting",
                        reason="front_camera_reserved_by_tryon",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=occupancy_snapshot,
                        ambient_light=ambient_light,
                    ),
                )

            return None

        if close_enough and not occupancy_gate.can_trigger():
            mark_vision_session_presence(
                "profile_pushed",
                reason="occupancy_gate_locked",
                proximity=proximity,
                occupancy=occupancy_gate.public_state(),
            )

            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="occupied",
                        reason="occupancy_gate_locked",
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=occupancy_gate.public_state(),
                        ambient_light=ambient_light,
                    ),
                )

            return None

        if include_status and track.announce_presence_once(track_state):
            mark_vision_session_presence(
                "approach_detected",
                reason=(
                    "person_close_profile_pending"
                    if close_enough
                    else "person_present_but_not_close"
                ),
                proximity=proximity,
                occupancy=occupancy_gate.public_state(),
            )

            return profile_update(
                "vision.presence_status",
                build_presence_status(
                    event_id=event_id,
                    state="approach",
                    reason=(
                        "person_close_profile_pending"
                        if close_enough
                        else "person_present_but_not_close"
                    ),
                    proximity=proximity,
                    tracking=track.public_state(),
                    occupancy=occupancy_gate.public_state(),
                    ambient_light=ambient_light,
                ),
            )

        return collect_front_profile_update(
            event_id=event_id,
            proximity=proximity,
            track=track,
            close_enough=close_enough,
            ambient_light=ambient_light,
            include_status=include_status,
        )
    else:
        ambient_light = None
        signature = {
            "source": "disabled",
            "centerX": 0.5,
            "centerY": 0.5,
            "areaRatio": 0.0,
            "count": 1,
        }
        track = ensure_active_track(signature, "close")
        occupancy_gate = get_occupancy_gate()
        occupancy_gate.mark_present()

        if not occupancy_gate.can_trigger():
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="occupied",
                        reason="occupancy_gate_locked",
                        tracking=track.public_state(),
                        occupancy=occupancy_gate.public_state(),
                        ambient_light=ambient_light,
                    ),
                )

            return None

    return collect_front_profile_update(
        event_id=event_id,
        proximity=proximity,
        track=track,
        close_enough=True,
        ambient_light=ambient_light,
        include_status=include_status,
    )


def collect_profile_event():
    update = collect_profile_update(include_status=False)

    if update and update["message_type"] == "vision.profile_result":
        return update["payload"]

    return None
