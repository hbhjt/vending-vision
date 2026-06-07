import math
import statistics
import time
from collections import Counter, deque
from uuid import uuid4

import cv2

from vision.camera import capture_image
from vision.config import settings
from vision.pipeline import infer_image
from vision.process_trace import ProcessTrace
from vision.profile_mapper import (
    age_to_age_range,
    body_type_to_protocol,
    calculate_confidence,
    vision_profile_to_protocol,
)
from vision.proximity import check_proximity_once_with_image
from vision.protocol import now_iso
from vision.schema import VisionProfile


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
    tracking=None,
    occupancy=None,
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
        "sampleCount": len(samples),
        "validFrameCount": valid_count,
        "minValidFrames": min_valid_frames,
        "targetSampleCount": settings.PROFILE_SAMPLE_COUNT,
        "bodyBufferFrameCount": len(
            [sample for sample in samples if sample.get("source") == "body_buffer"]
        ),
        "closeSampleFrameCount": len(
            [sample for sample in samples if sample.get("source") == "close_sample"]
        ),
        "faceVoteFrameCount": len(
            [sample for sample in samples if sample.get("source") == "face_vote"]
        ),
        "faceVoteQualifiedFrameCount": len(
            [sample for sample in samples if is_face_vote_candidate(sample)]
        ),
        "samples": samples,
    }

    if proximity is not None:
        quality["proximity"] = proximity

    if tracking is not None:
        quality["tracking"] = tracking

    if occupancy is not None:
        quality["occupancy"] = occupancy

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


def sample_frame(source, index, proximity=None, track=None):
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


def save_trace_sample(trace, sample):
    if trace is None:
        return

    trace.save_sample(
        index=sample["summary"]["index"],
        raw_image=sample["rawImage"],
        inference_image=sample["inferenceImage"],
        profile=sample["profile"],
        protocol_profile=sample["protocolProfile"],
        frame_quality=sample["quality"],
        valid=sample["valid"],
    )


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


def collect_face_vote_samples(samples, proximity, track, trace):
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
        save_trace_sample(trace, sample)

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

    if tracking is not None:
        payload["tracking"] = tracking

    if occupancy is not None:
        payload["occupancy"] = occupancy

    if sample is not None:
        payload["sample"] = sample

    if detail is not None:
        payload["detail"] = detail

    return payload


def profile_update(message_type, payload):
    return {
        "message_type": message_type,
        "payload": payload,
    }


def collect_profile_update(include_status: bool = False):
    if settings.MOCK_SCENARIO != "off":
        event_payload = mock_profile_event()

        if event_payload is not None:
            return profile_update("vision.profile_result", event_payload)

        if include_status:
            return profile_update(
                "vision.presence_status",
                build_presence_status(
                    event_id=f"vision-status-{uuid4()}",
                    state="empty"
                    if settings.MOCK_SCENARIO == "no_person"
                    else "waiting",
                    reason=f"mock_{settings.MOCK_SCENARIO}",
                ),
            )

        return None

    event_id = f"vision-event-{uuid4()}"
    trace = None
    proximity = None

    if settings.PROXIMITY_ENABLED:
        proximity, proximity_image = check_proximity_once_with_image()
        signature = target_signature_from_proximity(proximity)
        occupancy_gate = get_occupancy_gate()

        if not proximity["present"]:
            occupancy_gate.mark_absent()
            mark_active_track_missing()
            if include_status:
                return profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state="empty",
                        reason="no_person",
                        proximity=proximity,
                        occupancy=occupancy_gate.public_state(),
                    ),
                )
            return None

        occupancy_gate.mark_present()
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

        if close_enough and not occupancy_gate.can_trigger():
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
                    ),
                )

            return None

        trace = ProcessTrace(event_id)
        trace.save_proximity(proximity_image, proximity)

        if not close_enough:
            track.prune_body_samples()
            body_sample = sample_frame(
                source="body_buffer",
                index=len(track.body_samples) + 1,
                proximity=proximity,
                track=track,
            )
            if body_sample["protocolProfile"]["personPresent"]:
                track.append_body_sample(body_sample)

            trace.finish(
                status="not_pushed",
                reason="person_present_but_not_close",
                payload={
                    "proximity": proximity,
                    "tracking": track.public_state(),
                    "sample": to_public_sample(body_sample),
                },
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
                        occupancy=occupancy_gate.public_state(),
                        sample=to_public_sample(body_sample),
                    ),
                )

            return None
    else:
        signature = {
            "source": "disabled",
            "centerX": 0.5,
            "centerY": 0.5,
            "areaRatio": 0.0,
            "count": 1,
        }
        track = ensure_active_track(signature, "close")
        trace = ProcessTrace(event_id)
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
                    ),
                )

            return None

    track.prune_body_samples()
    samples = list(track.body_samples)

    for index, sample in enumerate(samples, start=1):
        sample["summary"]["index"] = index
        save_trace_sample(trace, sample)

    close_sample = sample_frame(
        source="close_sample",
        index=len(samples) + 1,
        proximity=proximity,
        track=track,
    )
    samples.append(close_sample)
    save_trace_sample(trace, close_sample)

    if close_sample["protocolProfile"]["personPresent"]:
        track.append_body_sample(close_sample)

    collect_face_vote_samples(samples, proximity, track, trace)

    valid_samples = [sample for sample in samples if sample["valid"]]
    public_samples = [to_public_sample(sample) for sample in samples]
    required_valid_frames = 1

    if len(valid_samples) < required_valid_frames:
        if trace is not None:
            trace.finish(
                status="not_pushed",
                reason="not_enough_valid_frames",
                payload={
                    "validFrameCount": len(valid_samples),
                    "minValidFrames": required_valid_frames,
                    "samples": public_samples,
                    "proximity": proximity,
                    "tracking": track.public_state(),
                },
            )
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
                    detail={
                        "validFrameCount": len(valid_samples),
                        "minValidFrames": required_valid_frames,
                    },
                ),
            )
        return None

    aggregated = aggregate_samples(samples)

    if aggregated is None:
        if trace is not None:
            trace.finish(
                status="not_pushed",
                reason="not_enough_valid_frames",
                payload={
                    "validFrameCount": len(valid_samples),
                    "minValidFrames": required_valid_frames,
                    "samples": public_samples,
                    "proximity": proximity,
                    "tracking": track.public_state(),
                },
            )
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
        tracking=track.public_state(),
        occupancy=get_occupancy_gate().public_state(),
        min_valid_frames=required_valid_frames,
        sampling_mode="approach_buffer_immediate_close",
    )

    if protocol_profile["confidence"] < settings.PROFILE_MIN_CONFIDENCE:
        if trace is not None:
            trace.finish(
                status="not_pushed",
                reason="confidence_below_threshold",
                payload={
                    "profile": protocol_profile,
                    "quality": quality,
                },
            )
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
        "profile": protocol_profile,
        "quality": quality,
    }

    trace_dir = trace.finish(status="pushed", payload=payload) if trace else None

    if trace_dir:
        payload["quality"]["trace"] = {
            "eventDir": trace_dir,
        }

    track.update(track.signature, "pushed", match_score=track.match_score)
    get_occupancy_gate().mark_pushed(event_id)
    reset_active_track()

    return profile_update("vision.profile_result", payload)


def collect_profile_event():
    update = collect_profile_update(include_status=False)

    if update and update["message_type"] == "vision.profile_result":
        return update["payload"]

    return None
