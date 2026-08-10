from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.fast_tryon import (
    FastTryOnRuntime,
    PoseUnavailableError,
    ValidatedGarmentSource,
)


def _pose(*, center_x=0.5, shoulder_width=0.28, torso=0.38, tilt=0.0, confidence=0.95):
    # Coordinates are deliberately based on an equivalent person's shoulder
    # and hip axes, not a screen band.  Optional face/arm points exercise
    # body-aware composition without requiring MediaPipe in this unit test.
    shoulder_y = 0.32
    left = np.array([-shoulder_width / 2, 0.0])
    right = np.array([shoulder_width / 2, 0.0])
    axis = np.array([np.cos(tilt), np.sin(tilt)])
    down = np.array([-np.sin(tilt), np.cos(tilt)])
    shoulder = np.array([center_x, shoulder_y])
    left_shoulder = shoulder + left * axis
    right_shoulder = shoulder + right * axis
    left_hip = shoulder + down * torso + left * axis * 0.9
    right_hip = shoulder + down * torso + right * axis * 0.9
    points = [SimpleNamespace(x=0.5, y=0.18, visibility=confidence) for _ in range(33)]
    values = {
        0: (center_x, 0.18),
        2: (center_x - 0.03, 0.17),
        5: (center_x + 0.03, 0.17),
        7: (center_x - 0.07, 0.18),
        8: (center_x + 0.07, 0.18),
        11: left_shoulder,
        12: right_shoulder,
        13: left_shoulder + down * torso * 0.42 - axis * 0.10,
        14: right_shoulder + down * torso * 0.42 + axis * 0.10,
        15: left_shoulder + down * torso * 0.78 - axis * 0.11,
        16: right_shoulder + down * torso * 0.78 + axis * 0.11,
        23: left_hip,
        24: right_hip,
    }
    for index, value in values.items():
        points[index] = SimpleNamespace(x=float(value[0]), y=float(value[1]), visibility=confidence)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))


def _garment(template="tshirt_short_sleeve"):
    source = np.zeros((80, 60, 4), dtype=np.uint8)
    source[5:75, 4:56] = (20, 120, 220, 230)
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    payload = encoded.tobytes()
    return ValidatedGarmentSource(
        png_bytes=payload,
        digest="sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
        template=template,
    )


def _changed_bbox(result, frame):
    image = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    changed = np.any(image != frame, axis=2)
    ys, xs = np.where(changed)
    assert len(xs)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def test_fast_geometry_follows_pose_translation_scale_and_tilt():
    runtime = FastTryOnRuntime()
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    source = _garment()
    centered = _changed_bbox(runtime.render(frame, source, pose_results=_pose()), frame)
    moved = _changed_bbox(runtime.render(frame, source, pose_results=_pose(center_x=0.68)), frame)
    near = _changed_bbox(runtime.render(frame, source, pose_results=_pose(shoulder_width=0.38)), frame)
    tilted = _changed_bbox(runtime.render(frame, source, pose_results=_pose(tilt=0.22)), frame)
    assert moved[0] > centered[0] + 40
    assert moved[2] > centered[2] + 40
    assert (near[2] - near[0]) > (centered[2] - centered[0]) * 1.2
    assert tilted[1] != centered[1] or tilted[3] != centered[3]


def test_fast_geometry_long_template_is_longer_and_face_stays_original():
    runtime = FastTryOnRuntime()
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    frame[50:90, 220:260] = (7, 31, 211)
    short = _changed_bbox(runtime.render(frame, _garment(), pose_results=_pose()), frame)
    long = _changed_bbox(runtime.render(frame, _garment("tshirt_long_sleeve"), pose_results=_pose()), frame)
    assert (long[3] - long[1]) > (short[3] - short[1]) * 1.2
    result = cv2.imdecode(
        np.frombuffer(runtime.render(frame, _garment(), pose_results=_pose()), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert np.array_equal(result[50:90, 220:260], frame[50:90, 220:260])


@pytest.mark.parametrize(
    "pose",
    [None, _pose(confidence=0.2), _pose(shoulder_width=0.01, torso=0.01)],
)
def test_fast_geometry_rejects_unavailable_or_degenerate_pose(pose):
    runtime = FastTryOnRuntime()
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    with pytest.raises(PoseUnavailableError, match="pose_unavailable"):
        runtime.render(frame, _garment(), pose_results=pose)
