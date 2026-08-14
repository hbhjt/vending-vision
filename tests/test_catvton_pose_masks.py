from types import SimpleNamespace

import numpy as np
import pytest

from vision.catvton_pose_masks import CatVTONPoseError, target_hands_sleeve_masks


class Landmark:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def pose(*, shoulder_y=0.30, hip_y=0.70, left_wrist_y=0.68, right_wrist_y=0.68):
    landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(33)]
    values = {
        0: (0.50, 0.14),
        2: (0.47, 0.13),
        5: (0.53, 0.13),
        7: (0.43, 0.16),
        8: (0.57, 0.16),
        11: (0.34, shoulder_y),
        12: (0.66, shoulder_y),
        13: (0.25, (shoulder_y + left_wrist_y) * 0.55),
        14: (0.75, (shoulder_y + right_wrist_y) * 0.55),
        15: (0.22, left_wrist_y),
        16: (0.78, right_wrist_y),
        23: (0.40, hip_y),
        24: (0.60, hip_y),
    }
    for index, (x, y) in values.items():
        landmarks[index] = Landmark(x, y)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))


def centroid(mask):
    ys, xs = np.where(mask > 0)
    return float(xs.mean()), float(ys.mean())


def test_pose_derived_masks_follow_shoulders_hips_and_sleeve_length():
    person = np.zeros((180, 120, 3), dtype=np.uint8)
    garment = np.zeros((80, 60, 4), dtype=np.uint8)
    garment[8:74, 6:54, :3] = (230, 80, 40)
    garment[8:74, 6:54, 3] = 255

    high_target, _high_hands, short_sleeves = target_hands_sleeve_masks(
        person,
        garment,
        template="tshirt_short_sleeve",
        pose_results=pose(shoulder_y=0.26, hip_y=0.62, left_wrist_y=0.62, right_wrist_y=0.62),
    )
    low_target, _low_hands, long_sleeves = target_hands_sleeve_masks(
        person,
        garment,
        template="tshirt_long_sleeve",
        pose_results=pose(shoulder_y=0.36, hip_y=0.82, left_wrist_y=0.82, right_wrist_y=0.82),
    )

    assert centroid(low_target)[1] > centroid(high_target)[1] + 12
    assert int(long_sleeves.sum()) > int(short_sleeves.sum()) * 1.5
    assert not np.array_equal(high_target, low_target)


def test_missing_pose_uses_nonempty_proportional_fallback_masks():
    person = np.zeros((180, 120, 3), dtype=np.uint8)
    garment = np.zeros((80, 60, 4), dtype=np.uint8)
    garment[8:74, 6:54, 3] = 255

    target, hands, sleeves = target_hands_sleeve_masks(
        person,
        garment,
        template="tshirt_short_sleeve",
        pose_results=SimpleNamespace(pose_landmarks=None),
    )

    assert target.shape == person.shape[:2]
    assert target.dtype == np.uint8
    assert int(target.sum()) > 0
    assert int(hands.sum()) == 0
    assert int(sleeves.sum()) > 0


def test_detector_error_uses_same_fallback_masks(monkeypatch):
    person = np.zeros((180, 120, 3), dtype=np.uint8)
    garment = np.zeros((80, 60, 4), dtype=np.uint8)
    garment[8:74, 6:54, 3] = 255

    def detector_failure(_person):
        raise RuntimeError("mediapipe is unavailable")

    monkeypatch.setattr("vision.catvton_pose_masks._detect_pose", detector_failure)

    target, hands, sleeves = target_hands_sleeve_masks(
        person,
        garment,
        template="tshirt_long_sleeve",
        pose_results=None,
    )

    assert int(target.sum()) > 0
    assert int(hands.sum()) == 0
    assert int(sleeves.sum()) > 0


def test_missing_pose_does_not_accept_an_invalid_garment():
    person = np.zeros((180, 120, 3), dtype=np.uint8)
    invalid_garment = np.zeros((80, 60, 3), dtype=np.uint8)

    with pytest.raises(CatVTONPoseError, match="official_catvton_invalid_garment"):
        target_hands_sleeve_masks(
            person,
            invalid_garment,
            template="tshirt_short_sleeve",
            pose_results=SimpleNamespace(pose_landmarks=None),
        )
