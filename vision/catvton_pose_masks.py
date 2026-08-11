"""Pose-derived CatVTON masks for the official AI worker.

This is the production replacement for the temporary fixed-percentage mask.
The geometry follows the captured person pose and the validated garment alpha:
no fallback pose and no screen-band placement are accepted on the customer
attempt path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


class CatVTONPoseError(RuntimeError):
    def __init__(self, code: str = "official_catvton_pose_unavailable"):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class PoseGeometry:
    shoulder_center: np.ndarray
    shoulder_span: float
    across_unit: np.ndarray
    torso_down_unit: np.ndarray
    torso_length: float
    landmarks: dict[str, np.ndarray]


_POSE_LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


def _landmark_points(pose_results, width: int, height: int) -> dict[str, np.ndarray]:
    if pose_results is None:
        raise CatVTONPoseError()
    collection = getattr(pose_results, "pose_landmarks", pose_results)
    raw = getattr(collection, "landmark", collection)
    if raw is None:
        raise CatVTONPoseError()
    points: dict[str, np.ndarray] = {}
    for name, index in _POSE_LANDMARKS.items():
        try:
            landmark = raw[index] if not isinstance(raw, dict) else raw.get(index, raw.get(name))
        except (IndexError, KeyError, TypeError):
            continue
        if landmark is None:
            continue
        visibility = float(getattr(landmark, "visibility", 1.0))
        x = float(getattr(landmark, "x", float("nan")))
        y = float(getattr(landmark, "y", float("nan")))
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not math.isfinite(visibility)
            or visibility < 0.55
            or x < -0.10
            or x > 1.10
            or y < -0.10
            or y > 1.10
        ):
            continue
        points[name] = np.array([x * width, y * height], dtype=np.float32)
    return points


def pose_geometry(pose_results, width: int, height: int) -> PoseGeometry:
    points = _landmark_points(pose_results, width, height)
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in points for name in required):
        raise CatVTONPoseError()
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    if left_shoulder[0] <= right_shoulder[0]:
        screen_left_shoulder = left_shoulder
        screen_right_shoulder = right_shoulder
    else:
        screen_left_shoulder = right_shoulder
        screen_right_shoulder = left_shoulder
    shoulder_axis = screen_right_shoulder - screen_left_shoulder
    shoulder_span = float(np.linalg.norm(shoulder_axis))
    shoulder_center = (screen_left_shoulder + screen_right_shoulder) * 0.5
    hip_center = (points["left_hip"] + points["right_hip"]) * 0.5
    torso_axis = hip_center - shoulder_center
    torso_length = float(np.linalg.norm(torso_axis))
    frame_diagonal = float(math.hypot(width, height))
    if (
        not math.isfinite(shoulder_span)
        or not math.isfinite(torso_length)
        or shoulder_span < max(12.0, width * 0.055)
        or torso_length < max(18.0, height * 0.08)
        or torso_length > frame_diagonal * 1.25
        or float(np.dot(torso_axis, np.array([0.0, 1.0]))) <= 0
    ):
        raise CatVTONPoseError()
    return PoseGeometry(
        shoulder_center=shoulder_center,
        shoulder_span=shoulder_span,
        across_unit=shoulder_axis / shoulder_span,
        torso_down_unit=torso_axis / torso_length,
        torso_length=torso_length,
        landmarks=points,
    )


def _detect_pose(person_rgb: np.ndarray):
    try:
        from vision.pose_estimator import PoseEstimator

        return PoseEstimator().detect(cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        raise CatVTONPoseError() from exc


def _garment_alpha_bounds(garment_rgba: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if garment_rgba.ndim != 3 or garment_rgba.shape[2] != 4:
        raise CatVTONPoseError("official_catvton_invalid_garment")
    alpha = np.where(garment_rgba[:, :, 3] >= 12, 255, 0).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    points = cv2.findNonZero(alpha)
    if points is None:
        raise CatVTONPoseError("official_catvton_invalid_garment")
    return alpha, cv2.boundingRect(points)


def target_hands_sleeve_masks(
    person_rgb: np.ndarray,
    garment_rgba: np.ndarray,
    *,
    template: str,
    pose_results=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = person_rgb.shape[:2]
    geometry = pose_geometry(pose_results or _detect_pose(person_rgb), width, height)
    alpha, (x0, y0, source_width, source_height) = _garment_alpha_bounds(garment_rgba)
    source = alpha[y0 : y0 + source_height, x0 : x0 + source_width]

    long_sleeve = template == "tshirt_long_sleeve"
    target_width = geometry.shoulder_span * (1.34 if long_sleeve else 1.26)
    target_height = geometry.torso_length * (1.38 if long_sleeve else 1.08)
    top_center = geometry.shoulder_center + geometry.torso_down_unit * geometry.torso_length * 0.025
    bottom_center = top_center + geometry.torso_down_unit * target_height
    half_width = geometry.across_unit * (target_width * 0.5)
    destination = np.asarray(
        [
            top_center - half_width,
            top_center + half_width,
            bottom_center + half_width,
            bottom_center - half_width,
        ],
        dtype=np.float32,
    )
    source_corners = np.asarray(
        [[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_corners, destination)
    target = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    target = cv2.morphologyEx(np.where(target >= 12, 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    hands = np.zeros((height, width), dtype=np.uint8)
    sleeves = np.zeros_like(hands)
    span = geometry.shoulder_span
    for side in ("left", "right"):
        shoulder = geometry.landmarks.get(f"{side}_shoulder")
        elbow = geometry.landmarks.get(f"{side}_elbow")
        wrist = geometry.landmarks.get(f"{side}_wrist")
        if shoulder is None or elbow is None:
            continue
        if wrist is not None:
            palm_center = wrist + (wrist - elbow) * 0.08
            cv2.circle(hands, tuple(np.rint(palm_center).astype(int)), max(8, round(span * 0.09)), 255, -1, cv2.LINE_AA)
        if long_sleeve and wrist is not None:
            sleeve_end = elbow + (wrist - elbow) * 0.92
            cv2.line(sleeves, tuple(np.rint(shoulder).astype(int)), tuple(np.rint(elbow).astype(int)), 255, max(12, round(span * 0.32)), cv2.LINE_AA)
            cv2.line(sleeves, tuple(np.rint(elbow).astype(int)), tuple(np.rint(sleeve_end).astype(int)), 255, max(10, round(span * 0.24)), cv2.LINE_AA)
        elif not long_sleeve:
            band_start = shoulder + (elbow - shoulder) * 0.49
            band_end = shoulder + (elbow - shoulder) * 0.66
            cv2.line(sleeves, tuple(np.rint(band_start).astype(int)), tuple(np.rint(band_end).astype(int)), 255, max(11, round(span * 0.30)), cv2.LINE_AA)
            if wrist is not None:
                protect_start = shoulder + (elbow - shoulder) * 0.72
                cv2.line(hands, tuple(np.rint(protect_start).astype(int)), tuple(np.rint(wrist).astype(int)), 255, max(8, round(span * 0.12)), cv2.LINE_AA)
    return target, hands, sleeves
