"""Semantic public-seam checks for the single garment composition module."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vision.garment_composer import GarmentComposer, GarmentFetchError, PoseUnavailableError, TransparentGarmentSource


def _pose(*, crossed=False, arms=True, shoulder_width=0.30, torso=0.41, hips=None):
    points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    values = {
        0: (0.50, 0.16),
        2: (0.47, 0.16),
        5: (0.53, 0.16),
        7: (0.43, 0.18),
        8: (0.57, 0.18),
        11: (0.50 - shoulder_width / 2, 0.31),
        12: (0.50 + shoulder_width / 2, 0.31),
        23: (0.50 - shoulder_width * 0.4, 0.31 + torso),
        24: (0.50 + shoulder_width * 0.4, 0.31 + torso),
    }
    if hips is not None:
        values.update({23: hips[0], 24: hips[1]})
    if crossed:
        values.update({13: (0.54, 0.45), 15: (0.65, 0.54), 14: (0.46, 0.45), 16: (0.35, 0.54)})
    elif arms:
        values.update({13: (0.28, 0.48), 15: (0.25, 0.66), 14: (0.72, 0.48), 16: (0.75, 0.66)})
    for index, (x, y) in values.items():
        points[index] = SimpleNamespace(x=x, y=y, visibility=0.95)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points))


class _FixturePoseEstimator:
    def __init__(self, pose):
        self._pose = pose

    def detect(self, _frame):
        return self._pose


def _source(template="tshirt_short_sleeve"):
    # Independently meaningful source regions: red left sleeve, green torso,
    # and blue right sleeve. Their expected survival is not derived from the
    # compositor's transforms or masks.
    image = np.zeros((120, 140, 4), dtype=np.uint8)
    image[24:72, 4:44] = (0, 0, 255, 255)
    image[18:116, 36:104] = (0, 220, 0, 255)
    image[24:72, 96:136] = (255, 0, 0, 255)
    if template == "tshirt_long_sleeve":
        image[64:111, 12:43] = (0, 0, 255, 255)
        image[64:111, 97:128] = (255, 0, 0, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(
        png_bytes=png,
        digest="sha256:" + hashlib.sha256(png).hexdigest(),
        template=template,
    )


def _wide_torso_short_source():
    """A wide short-sleeve shirt has real bilateral sleeve evidence."""
    image = np.zeros((140, 180, 4), dtype=np.uint8)
    image[22:128, 30:150] = (20, 120, 220, 255)
    image[38:82, 8:35] = (20, 120, 220, 255)
    image[38:82, 145:172] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def _antialiased_boundary_short_source(*, fringe_alpha=100, fringe_length=24):
    """完整短袖主体只留一像素边距，右侧可带短小抗锯齿尾部。"""
    image = np.zeros((140, 180, 4), dtype=np.uint8)
    image[1:128, 30:150] = (20, 120, 220, 255)
    image[38:82, 8:35] = (20, 120, 220, 255)
    image[38:82, 145:179] = (20, 120, 220, 255)
    image[48 : 48 + fringe_length, 179] = (0, 255, 255, fringe_alpha)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def _low_confidence_short_source(*, high_alpha_pixel=False):
    """同一低置信 T 恤可只改变一个像素，用于证明判定不依赖单点阈值。"""
    image = np.zeros((140, 180, 4), dtype=np.uint8)
    image[22:128, 30:150] = (20, 120, 220, 127)
    image[38:82, 8:35] = (20, 120, 220, 127)
    image[38:82, 145:172] = (20, 120, 220, 127)
    if high_alpha_pixel:
        image[64, 90, 3] = 128
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def _right_edge_cropped_short_source(*, canvas_height=140, low_alpha_tail=False):
    """同一 12px 实心右缘裁切主体，可增加无关纵向透明留白或近透明尾部。"""
    image = np.zeros((canvas_height, 180, 4), dtype=np.uint8)
    image[22:128, 30:150] = (20, 120, 220, 255)
    image[38:82, 8:35] = (20, 120, 220, 255)
    image[38:82, 145:179] = (20, 120, 220, 255)
    image[38:50, 179] = (20, 120, 220, 255)
    if low_alpha_tail:
        image[50:54, 179] = (20, 120, 220, 1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def _alpha_blend_probe_source(patch_alpha):
    """固定透明开口的同一条抗锯齿边缘，只改变其 alpha 混合强度。"""
    image = np.zeros((140, 180, 4), dtype=np.uint8)
    image[22:128, 30:150] = (20, 120, 220, 255)
    image[38:82, 8:35] = (20, 120, 220, 255)
    image[38:82, 145:172] = (20, 120, 220, 255)
    image[86:102, 76:104] = (240, 30, 20, 0)
    image[86:102, 76:80] = (240, 30, 20, patch_alpha)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def _composer(*, crossed=False, arms=True, shoulder_width=0.30, torso=0.41, hips=None):
    return GarmentComposer(
        pose_estimator=_FixturePoseEstimator(
            _pose(crossed=crossed, arms=arms, shoulder_width=shoulder_width, torso=torso, hips=hips)
        )
    )


def _decoded(result):
    image = cv2.imdecode(np.frombuffer(result.png, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _colour_area(image, channel, minimum=220):
    return int(np.count_nonzero(image[:, :, channel] >= minimum))


def _green_torso_bbox(image):
    mask = (image[:, :, 1] >= 200) & (image[:, :, 0] <= 30) & (image[:, :, 2] <= 30)
    ys, xs = np.where(mask)
    assert xs.size
    return xs.min(), ys.min(), xs.max(), ys.max()


def _garment_bbox_and_area(image):
    saturated = image.max(axis=2) - image.min(axis=2) >= 100
    ys, xs = np.where(saturated)
    assert xs.size
    return (xs.min(), ys.min(), xs.max(), ys.max()), int(xs.size)


def _colour_mask(image, colour):
    return np.all(image == colour, axis=2)


def _marker_centroid(image, colour):
    ys, xs = np.where(_colour_mask(image, colour))
    assert xs.size >= 20
    return np.array([xs.mean(), ys.mean()])


def _marker_source():
    """Short-sleeve silhouette with independently placed four-colour markers."""
    image = np.zeros((160, 180, 4), dtype=np.uint8)
    image[26:146, 48:132] = (25, 110, 210, 255)
    image[42:88, 16:53] = (25, 110, 210, 255)
    image[42:88, 127:164] = (25, 110, 210, 255)
    image[52:62, 66:76] = (0, 0, 255, 255)      # red
    image[52:62, 106:116] = (0, 255, 255, 255)  # yellow
    image[102:112, 66:76] = (255, 255, 0, 255)  # cyan
    image[102:112, 106:116] = (255, 0, 255, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")


def test_compose_preserves_source_aspect_and_reports_placement_facts():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    result = _composer().compose(frame, _source(), 1.0)

    assert result.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert result.geometry.source_aspect_ratio == 132 / 98
    assert abs(result.geometry.placed_aspect_ratio - result.geometry.source_aspect_ratio) <= 0.02
    assert result.geometry.center[0] == 240
    assert abs(result.geometry.center[1] - 183) <= 2


def test_compose_garment_pixels_grow_strictly_with_person_scale():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    measurements = []
    for person_scale in (0.8, 1.0, 1.2):
        result = _composer(
            shoulder_width=0.30 * person_scale,
            torso=0.41 * person_scale,
        ).compose(frame, _source(), 1.0)
        bbox, area = _garment_bbox_and_area(_decoded(result))
        measurements.append((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, area))

    widths, heights, areas = zip(*measurements)
    assert widths[0] < widths[1] < widths[2]
    assert heights[0] < heights[1] < heights[2]
    assert areas[0] < areas[1] < areas[2]
    assert widths[2] >= widths[0] * 1.35
    assert heights[2] >= heights[0] * 1.35
    assert areas[2] >= areas[0] * 1.8


def test_compose_fails_closed_when_the_declared_transparent_source_has_no_alpha_boundary():
    opaque = np.full((64, 64, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", opaque)
    assert ok
    png = encoded.tobytes()

    with pytest.raises(GarmentFetchError, match="transparent_boundary"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"),
            1.0,
        )


def test_compose_fails_closed_for_cropped_and_template_mismatched_sources():
    cropped = np.full((64, 64, 4), (20, 120, 220, 255), dtype=np.uint8)
    cropped[32, 32, 3] = 0
    ok, encoded = cv2.imencode(".png", cropped)
    assert ok
    png = encoded.tobytes()
    cropped_source = TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve")
    with pytest.raises(GarmentFetchError, match="garment_cropped"):
        _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), cropped_source, 1.0)

    short = _source()
    long_claim = TransparentGarmentSource(short.png_bytes, short.digest, "tshirt_long_sleeve")
    with pytest.raises(GarmentFetchError, match="template_mismatch"):
        _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), long_claim, 1.0)

    long = _source("tshirt_long_sleeve")
    short_claim = TransparentGarmentSource(long.png_bytes, long.digest, "tshirt_short_sleeve")
    with pytest.raises(GarmentFetchError, match="template_mismatch"):
        _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), short_claim, 1.0)


def test_compose_accepts_a_wide_constant_torso_short_source():
    result = _composer().compose(
        np.full((360, 480, 3), 180, dtype=np.uint8),
        _wide_torso_short_source(),
        1.0,
    )
    assert _decoded(result).shape == (360, 480, 3)


@pytest.mark.parametrize("high_alpha_pixel", (False, True))
def test_compose_rejects_low_confidence_subject_without_a_single_pixel_verdict(
    high_alpha_pixel,
):
    """全主体低 alpha 与仅一个越过 128 的像素都不是实质成衣主体。"""
    with pytest.raises(GarmentFetchError, match="garment_quality"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            _low_confidence_short_source(high_alpha_pixel=high_alpha_pixel),
            1.0,
        )


@pytest.mark.parametrize(
    ("fringe_alpha", "fringe_length"),
    ((127, 12), (128, 12), (220, 6), (255, 1), (255, 8)),
)
def test_compose_accepts_complete_short_source_with_a_small_edge_fringe(
    fringe_alpha, fringe_length
):
    """短小边缘像素不是主体被持续裁切的可观察证据。"""
    result = _composer().compose(
        np.full((360, 480, 3), 180, dtype=np.uint8),
        _antialiased_boundary_short_source(
            fringe_alpha=fringe_alpha,
            fringe_length=fringe_length,
        ),
        1.0,
    )

    assert _decoded(result).shape == (360, 480, 3)


def test_compose_rejects_solid_edge_material_even_with_a_low_alpha_tail():
    """近透明尾部不能稀释其前方已经成立的实心裁切核心。"""
    with pytest.raises(GarmentFetchError, match="garment_cropped"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            _right_edge_cropped_short_source(low_alpha_tail=True),
            1.0,
        )


@pytest.mark.parametrize("canvas_height", (140, 401))
def test_compose_cropped_verdict_is_unchanged_by_orthogonal_transparent_padding(
    canvas_height,
):
    """12px 右缘材料约占主体高度 11%，不随 PNG 高度 140→401 改变判定。"""
    with pytest.raises(GarmentFetchError, match="garment_cropped"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            _right_edge_cropped_short_source(canvas_height=canvas_height),
            1.0,
        )


def test_compose_preserves_fractional_alpha_blend_strength():
    """alpha=100 的公开合成强度必须严格位于 alpha=0 与 255 之间。"""
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    outputs = {
        alpha: _decoded(
            _composer(arms=False).compose(
                frame,
                _alpha_blend_probe_source(alpha),
                1.0,
            )
        )
        for alpha in (0, 100, 255)
    }

    assert not np.array_equal(outputs[0], outputs[100])
    assert not np.array_equal(outputs[100], outputs[255])
    assert not np.array_equal(outputs[0], outputs[255])
    affected_pixels = np.any(outputs[255] != outputs[0], axis=2)
    assert np.count_nonzero(affected_pixels) > 0
    increment_100 = np.abs(
        outputs[100].astype(np.int16) - outputs[0].astype(np.int16)
    )[affected_pixels].sum()
    increment_255 = np.abs(
        outputs[255].astype(np.int16) - outputs[0].astype(np.int16)
    )[affected_pixels].sum()
    observed_blend_ratio = float(increment_100) / float(increment_255)

    assert 0 < increment_100 < increment_255
    assert observed_blend_ratio == pytest.approx(100 / 255, abs=0.02)


def test_compose_uses_an_orthonormal_uniform_basis_for_asymmetric_torso_pose():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    # These hips make the shoulder-to-hip axis non-orthogonal to shoulders.
    # The visible, separately coloured source markers are the oracle: no
    # geometry facts from the compositor participate in the measurement.
    result = _decoded(
        _composer(arms=False, hips=((0.34, 0.76), (0.74, 0.52))).compose(
            frame, _marker_source(), 1.0
        )
    )
    red = _marker_centroid(result, (0, 0, 255))
    yellow = _marker_centroid(result, (0, 255, 255))
    cyan = _marker_centroid(result, (255, 255, 0))
    across = yellow - red
    down = cyan - red
    across_scale = np.linalg.norm(across) / 40.0
    down_scale = np.linalg.norm(down) / 50.0

    assert abs(float(np.dot(across, down))) <= np.linalg.norm(across) * np.linalg.norm(down) * 0.02
    assert abs(across_scale - down_scale) / max(across_scale, down_scale) <= 0.02
    # The source's 40:50 marker contour ratio survives the public PNG render.
    assert abs((np.linalg.norm(across) / np.linalg.norm(down)) - (40.0 / 50.0)) <= 0.02


@pytest.mark.parametrize("edge", ("top", "bottom", "left", "right"))
def test_compose_rejects_a_silhouette_cropped_at_any_canvas_edge(edge):
    image = np.zeros((72, 72, 4), dtype=np.uint8)
    image[12:60, 12:60] = (20, 120, 220, 255)
    if edge == "top":
        image[:36, 24:48] = (20, 120, 220, 255)
    elif edge == "bottom":
        image[36:, 24:48] = (20, 120, 220, 255)
    elif edge == "left":
        image[24:48, :36] = (20, 120, 220, 255)
    else:
        image[24:48, 36:] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()

    with pytest.raises(GarmentFetchError, match="garment_cropped"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"),
            1.0,
        )


def test_compose_rejects_empty_multiple_component_and_low_confidence_sources():
    for image, expected in (
        (np.zeros((72, 72, 4), dtype=np.uint8), "transparent_png"),
        (np.pad(np.full((12, 12, 4), (20, 120, 220, 255), dtype=np.uint8), ((8, 52), (8, 52), (0, 0))), "garment_quality"),
    ):
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        png = encoded.tobytes()
        with pytest.raises(GarmentFetchError, match=expected):
            _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"), 1.0)

    split = np.zeros((72, 72, 4), dtype=np.uint8)
    split[12:30, 12:30] = (20, 120, 220, 255)
    split[42:60, 42:60] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", split)
    assert ok
    png = encoded.tobytes()
    with pytest.raises(GarmentFetchError, match="garment_quality"):
        _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"), 1.0)


@pytest.mark.parametrize("gap", (1, 2))
def test_compose_rejects_components_before_boundary_noise_closing(gap):
    image = np.zeros((72, 72, 4), dtype=np.uint8)
    image[20:48, 14:32] = (20, 120, 220, 255)
    image[20:48, 32 + gap:50 + gap] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()

    with pytest.raises(GarmentFetchError, match="garment_quality"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"),
            1.0,
        )


@pytest.mark.parametrize("kind", ("no_sleeves", "left_only", "right_only", "noise"))
def test_compose_rejects_short_template_without_bilateral_sleeve_evidence(kind):
    image = np.zeros((140, 180, 4), dtype=np.uint8)
    image[22:128, 30:150] = (20, 120, 220, 255)
    if kind == "left_only":
        image[38:82, 8:35] = (20, 120, 220, 255)
    elif kind == "right_only":
        image[38:82, 145:172] = (20, 120, 220, 255)
    elif kind == "noise":
        image[38:42, 8:35] = (20, 120, 220, 255)
        image[78:82, 145:172] = (20, 120, 220, 255)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    png = encoded.tobytes()

    with pytest.raises(GarmentFetchError, match="garment_quality"):
        _composer().compose(
            np.full((360, 480, 3), 180, dtype=np.uint8),
            TransparentGarmentSource(png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"),
            1.0,
        )


def test_compose_reports_the_verified_source_digest_deterministically():
    source = _source()
    first = _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), source, 1.0)
    second = _composer().compose(np.full((360, 480, 3), 180, dtype=np.uint8), source, 1.0)

    assert first.geometry.garment_digest == source.digest
    assert second.geometry.garment_digest == source.digest


def test_compose_rejects_pose_rotation_outside_the_limited_range():
    pose = _pose()
    pose.pose_landmarks.landmark[11] = SimpleNamespace(x=0.35, y=0.20, visibility=0.95)
    pose.pose_landmarks.landmark[12] = SimpleNamespace(x=0.65, y=0.55, visibility=0.95)

    with pytest.raises(PoseUnavailableError, match="pose_unavailable"):
        GarmentComposer(pose_estimator=_FixturePoseEstimator(pose)).compose(
            np.full((360, 480, 3), 180, dtype=np.uint8), _source(), 1.0
        )


def test_compose_short_sleeves_survive_while_bare_forearms_remain_foreground():
    bare_frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    # Independent public-compose expectation: this pose deliberately has no
    # arm landmarks, therefore it cannot describe a camera foreground arm.
    baseline = _decoded(_composer(arms=False).compose(bare_frame, _source(), 1.0))
    frame = bare_frame.copy()
    # Flesh-coloured lower arms deliberately lie below the short sleeve ends.
    cv2.line(frame, (134, 172), (120, 238), (30, 170, 240), 18)
    cv2.line(frame, (346, 172), (360, 238), (30, 170, 240), 18)
    result = _decoded(_composer().compose(frame, _source(), 1.0))

    red_retention = _colour_mask(result, (0, 0, 255)).sum() / _colour_mask(baseline, (0, 0, 255)).sum()
    blue_retention = _colour_mask(result, (255, 0, 0)).sum() / _colour_mask(baseline, (255, 0, 0)).sum()
    skin = _colour_mask(frame, (30, 170, 240))
    skin_retention = np.count_nonzero(_colour_mask(result, (30, 170, 240)) & skin) / np.count_nonzero(skin)
    assert red_retention >= 0.65
    assert blue_retention >= 0.65
    assert min(red_retention, blue_retention) / max(red_retention, blue_retention) >= 0.8
    assert skin_retention >= 0.95


def test_compose_crossed_hands_and_long_sleeves_follow_template_occlusion():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    cv2.line(frame, (259, 162), (312, 194), (30, 170, 240), 18)
    cv2.line(frame, (221, 162), (168, 194), (30, 170, 240), 18)
    result = _decoded(_composer(crossed=True).compose(frame, _source("tshirt_long_sleeve"), 1.0))

    assert _colour_area(result, 2) >= 1_300
    assert _colour_area(result, 0) >= 1_300
    crossed_roi = np.s_[150:210, 150:330]
    skin = np.all(frame[crossed_roi] == (30, 170, 240), axis=2)
    assert np.count_nonzero(np.all(result[crossed_roi] == frame[crossed_roi], axis=2) & skin) >= np.count_nonzero(skin) * 0.90


def test_compose_long_sleeves_reach_the_wrist_but_leave_hands_in_front():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    cv2.line(frame, (134, 173), (120, 238), (30, 170, 240), 18)
    cv2.line(frame, (346, 173), (360, 238), (30, 170, 240), 18)
    cv2.circle(frame, (119, 244), 10, (30, 170, 240), -1)
    cv2.circle(frame, (361, 244), 10, (30, 170, 240), -1)
    result = _decoded(_composer().compose(frame, _source("tshirt_long_sleeve"), 1.0))

    wrist_bands = (result[205:238, 95:150], result[205:238, 330:385])
    assert sum(np.count_nonzero(_colour_mask(band, (0, 0, 255))) + np.count_nonzero(_colour_mask(band, (255, 0, 0))) for band in wrist_bands) >= 150
    assert np.array_equal(result[244, 119], frame[244, 119])


def test_compose_scale_changes_decoded_garment_dimensions_without_moving_center():
    frame = np.full((360, 480, 3), 180, dtype=np.uint8)
    full = _composer().compose(frame, _source(), 1.0)
    adjusted = _composer().compose(frame, _source(), 1.05)

    assert adjusted.geometry.width >= full.geometry.width * 1.03
    assert adjusted.geometry.height >= full.geometry.height * 1.03
    assert np.linalg.norm(np.subtract(adjusted.geometry.center, full.geometry.center)) <= 2
    full_bbox = _green_torso_bbox(_decoded(full))
    adjusted_bbox = _green_torso_bbox(_decoded(adjusted))
    full_width, full_height = full_bbox[2] - full_bbox[0] + 1, full_bbox[3] - full_bbox[1] + 1
    adjusted_width, adjusted_height = adjusted_bbox[2] - adjusted_bbox[0] + 1, adjusted_bbox[3] - adjusted_bbox[1] + 1
    assert adjusted_width >= full_width * 1.03
    assert adjusted_height >= full_height * 1.03
