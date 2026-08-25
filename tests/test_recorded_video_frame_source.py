import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from statistics import median

import cv2
import numpy as np
import pytest

import app as vision_app
from vision import camera_manager
from vision.config import settings
from vision.acquisition_session import AcquisitionSession
from vision.acquisition_observer import _observe_frame
from vision.garment_composer import GarmentComposer, TransparentGarmentSource
from vision.person_detector import PersonDetector
from vision.pose_estimator import PoseEstimator
from vision.profile_state import protocol_occupancy_snapshot
from vision.proximity import ProximityMonitor
from vision.try_on_attempt_registry import TryOnAttemptRegistry
from vision.frame_source import RecordedVideoFrameSource
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_occupancy_gate, reset_active_track
from vision.self_check import check_camera
from vision.vision_pipeline import VisionPipeline


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "recorded-video"


def fixture_manifest():
    return json.loads((FIXTURE_ROOT / "expected-results.json").read_text())


def configure_recorded_sources(monkeypatch, manifest):
    for setting_name, role in (("TOP_CAMERA_CONFIG", "top"), ("FRONT_CAMERA_CONFIG", "front")):
        recording = manifest["recordings"][role]
        monkeypatch.setattr(
            settings,
            setting_name,
            {
                "role": "presence" if role == "top" else "profile_try_on",
                "source": "recorded_video",
                "video_path": str(FIXTURE_ROOT / recording["file"]),
                "loop": recording["loop"],
                "rotate": 0,
            },
        )
    camera_manager.release_all_cameras()


def write_managed_recorded_video_config(path, manifest, *, front_loop=None):
    cameras = {}
    for role in ("top", "front"):
        recording = manifest["recordings"][role]
        cameras[role] = {
            "role": "presence" if role == "top" else "profile_try_on",
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / recording["file"]),
            "loop": recording["loop"] if role == "top" or front_loop is None else front_loop,
            "rotate": 0,
        }
    content = {
        "schemaVersion": "vending-vision-site-config/v1",
        "host": "127.0.0.1",
        "port": 7892,
        "allowed_origins": ["http://127.0.0.1:7892"],
        "cameras": cameras,
    }
    config_bytes = (json.dumps(content, indent=2) + "\n").encode()
    path.write_bytes(config_bytes)
    return hashlib.sha256(config_bytes).hexdigest()


def reset_presence_state():
    gate = get_occupancy_gate()
    for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
        gate.mark_absent()
    reset_active_track()


def assert_recorded_video_source_frame(frame, *, role, video_sha256, config_sha256):
    assert frame["adapter"] == "recorded_video"
    assert frame["role"] == role
    assert frame["fixtureSha256"] == video_sha256
    assert frame["configSha256"] == config_sha256
    assert isinstance(frame["frameIndex"], int)
    assert isinstance(frame["decodedFrameCount"], int)
    assert 0 <= frame["frameIndex"] < frame["decodedFrameCount"]
    assert frame["synthetic"] is False
    assert frame["relabeled"] is False


def test_recorded_video_fixture_manifest_binds_top_and_front_recordings():
    manifest = fixture_manifest()

    assert manifest["schemaVersion"] == "vending-vision-recorded-video-fixture/v1"
    assert set(manifest["recordings"]) == {
        "top", "front", "manFront", "geometryFar", "geometryMid", "geometryNear",
        "frontVertical", "frontVerticalUnstable", "manUnalignedFront", "emptyFront",
        "fieldRecommendationNearTop", "fieldRecommendationNearFront",
        "fieldRecommendationFarTop", "fieldRecommendationFarFront",
    }
    for role, recording in manifest["recordings"].items():
        video = FIXTURE_ROOT / recording["file"]
        assert video.is_file(), role
        assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
        assert isinstance(recording["loop"], bool)
    assert manifest["expected"]["top"]["protocolEvents"] == [
        "vision.presence_status",
        "vision.person_departed",
    ]
    assert manifest["expected"]["front"]["profile"]["minimumFields"]


def test_field_recommendation_fixtures_are_traceable_and_drive_stable_production_facts():
    """The authorized near/far field captures exercise real presence and profile models."""
    manifest = fixture_manifest()
    pipeline = VisionPipeline()
    body_types = []

    for distance in ("Near", "Far"):
        monitor = ProximityMonitor()
        top_recording = manifest["recordings"][f"fieldRecommendation{distance}Top"]
        front_recording = manifest["recordings"][f"fieldRecommendation{distance}Front"]
        for recording, expected_shape in (
            (top_recording, (1080, 1920, 3)),
            (front_recording, (1920, 1080, 3)),
        ):
            source = FIXTURE_ROOT / recording["source"]
            video = FIXTURE_ROOT / recording["file"]
            assert source.is_file()
            assert (FIXTURE_ROOT / recording["generator"]).is_file()
            assert recording["sourceSha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
            assert recording["loop"] is True
            capture = cv2.VideoCapture(str(video))
            ok, frame = capture.read()
            capture.release()
            assert ok and frame.shape == expected_shape

        top_capture = cv2.VideoCapture(str(FIXTURE_ROOT / top_recording["file"]))
        top_observations = []
        for _ in range(4):
            ok, frame = top_capture.read()
            assert ok
            top_observations.append(monitor.check_image(frame))
        top_capture.release()
        assert protocol_occupancy_snapshot(top_observations[-1])["state"] == "single"

        front_capture = cv2.VideoCapture(str(FIXTURE_ROOT / front_recording["file"]))
        ok, frame = front_capture.read()
        front_capture.release()
        assert ok
        profile = pipeline.infer(frame)
        assert profile.presence is True
        assert profile.body_type != "unknown"
        body_types.append(profile.body_type)

    assert body_types[0] == body_types[1]


class _DetectedPoseEstimator:
    """Reuse one production detection while exercising every compose call."""

    def __init__(self, pose):
        self._pose = pose

    def detect(self, _frame):
        return self._pose


def _decoded_garment_change(frame, result):
    rendered = cv2.imdecode(
        np.frombuffer(result.png, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert rendered is not None and rendered.shape == frame.shape
    changed = (np.max(cv2.absdiff(rendered, frame), axis=2) >= 30).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(changed, 8)
    assert count > 1
    garment_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == garment_component


def _field_front_source(manifest, distance):
    recording = manifest["recordings"][f"fieldRecommendation{distance.title()}Front"]
    source = FIXTURE_ROOT / recording["source"]
    assert recording["sourceSha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
    assert raw is not None and raw.shape == (480, 640, 3)
    return cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _field_garment_source(manifest):
    asset = manifest["assets"]["tryOnSilhouette"]
    path = FIXTURE_ROOT / asset["file"]
    payload = path.read_bytes()
    assert asset["sha256"] == hashlib.sha256(payload).hexdigest()
    return TransparentGarmentSource(
        payload,
        "sha256:" + asset["sha256"],
        asset["template"],
    )


def test_field_near_far_default_garment_matches_confirmed_visible_geometry():
    """The public 100% result, not a hidden product branch, owns field geometry."""
    manifest = fixture_manifest()
    garment = _field_garment_source(manifest)
    observations = {}

    for distance in ("near", "far"):
        frame = _field_front_source(manifest, distance)
        production_pose = PoseEstimator().detect(frame)
        assert production_pose.pose_landmarks is not None
        composer = GarmentComposer(
            pose_estimator=_DetectedPoseEstimator(production_pose)
        )
        result = composer.compose(frame, garment, 1.0)
        expected = manifest["expected"]["fieldGarmentGeometry"][distance]

        assert result.geometry.width == pytest.approx(expected["width"], rel=0.03)
        assert result.geometry.height == pytest.approx(expected["height"], rel=0.03)
        assert result.geometry.placed_aspect_ratio == pytest.approx(
            result.geometry.source_aspect_ratio, rel=1e-6
        )

        changed = _decoded_garment_change(frame, result)
        ys, xs = np.nonzero(changed)
        top = int(ys.min())
        center_x = round(result.geometry.center[0])
        center_strip = changed[:, max(0, center_x - 8) : center_x + 9]
        neckline_y = int(np.nonzero(center_strip)[0].min())
        neckline_offset = (neckline_y - top) / result.geometry.height

        sleeve_band_end = min(
            changed.shape[0], round(neckline_y + result.geometry.height * 0.18)
        )
        sleeve_band = changed[top:sleeve_band_end]
        left_pixels = int(np.count_nonzero(sleeve_band[:, :center_x]))
        right_pixels = int(np.count_nonzero(sleeve_band[:, center_x:]))
        assert min(left_pixels, right_pixels) >= 4_000
        assert min(left_pixels, right_pixels) / max(left_pixels, right_pixels) >= 0.70
        assert 0.06 <= neckline_offset <= 0.16

        scaled = [composer.compose(frame, garment, scale) for scale in (1.0, 1.05, 1.10)]
        assert all(
            np.linalg.norm(np.subtract(item.geometry.center, scaled[0].geometry.center))
            <= 2
            for item in scaled[1:]
        )
        assert [item.geometry.width for item in scaled] == sorted(
            item.geometry.width for item in scaled
        )
        assert [item.geometry.height for item in scaled] == sorted(
            item.geometry.height for item in scaled
        )
        observations[distance] = neckline_offset

    assert abs(observations["near"] - observations["far"]) <= 0.05


def test_recorded_video_geometry_fixtures_are_distinct_dynamic_and_traceable():
    """The public fixture manifest exposes three reproducible live front clips."""
    manifest = fixture_manifest()
    source_sha = hashlib.sha256(
        (FIXTURE_ROOT / "sources" / "person-man-front.png").read_bytes()
    ).hexdigest()
    recordings = [manifest["recordings"][name] for name in ("geometryFar", "geometryMid", "geometryNear")]

    assert len({recording["file"] for recording in recordings}) == 3
    assert len({recording["sha256"] for recording in recordings}) == 3
    for recording in recordings:
        assert recording["loop"] is True
        assert recording["source"] == "sources/person-man-front.png"
        assert recording["sourceSha256"] == source_sha
        assert (FIXTURE_ROOT / recording["generator"]).is_file()
        video = FIXTURE_ROOT / recording["file"]
        assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
        capture = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        assert len(frames) >= 24  # >=4 seconds at the documented 6fps.
        assert len({hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames}) >= 2


def _preview_dom_hash(frame):
    """Mirror the CDP preview rect normalization before hashing its pixels."""
    height, width = frame.shape[:2]
    scale = min(1, 16 / max(width, height))
    preview = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return hashlib.sha256(preview.tobytes()).hexdigest()


def test_recorded_video_geometry_preview_remains_live_after_cdp_rect_normalization():
    """Every countdown bucket carries two distinct renderer-sized preview identities."""
    for name in ("geometryFar", "geometryMid", "geometryNear"):
        video = FIXTURE_ROOT / fixture_manifest()["recordings"][name]["file"]
        capture = cv2.VideoCapture(str(video))
        hashes = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            hashes.append(_preview_dom_hash(frame))
        capture.release()
        # Six fps yields twelve source frames for each of the three 2-second
        # countdown buckets. The probe intentionally uses the actual 16px
        # CDP normalization, not source-frame hashes.
        assert len(hashes) == 36
        assert all(len(set(hashes[offset : offset + 12])) >= 2 for offset in (0, 12, 24)), name


def test_recorded_video_geometry_fixtures_pass_the_production_observer_with_monotonic_shoulders():
    """Every decoded marker phase remains one aligned person at its fixed scale."""
    from vision.acquisition_observer import _observe_frame
    from vision.person_detector import PersonDetector
    from vision.pose_estimator import PoseEstimator

    detector = PersonDetector()
    estimator = PoseEstimator()
    shoulder_spans = []
    for name in ("geometryFar", "geometryMid", "geometryNear"):
        video = FIXTURE_ROOT / fixture_manifest()["recordings"][name]["file"]
        capture = cv2.VideoCapture(str(video))
        spans = []
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            observation = _observe_frame(detector, estimator, frame)
            assert observation.occupancy == "single", (name, frames)
            assert observation.aligned is True, (name, frames)
            landmarks = estimator.detect(frame).pose_landmarks.landmark
            spans.append(abs(landmarks[11].x - landmarks[12].x))
        capture.release()
        assert frames == 36
        # Marker luminance spans its complete cycle yet must not perturb pose.
        assert max(spans) - min(spans) < 0.03
        shoulder_spans.append(median(spans))
    assert shoulder_spans[0] < shoulder_spans[1] < shoulder_spans[2]
    assert min(shoulder_spans[1] - shoulder_spans[0], shoulder_spans[2] - shoulder_spans[1]) > 0.05


def test_recorded_video_geometry_generator_is_byte_reproducible(tmp_path):
    """The generator is deterministic without overwriting the committed fixture."""
    root = FIXTURE_ROOT
    generator = root / "generate-geometry-front.py"
    before = {path.name: path.read_bytes() for path in root.glob("geometry-*.mp4")}
    command = [
        sys.executable,
        str(generator),
        "--output-directory",
        str(tmp_path),
    ]
    subprocess.run(command, cwd=root.parents[1], check=True)
    after_first = {path.name: path.read_bytes() for path in tmp_path.glob("geometry-*.mp4")}
    subprocess.run(command, cwd=root.parents[1], check=True)
    after_second = {path.name: path.read_bytes() for path in tmp_path.glob("geometry-*.mp4")}

    assert set(after_first) == set(before)
    assert after_first == after_second
    assert before == {
        path.name: path.read_bytes() for path in root.glob("geometry-*.mp4")
    }


class _RecordedAcquisitionCamera:
    """Public AcquisitionSession camera adapter backed by the production decoder."""

    def __init__(self, video):
        self._source = RecordedVideoFrameSource(
            role="front",
            config={"source": "recorded_video", "video_path": str(video), "loop": True},
        )

    async def acquire(self, attempt_id, deadline):
        return attempt_id

    async def read(self, _lease_token, _timeout):
        return self._source.read(), self._source.last_frame() or {}

    async def release(self, _attempt_id, _lease_token):
        self._source.release()


class _ProductionAcquisitionObserver:
    def __init__(self):
        self._detector = PersonDetector()
        self._pose = PoseEstimator()

    async def observe(self, frame, timeout):
        return _observe_frame(self._detector, self._pose, frame)

    async def wait_idle(self):
        return None


class _DiscardedPreview:
    async def open(self, _attempt_id, _jpeg):
        return "preview"

    async def update(self, _attempt_id, _token, _jpeg):
        return True

    async def close(self, _attempt_id):
        return None


def _semantic_geometry_garment():
    """A distinct magenta shirt silhouette makes decoded result pixels observable."""
    garment = np.zeros((160, 180, 4), dtype=np.uint8)
    garment[24:148, 42:138] = (255, 0, 255, 255)
    garment[38:88, 12:46] = (255, 0, 255, 255)
    garment[38:88, 134:168] = (255, 0, 255, 255)
    ok, encoded = cv2.imencode(".png", garment)
    assert ok
    png = encoded.tobytes()
    return TransparentGarmentSource(
        png, "sha256:" + hashlib.sha256(png).hexdigest(), "tshirt_short_sleeve"
    )


def _magenta_bbox(result_png):
    rendered = cv2.imdecode(np.frombuffer(result_png, dtype=np.uint8), cv2.IMREAD_COLOR)
    mask = (rendered[:, :, 0] >= 180) & (rendered[:, :, 1] <= 80) & (rendered[:, :, 2] >= 180)
    points = cv2.findNonZero(mask.astype(np.uint8))
    assert points is not None
    return cv2.boundingRect(points)


def _assert_significant_geometry_steps(measurements):
    widths, heights = zip(*measurements)
    assert widths[1] >= widths[0] * 1.08
    assert widths[2] >= widths[1] * 1.08
    assert heights[1] >= heights[0] * 1.08
    assert heights[2] >= heights[1] * 1.08


def test_recorded_geometry_acquisition_capture_drives_production_composer_monotonically():
    """Real captured frames, not labels or resized outputs, determine garment scale."""
    async def scenario():
        observer = _ProductionAcquisitionObserver()
        composer = GarmentComposer(pose_estimator=PoseEstimator())
        garment = _semantic_geometry_garment()
        measurements = []
        for name in ("geometryFar", "geometryMid", "geometryNear"):
            video = FIXTURE_ROOT / fixture_manifest()["recordings"][name]["file"]
            session = AcquisitionSession(
                attempt_id=name,
                camera=_RecordedAcquisitionCamera(video),
                observer=observer,
                preview=_DiscardedPreview(),
                publish=lambda *_fact: asyncio.sleep(0),
                stable_seconds=0,
                timeout_seconds=4,
                preview_interval_seconds=0.01,
            )
            captured = await session.acquire(
                manual_requested=lambda: asyncio.sleep(0, result=False),
                consume_manual=lambda: asyncio.sleep(0),
            )
            rendered = composer.compose(captured.frame, garment, 1.0)
            _x, _y, width, height = _magenta_bbox(rendered.png)
            measurements.append((width, height))
        _assert_significant_geometry_steps(measurements)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "mutated",
    (
        ((100, 100), (101, 109), (126, 126)),
        ((100, 100), (109, 109), (110, 126)),
    ),
    ids=("one-pixel-far-mid-step", "only-far-near-is-significant"),
)
def test_recorded_geometry_assertion_rejects_insignificant_adjacent_steps(mutated):
    """One-pixel and skipped-middle mutations cannot satisfy the geometry journey."""
    with pytest.raises(AssertionError):
        _assert_significant_geometry_steps(mutated)


def test_recorded_video_front_vertical_fixture_is_traceable_and_vertical():
    """The vertical close-up fixture is reproducible and 1080x1920 with a single aligned person."""
    recording = fixture_manifest()["recordings"]["frontVertical"]
    video = FIXTURE_ROOT / recording["file"]
    generator = FIXTURE_ROOT / recording["generator"]
    assert video.is_file()
    assert generator.is_file()
    assert recording["sourceSha256"] == hashlib.sha256(
        (FIXTURE_ROOT / recording["source"]).read_bytes()
    ).hexdigest()
    assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (1920, 1080, 3)
    assert fixture_manifest()["expected"]["frontVertical"]["profile"]["minimumFields"]


def test_recorded_video_front_vertical_unstable_fixture_is_traceable():
    """The unstable vertical fixture never reaches a stable aligned run."""
    recording = fixture_manifest()["recordings"]["frontVerticalUnstable"]
    video = FIXTURE_ROOT / recording["file"]
    generator = FIXTURE_ROOT / recording["generator"]
    assert video.is_file()
    assert generator.is_file()
    assert recording["sourceSha256"] == hashlib.sha256(
        (FIXTURE_ROOT / recording["source"]).read_bytes()
    ).hexdigest()
    assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (1920, 1080, 3)


def test_recorded_video_man_front_fixture_is_traceable_and_decodes():
    """The acquisition fixture is reproducible from the declared fictional input."""
    recording = fixture_manifest()["recordings"]["manFront"]
    video = FIXTURE_ROOT / recording["file"]
    generator = FIXTURE_ROOT / recording["generator"]
    assert video.is_file()
    assert generator.is_file()
    assert recording["sourceSha256"] == hashlib.sha256(
        (FIXTURE_ROOT / recording["source"]).read_bytes()
    ).hexdigest()
    assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (1920, 1080, 3)


def test_recorded_video_source_decodes_fixture_frames_in_order():
    source = RecordedVideoFrameSource(
        role="top",
        config={"source": "recorded_video", "video_path": str(FIXTURE_ROOT / "top.mp4"), "loop": False},
    )
    expected_capture = cv2.VideoCapture(str(FIXTURE_ROOT / "top.mp4"))
    ok, expected_first = expected_capture.read()
    expected_capture.release()
    assert ok

    first = source.read()
    second = source.read()

    assert first.shape == second.shape
    assert first.size > 0
    assert np.array_equal(first, expected_first)
    assert source.status()["ready"] is True
    assert source.status()["source"] == "recorded_video"
    source.release()


def test_recorded_video_source_reports_corrupt_and_zero_frame_inputs_not_ready(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    zero_frames = tmp_path / "zero-frames.mp4"
    writer = cv2.VideoWriter(str(zero_frames), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 32))
    assert writer.isOpened()
    writer.release()

    for path in (corrupt, zero_frames):
        source = RecordedVideoFrameSource(
            role="top", config={"source": "recorded_video", "video_path": str(path), "loop": False}
        )
        with pytest.raises(RuntimeError):
            source.read()
        status = source.status()
        assert status["ok"] is False
        assert status["ready"] is False
        assert status["lastError"]


def test_recorded_video_source_exhaustion_and_loop_semantics_are_explicit():
    non_looping = RecordedVideoFrameSource(
        role="front", config={"source": "recorded_video", "video_path": str(FIXTURE_ROOT / "front.mp4"), "loop": False}
    )
    frame_total = int(cv2.VideoCapture(str(FIXTURE_ROOT / "front.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frame_total):
        non_looping.read()
    frozen = non_looping.read()
    assert non_looping.status()["exhausted"] is True
    assert non_looping.status()["ready"] is True
    assert non_looping.status()["ok"] is True
    assert np.array_equal(frozen, non_looping.read())

    looping = RecordedVideoFrameSource(
        role="front", config={"source": "recorded_video", "video_path": str(FIXTURE_ROOT / "front.mp4"), "loop": True}
    )
    first = looping.read()
    for _ in range(frame_total - 1):
        looping.read()
    assert np.array_equal(looping.read(), first)
    assert looping.status()["ready"] is True


def test_health_camera_check_accepts_both_recorded_video_sources(monkeypatch):
    configure_recorded_sources(monkeypatch, fixture_manifest())

    status = check_camera()

    assert status["ok"] is True
    assert status["detail"]["sources"] == {"top": "recorded_video", "front": "recorded_video"}
    monkeypatch.setattr(vision_app, "startup_check", None)
    assert vision_app.health()["cameraReady"] is True


def test_read_camera_records_recorded_video_source_metadata(monkeypatch, tmp_path):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    managed_config_sha = write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))
    configure_recorded_sources(monkeypatch, manifest)

    camera_manager.read_camera("top", warmup_frames=1)
    assert_recorded_video_source_frame(
        camera_manager.get_last_frame_source("top"), role="top",
        video_sha256=manifest["recordings"]["top"]["sha256"], config_sha256=managed_config_sha,
    )
    camera_manager.read_camera("front", warmup_frames=1)
    assert_recorded_video_source_frame(
        camera_manager.get_last_frame_source("front"), role="front",
        video_sha256=manifest["recordings"]["front"]["sha256"], config_sha256=managed_config_sha,
    )


def test_recorded_video_metadata_is_omitted_when_managed_camera_config_does_not_match(monkeypatch, tmp_path):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    write_managed_recorded_video_config(managed_config, manifest, front_loop=not manifest["recordings"]["front"]["loop"])
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))
    configure_recorded_sources(monkeypatch, manifest)

    _, source_frame = camera_manager.read_camera_with_source("front", warmup_frames=1)

    assert source_frame is None


def test_recorded_video_drives_real_presence_profile_and_departure(monkeypatch, tmp_path):
    manifest = fixture_manifest()
    expected = manifest["expected"]
    managed_config = tmp_path / "managed-site-config.json"
    config_sha = write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))
    configure_recorded_sources(monkeypatch, manifest)
    reset_presence_state()
    runtime = PresenceRuntime()
    monkeypatch.setattr(vision_app, "get_presence_runtime", lambda: runtime)
    published = []

    async def record_update(update):
        published.append(update)

    monkeypatch.setattr(vision_app, "broadcast_profile_update", record_update)
    first = runtime.poll(include_status=True, include_ambient_light=False, include_departure=True)
    assert first.update["message_type"] == expected["top"]["protocolEvents"][0]
    assert_recorded_video_source_frame(
        first.update["payload"]["sourceFrame"], role="top",
        video_sha256=manifest["recordings"]["top"]["sha256"], config_sha256=config_sha,
    )
    asyncio.run(vision_app.profile_collection_worker(first.candidate, threading.Event()))
    profile_update = next(item for item in published if item["message_type"] == "vision.profile_result")
    assert set(expected["front"]["profile"]["minimumFields"]).issubset(profile_update["payload"]["profile"])
    assert_recorded_video_source_frame(
        profile_update["payload"]["sourceFrame"], role="front",
        video_sha256=manifest["recordings"]["front"]["sha256"], config_sha256=config_sha,
    )
    for _ in range(expected["top"]["departureWithinPolls"]):
        result = runtime.poll(include_status=True, include_ambient_light=False, include_departure=True)
        if result.update and result.update["message_type"] == "vision.person_departed":
            assert result.snapshot["occupancy"]["state"] == "none"
            return
    pytest.fail("recorded top source did not produce a departure edge")


def test_recorded_top_departure_leaves_cancellation_to_the_stable_machine_owner(
    monkeypatch, tmp_path
):
    """Recorded top departure is one raw fact, not a second business debounce."""
    async def scenario():
        manifest = fixture_manifest()
        expected = manifest["expected"]
        managed_config = tmp_path / "managed-site-config.json"
        write_managed_recorded_video_config(managed_config, manifest)
        monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))
        configure_recorded_sources(monkeypatch, manifest)
        reset_presence_state()
        runtime = PresenceRuntime()
        registry = TryOnAttemptRegistry(terminal_ttl_seconds=60)
        monkeypatch.setattr(vision_app, "get_presence_runtime", lambda: runtime)
        monkeypatch.setattr(vision_app, "_try_on_attempt_registry", registry)
        attempt_id = "550e8400-e29b-41d4-a716-446655440124"

        async def active_owner():
            await asyncio.Event().wait()

        owner_task = asyncio.create_task(active_owner())
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=owner_task,
            accepted={
                "type": "vision.try_on.attempt.accepted",
                "messageId": "accepted",
                "payload": {"attemptId": attempt_id},
            },
            generating={
                "type": "vision.try_on.attempt.generating",
                "messageId": "generating",
                "payload": {"attemptId": attempt_id, "stage": "preparing"},
            },
        )
        assert admission.is_owner

        departure = None
        for _ in range(expected["top"]["departureWithinPolls"]):
            result = runtime.poll(
                include_status=True,
                include_ambient_light=True,
                include_departure=True,
            )
            if result.update and result.update["message_type"] == "vision.person_departed":
                departure = result.update
                break

        assert departure is not None
        assert await registry.active_attempt_id() == attempt_id
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert replay.is_owner is False
        assert [message["type"] for message in replay.replay] == [
            "vision.try_on.attempt.accepted",
            "vision.try_on.attempt.generating",
        ]
        assert admission.receipt is not None
        await registry.cancel_owner_and_join(
            admission.receipt,
            {
                "type": "vision.try_on.attempt.canceled",
                "messageId": "disconnected",
                "payload": {"attemptId": attempt_id, "reason": "disconnect"},
            },
        )
        assert await registry.active_attempt_id() is None

    asyncio.run(scenario())
