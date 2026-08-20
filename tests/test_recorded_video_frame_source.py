import asyncio
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

import app as vision_app
from vision import camera_manager
from vision.config import settings
from vision.try_on_attempt_registry import TryOnAttemptRegistry
from vision.frame_source import RecordedVideoFrameSource
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_occupancy_gate, reset_active_track
from vision.self_check import check_camera


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
        "top", "front", "manFront", "frontVertical", "frontVerticalUnstable", "manUnalignedFront", "emptyFront"
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


def test_recorded_video_front_vertical_fixture_is_traceable_and_vertical():
    """The vertical close-up fixture is reproducible and 720x1280 with a single aligned person."""
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
    assert ok and frame.shape == (1280, 720, 3)
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
    assert ok and frame.shape == (1280, 720, 3)


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
    assert ok and frame.shape == (768, 512, 3)


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


def test_recorded_top_departure_cancels_active_public_attempt_once(monkeypatch, tmp_path):
    """Production recorded top departure fences the active V2 attempt as departure."""
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
                await vision_app._try_on_attempt_runtime_module.cancel_active(
                    "departure"
                )
                break

        assert departure is not None
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert len(replay.replay) == 1
        assert replay.replay[0]["type"] == "vision.try_on.attempt.canceled"
        assert replay.replay[0]["payload"] == {
            "attemptId": attempt_id,
            "reason": "departure",
        }
        assert await registry.active_attempt_id() is None
        owner_task.cancel()
        await asyncio.gather(owner_task, return_exceptions=True)

    asyncio.run(scenario())
