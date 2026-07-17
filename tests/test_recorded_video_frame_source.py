import asyncio
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

import app as vision_app
from vision.frame_source import RecordedVideoFrameSource
from vision import camera_manager
from vision.config import settings
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_occupancy_gate, reset_active_track
from vision.self_check import check_camera
from vision.try_on_session import iter_try_on_mjpeg, start_try_on_session, stop_try_on_session


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
                "role": "presence" if role == "top" else "profile_tryon",
                "source": "recorded_video",
                "video_path": str(FIXTURE_ROOT / recording["file"]),
                "loop": recording["loop"],
                "rotate": 0,
            },
        )
    camera_manager.release_all_cameras()


def reset_presence_state():
    gate = get_occupancy_gate()
    for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
        gate.mark_absent()
    reset_active_track()


def test_recorded_video_fixture_manifest_binds_top_and_front_recordings():
    manifest = fixture_manifest()

    assert manifest["schemaVersion"] == "vending-vision-recorded-video-fixture/v1"
    assert set(manifest["recordings"]) == {"top", "front"}
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
    assert manifest["expected"]["front"]["tryOn"]["jpeg"] is True


def test_recorded_video_source_decodes_fixture_frames_in_order():
    source = RecordedVideoFrameSource(
        role="top",
        config={
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / "top.mp4"),
            "loop": False,
        },
    )

    # Opening probes a decodable frame, then resets so the first production
    # read remains frame zero.
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
    writer = cv2.VideoWriter(
        str(zero_frames), cv2.VideoWriter_fourcc(*"mp4v"), 6, (32, 32),
    )
    assert writer.isOpened()
    writer.release()

    for path in (corrupt, zero_frames):
        source = RecordedVideoFrameSource(
            role="top",
            config={"source": "recorded_video", "video_path": str(path), "loop": False},
        )
        with pytest.raises(RuntimeError):
            source.read()
        status = source.status()
        assert status["ok"] is False
        assert status["ready"] is False
        assert status["lastError"]


def test_recorded_video_source_exhaustion_and_loop_semantics_are_explicit():
    non_looping = RecordedVideoFrameSource(
        role="front",
        config={"source": "recorded_video", "video_path": str(FIXTURE_ROOT / "front.mp4"), "loop": False},
    )
    frame_total = int(cv2.VideoCapture(str(FIXTURE_ROOT / "front.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frame_total):
        non_looping.read()
    with pytest.raises(RuntimeError, match="exhausted"):
        non_looping.read()
    exhausted = non_looping.status()
    assert exhausted["exhausted"] is True
    assert exhausted["ready"] is False
    assert "exhausted" in exhausted["lastError"]

    looping = RecordedVideoFrameSource(
        role="front",
        config={"source": "recorded_video", "video_path": str(FIXTURE_ROOT / "front.mp4"), "loop": True},
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
    health = vision_app.health()
    assert health["cameraReady"] is True
    assert health["modelReady"] is True


def test_read_camera_uses_recorded_video_source_without_directshow_binding(monkeypatch):
    monkeypatch.setattr(
        settings,
        "TOP_CAMERA_CONFIG",
        {
            "role": "presence",
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / "top.mp4"),
            "loop": False,
            "rotate": 0,
        },
    )

    camera_manager.release_all_cameras()
    frame = camera_manager.read_camera("top", warmup_frames=1)

    assert frame.shape == (320, 320, 3)


def test_expected_manifest_drives_real_presence_profile_departure_and_try_on_pipeline(monkeypatch):
    manifest = fixture_manifest()
    expected = manifest["expected"]
    configure_recorded_sources(monkeypatch, manifest)
    reset_presence_state()
    monkeypatch.setattr(settings, "PROFILE_SAMPLING_CONFIG", {
        **settings.PROFILE_SAMPLING_CONFIG,
        "duration_sec": 0.1,
        "early_finish_after_sec": 0,
        "target_fps": 60,
        "min_good_frames": 1,
        "max_good_frames": 2,
    })
    monkeypatch.setattr(settings, "PROFILE_MIN_VALID_FRAMES", 1)
    monkeypatch.setattr(settings, "FRONT_CAMERA_PROFILE_SAMPLE_COUNT", 2)
    monkeypatch.setattr(settings, "FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS", 1)
    monkeypatch.setattr(settings, "PROFILE_FACE_VOTE_ENABLED", False)

    runtime = PresenceRuntime()
    monkeypatch.setattr(vision_app, "get_presence_runtime", lambda: runtime)
    published = []

    async def record_update(update):
        published.append(update)

    monkeypatch.setattr(vision_app, "broadcast_profile_update", record_update)
    first = runtime.poll(include_status=True, include_ambient_light=False, include_departure=True)
    approach_stage, departure_stage = expected["top"]["stages"]
    assert (first.candidate is not None) is approach_stage["candidate"]
    assert first.snapshot["occupancy"]["state"] == approach_stage["occupancy"]
    assert first.update["message_type"] == expected["top"]["protocolEvents"][0]

    asyncio.run(
        vision_app.profile_collection_worker(first.candidate, threading.Event())
    )

    profile_update = next(update for update in published if update["message_type"] == "vision.profile_result")
    profile = profile_update["payload"]["profile"]
    assert profile["personPresent"] is True
    assert set(expected["front"]["profile"]["minimumFields"]).issubset(profile)
    assert profile["confidence"] >= expected["front"]["profile"]["minimumConfidence"]

    session = start_try_on_session("recorded-video", owner_id="fixture-test")
    stream = iter_try_on_mjpeg("recorded-video", session["streamToken"], fps=60)
    try:
        chunk = next(stream)
        header, jpeg = chunk.split(b"\r\n\r\n", 1)
        assert b"Content-Type: image/jpeg" in header
        assert expected["front"]["tryOn"]["jpeg"] is True
        assert cv2.imdecode(np.frombuffer(jpeg.rstrip(b"\r\n"), np.uint8), cv2.IMREAD_COLOR) is not None
    finally:
        stream.close()
        stop_try_on_session("recorded-video", owner_id="fixture-test")

    updates = [first.update]
    departure_snapshot = None
    for _ in range(expected["top"]["departureWithinPolls"]):
        result = runtime.poll(include_status=True, include_ambient_light=False, include_departure=True)
        if result.update is not None:
            updates.append(result.update)
        if result.update and result.update["message_type"] == "vision.person_departed":
            departure_snapshot = result.snapshot
            break
    camera_manager.release_all_cameras()
    assert departure_snapshot is not None
    assert departure_snapshot["occupancy"]["state"] == departure_stage["occupancy"]
    actual_events = [update["message_type"] for update in updates if update]
    event_index = 0
    for event_type in actual_events:
        if event_type == expected["top"]["protocolEvents"][event_index]:
            event_index += 1
            if event_index == len(expected["top"]["protocolEvents"]):
                break
    assert event_index == len(expected["top"]["protocolEvents"])
