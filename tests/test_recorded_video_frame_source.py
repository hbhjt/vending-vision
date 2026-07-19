import asyncio
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

import app as vision_app
import vision.try_on_session as try_on_session
from vision.frame_source import RecordedVideoFrameSource
from vision import camera_manager
from vision.config import settings
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_occupancy_gate, reset_active_track
from vision.self_check import check_camera
from vision.try_on_session import (
    get_try_on_status,
    iter_try_on_mjpeg,
    start_try_on_session,
    stop_try_on_session,
)


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


def write_managed_recorded_video_config(path, manifest, *, front_loop=None):
    cameras = {}
    for role in ("top", "front"):
        recording = manifest["recordings"][role]
        cameras[role] = {
            "role": "presence" if role == "top" else "profile_tryon",
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


def assert_recorded_video_source_frame(
    frame, *, role, video_sha256, config_sha256,
):
    assert frame["adapter"] == "recorded_video"
    assert frame["role"] == role
    assert frame["fixtureSha256"] == video_sha256
    assert frame["configSha256"] == config_sha256
    assert isinstance(frame["frameIndex"], int)
    assert isinstance(frame["decodedFrameCount"], int)
    assert frame["frameIndex"] >= 0
    assert 0 <= frame["frameIndex"] < frame["decodedFrameCount"]
    assert frame["synthetic"] is False
    assert frame["relabeled"] is False


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


def test_read_camera_records_recorded_video_source_frame_metadata(monkeypatch, tmp_path):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    managed_config_sha = write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))

    configure_recorded_sources(monkeypatch, manifest)
    camera_manager.release_all_cameras()

    camera_manager.read_camera("top", warmup_frames=1)
    source_top = camera_manager.get_last_frame_source("top")
    assert source_top is not None
    assert_recorded_video_source_frame(
        source_top,
        role="top",
        video_sha256=manifest["recordings"]["top"]["sha256"],
        config_sha256=managed_config_sha,
    )

    camera_manager.read_camera("front", warmup_frames=1)
    source_front = camera_manager.get_last_frame_source("front")
    assert source_front is not None
    assert_recorded_video_source_frame(
        source_front,
        role="front",
        video_sha256=manifest["recordings"]["front"]["sha256"],
        config_sha256=managed_config_sha,
    )


def test_recorded_video_metadata_is_omitted_when_managed_camera_config_does_not_match(
    monkeypatch, tmp_path,
):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    write_managed_recorded_video_config(
        managed_config, manifest, front_loop=not manifest["recordings"]["front"]["loop"],
    )
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))

    configure_recorded_sources(monkeypatch, manifest)

    _, source_frame = camera_manager.read_camera_with_source("front", warmup_frames=1)

    assert source_frame is None


def test_expected_manifest_drives_real_presence_profile_departure_and_try_on_pipeline(
    monkeypatch,
    tmp_path,
):
    manifest = fixture_manifest()
    expected = manifest["expected"]

    managed_config = tmp_path / "managed-site-config.json"
    managed_config_sha = write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))

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
    assert first.update["payload"]["source"] == "top"
    assert_recorded_video_source_frame(
        first.update["payload"]["sourceFrame"],
        role="top",
        video_sha256=manifest["recordings"]["top"]["sha256"],
        config_sha256=managed_config_sha,
    )

    asyncio.run(
        vision_app.profile_collection_worker(first.candidate, threading.Event())
    )

    profile_update = next(update for update in published if update["message_type"] == "vision.profile_result")
    profile = profile_update["payload"]["profile"]
    assert profile["personPresent"] is True
    assert set(expected["front"]["profile"]["minimumFields"]).issubset(profile)
    assert profile["confidence"] >= expected["front"]["profile"]["minimumConfidence"]
    assert profile_update["payload"]["source"] == "front"
    assert_recorded_video_source_frame(
        profile_update["payload"]["sourceFrame"],
        role="front",
        video_sha256=manifest["recordings"]["front"]["sha256"],
        config_sha256=managed_config_sha,
    )

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
    if any(update["message_type"] == "vision.person_departed" for update in updates):
        departure_update = next(
            update for update in updates if update["message_type"] == "vision.person_departed"
        )
        assert departure_update["payload"]["source"] == "top"
        assert_recorded_video_source_frame(
            departure_update["payload"]["sourceFrame"],
            role="top",
            video_sha256=manifest["recordings"]["top"]["sha256"],
            config_sha256=managed_config_sha,
        )
    assert departure_snapshot["occupancy"]["state"] == departure_stage["occupancy"]
    actual_events = [update["message_type"] for update in updates if update]
    event_index = 0
    for event_type in actual_events:
        if event_type == expected["top"]["protocolEvents"][event_index]:
            event_index += 1
            if event_index == len(expected["top"]["protocolEvents"]):
                break
    assert event_index == len(expected["top"]["protocolEvents"])


def test_try_on_mjpeg_returns_recorded_video_vem_headers_for_first_frame(monkeypatch, tmp_path):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))

    configure_recorded_sources(monkeypatch, manifest)

    captured = {}
    original_prepare = vision_app.prepare_first_try_on_frame

    def spy_prepare(session_id, stream_token, jpeg_quality=80):
        jpeg, source = original_prepare(
            session_id, stream_token, jpeg_quality=jpeg_quality,
        )
        captured["jpeg"] = jpeg
        captured["source"] = source
        return jpeg, source

    monkeypatch.setattr(vision_app, "prepare_first_try_on_frame", spy_prepare)
    session = start_try_on_session("recorded-video-mjpeg", owner_id="fixture-test")
    response = None

    try:
        response = vision_app.try_on_mjpeg(
            session["sessionId"], token=session["streamToken"]
        )
        assert response.status_code == 200
        source = captured["source"]
        assert source["adapter"] == "recorded_video"
        assert source["role"] == "front"
        assert response.headers["x-vem-frame-source-adapter"] == "recorded_video"
        assert response.headers["x-vem-frame-source-role"] == "front"
        assert response.headers["x-vem-frame-source-config-sha256"] == source["configSha256"]
        assert response.headers["x-vem-frame-source-file-sha256"] == source["fixtureSha256"]
        assert int(response.headers["x-vem-frame-source-frame-index"]) == source["frameIndex"]
        assert (
            int(response.headers["x-vem-frame-source-decoded-frame-count"])
            == source["decodedFrameCount"]
        )
        assert response.headers["x-vem-frame-session-id"] == session["sessionId"]

        async def read_first_chunk():
            iterator = response.body_iterator
            return await iterator.__anext__()

        frame_chunk = asyncio.run(read_first_chunk())
        _, jpeg = frame_chunk.split(b"\r\n\r\n", 1)
        jpeg = jpeg.rstrip(b"\r\n")
        assert jpeg == captured["jpeg"]
        assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) is not None
    finally:
        if response is not None and hasattr(response, "body_iterator"):
            iterator = response.body_iterator
            if hasattr(iterator, "aclose"):
                asyncio.run(iterator.aclose())
        stop_try_on_session(session["sessionId"], owner_id="fixture-test")


def test_try_on_mjpeg_rejects_client_limit_before_reading_first_frame(monkeypatch):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    read_count = 0

    def read_front(role, warmup_frames=None):
        nonlocal read_count
        read_count += 1
        return frame, None

    monkeypatch.setattr(try_on_session, "read_camera_with_source", read_front)
    monkeypatch.setattr(
        try_on_session,
        "read_camera",
        lambda role, warmup_frames=None: read_front(role, warmup_frames)[0],
    )
    monkeypatch.setattr(settings, "TRY_ON_MAX_STREAM_CLIENTS", 1)
    session = start_try_on_session("try-on-client-limit", owner_id="fixture-test")
    first_stream = iter_try_on_mjpeg(
        session["sessionId"], session["streamToken"], fps=60,
    )
    try:
        next(first_stream)
        reads_before_rejection = read_count

        response = vision_app.try_on_mjpeg(
            session["sessionId"], token=session["streamToken"],
        )

        assert response.status_code == 500
        assert read_count == reads_before_rejection
    finally:
        first_stream.close()
        assert get_try_on_status()["activeSession"]["streamClientCount"] == 0
        stop_try_on_session(session["sessionId"], owner_id="fixture-test")


def test_try_on_mjpeg_releases_stream_admission_when_first_frame_fails(monkeypatch):
    def fail_read(role, warmup_frames=None):
        raise RuntimeError("front frame unavailable")

    monkeypatch.setattr(try_on_session, "read_camera_with_source", fail_read)
    session = start_try_on_session("try-on-first-frame-fails", owner_id="fixture-test")
    try:
        response = vision_app.try_on_mjpeg(
            session["sessionId"], token=session["streamToken"],
        )

        assert response.status_code == 500
        assert get_try_on_status()["activeSession"]["streamClientCount"] == 0
    finally:
        stop_try_on_session(session["sessionId"], owner_id="fixture-test")


def test_try_on_mjpeg_does_not_send_prepared_frame_after_session_stops(
    monkeypatch, tmp_path,
):
    manifest = fixture_manifest()
    managed_config = tmp_path / "managed-site-config.json"
    write_managed_recorded_video_config(managed_config, manifest)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_config))
    configure_recorded_sources(monkeypatch, manifest)
    session = start_try_on_session("try-on-stop-before-body", owner_id="fixture-test")
    response = vision_app.try_on_mjpeg(
        session["sessionId"], token=session["streamToken"],
    )
    stop_try_on_session(session["sessionId"], owner_id="fixture-test")

    async def read_after_stop():
        with pytest.raises(StopAsyncIteration):
            await response.body_iterator.__anext__()

    asyncio.run(read_after_stop())
