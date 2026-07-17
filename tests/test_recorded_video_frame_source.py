import hashlib
import json
from pathlib import Path
import time

from vision.frame_source import RecordedVideoFrameSource
from vision import camera_manager
from vision.config import settings
from vision.proximity import ProximityMonitor
from vision.profile_mapper import vision_profile_to_protocol
from vision.profile_push import collect_front_profile_update
from vision.profile_state import ensure_active_track, get_occupancy_gate, reset_active_track, target_signature_from_proximity
from vision.self_check import check_camera
from vision.try_on_session import iter_try_on_mjpeg, start_try_on_session, stop_try_on_session
from vision.vision_pipeline import VisionPipeline


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "recorded-video"


def test_recorded_video_fixture_manifest_binds_top_and_front_recordings():
    manifest = json.loads((FIXTURE_ROOT / "expected-results.json").read_text())

    assert manifest["schemaVersion"] == "vending-vision-recorded-video-fixture/v1"
    assert set(manifest["recordings"]) == {"top", "front"}
    for role, recording in manifest["recordings"].items():
        video = FIXTURE_ROOT / recording["file"]
        assert video.is_file(), role
        assert recording["sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()


def test_recorded_video_source_decodes_fixture_frames_in_order():
    source = RecordedVideoFrameSource(
        role="top",
        config={
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / "top.mp4"),
            "loop": False,
        },
    )

    first = source.read()
    second = source.read()

    assert first.shape == second.shape
    assert first.size > 0
    assert source.status()["source"] == "recorded_video"

    source.release()


def test_health_camera_check_accepts_both_recorded_video_sources(monkeypatch):
    for role, role_name in (("TOP", "top"), ("FRONT", "front")):
        monkeypatch.setattr(
            settings,
            f"{role}_CAMERA_CONFIG",
            {
                "role": "presence" if role_name == "top" else "profile_tryon",
                "source": "recorded_video",
                "video_path": str(FIXTURE_ROOT / f"{role_name}.mp4"),
                "loop": role_name == "front",
                "rotate": 0,
            },
        )
    camera_manager.release_all_cameras()

    status = check_camera()

    assert status["ok"] is True
    assert status["detail"]["sources"] == {"top": "recorded_video", "front": "recorded_video"}

    import app as vision_app

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


def test_recorded_frames_reach_presence_profile_and_try_on_paths(monkeypatch):
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
    monkeypatch.setattr(
        settings,
        "FRONT_CAMERA_CONFIG",
        {
            "role": "profile_tryon",
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / "front.mp4"),
            "loop": True,
            "rotate": 0,
        },
    )
    camera_manager.release_all_cameras()

    monitor = ProximityMonitor()
    snapshots = [monitor.check_once(camera_role="top") for _ in range(12)]
    for _ in range(12):
        snapshots.append(monitor.check_once(camera_role="top"))
        time.sleep(0.1)
    assert snapshots[0]["topOccupancy"]["occupancy"] == "single"
    assert snapshots[-1]["present"] is False

    profile = vision_profile_to_protocol(
        VisionPipeline().infer(camera_manager.read_camera("front", warmup_frames=1))
    )
    assert profile["personPresent"] is True
    assert profile["confidence"] >= settings.PROFILE_MIN_CONFIDENCE

    session = start_try_on_session("recorded-video", owner_id="fixture-test")
    stream = iter_try_on_mjpeg("recorded-video", session["streamToken"], fps=60)
    try:
        assert next(stream).startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
    finally:
        stream.close()
        stop_try_on_session("recorded-video", owner_id="fixture-test")


def test_recorded_front_frames_produce_profile_result_through_existing_protocol(monkeypatch):
    monkeypatch.setattr(
        settings,
        "FRONT_CAMERA_CONFIG",
        {
            "role": "profile_tryon",
            "source": "recorded_video",
            "video_path": str(FIXTURE_ROOT / "front.mp4"),
            "loop": True,
            "rotate": 0,
        },
    )
    camera_manager.release_all_cameras()
    proximity = {
        "present": True,
        "personPresent": True,
        "facePresent": True,
        "personCount": 1,
        "faceCount": 1,
        "largestPersonRatio": 0.2,
        "largestPersonBox": {"centerX": 0.5, "centerY": 0.5},
        "topOccupancy": {"occupancy": "single", "confidence": 1.0},
    }
    gate = get_occupancy_gate()
    gate.mark_present()
    track = ensure_active_track(target_signature_from_proximity(proximity), "single")
    try:
        update = collect_front_profile_update(
            event_id="recorded-video-profile",
            proximity=proximity,
            track=track,
            close_enough=True,
            ambient_light=None,
            include_status=True,
            completion_validator=lambda: True,
            completion_occupancy=lambda: {"state": "single", "confidence": 1.0},
        )
    finally:
        reset_active_track()

    assert update["message_type"] == "vision.profile_result"
    assert update["payload"]["profile"]["personPresent"] is True
