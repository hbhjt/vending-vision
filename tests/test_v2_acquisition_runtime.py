"""Public recorded-frame contract for V2 acquisition preview.

This is deliberately a websocket/HTTP test: it does not seed a registry or
reach into the preview store.  The test is the first Phase-B tracer bullet.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import multiprocessing
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as vision_app
from vision import camera_manager
from vision import presence_runtime
from vision import profile_push
from vision.config import settings
from vision.acquisition_observer import AcquisitionObservationWorker
from vision.frame_source import RecordedVideoFrameSource
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_departure_tracker, get_occupancy_gate, reset_active_track

_RECORDED_FIXTURE_WATCHDOG_SECONDS = 5.0


def _wait_for_recorded_fixture_event(event, *, timeout=_RECORDED_FIXTURE_WATCHDOG_SECONDS):
    """Bound a test-only cross-thread barrier without changing Vision deadlines."""
    return event.wait(timeout)


def _permanently_blocking_acquisition_observer(connection):
    connection.send(("ready", None))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "observe":
                while True:
                    time.sleep(1)
    finally:
        connection.close()


def _png_bytes():
    image = np.full((48, 36, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _GarmentHandler(BaseHTTPRequestHandler):
    payload = _png_bytes()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_):
        return


def _envelope(message_type, payload):
    return {
        "protocol": "vem.vision.v2",
        "type": message_type,
        "messageId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


def _configure_recorded_front(monkeypatch, filename="front.mp4"):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "recorded-video"
    monkeypatch.setattr(settings, "FRONT_CAMERA_CONFIG", {
        "role": "profile_fast_try_on", "source": "recorded_video",
        "video_path": str(fixture_root / filename), "loop": True, "rotate": 0,
    })
    camera_manager.release_all_cameras()


def _configure_recorded_top(monkeypatch):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "recorded-video"
    monkeypatch.setattr(settings, "TOP_CAMERA_CONFIG", {
        "role": "presence", "source": "recorded_video",
        "video_path": str(fixture_root / "top.mp4"), "loop": False, "rotate": 0,
    })
    monkeypatch.setattr(presence_runtime, "_runtime", None)
    camera_manager.release_all_cameras()


class _ReadyFastBroker:
    ready = True
    pose_ready = True

    async def start(self):
        return None

    def quiesce(self):
        return None

    async def shutdown(self):
        return None


class _RecordedTopDepartureMonitor:
    """Drive real PresenceRuntime state with the declared recorded top source.

    The production detector's OpenCV/MediaPipe latency is not part of this
    websocket ordering contract.  Each poll still decodes exactly one recorded
    top frame; this boundary supplies the fixture's deterministic occupied then
    absent detector facts to PresenceRuntime's real debounce and departure
    tracker.
    """

    def __init__(
        self,
        config,
        *,
        present_polls,
        front_sampling_started=None,
        profile_result_broadcasted=None,
    ):
        self.source = RecordedVideoFrameSource("top", config)
        self.present_polls = present_polls
        self.front_sampling_started = front_sampling_started
        self.profile_result_broadcasted = profile_result_broadcasted
        self.poll_count = 0
        self.front_sampling_observed = False
        self.profile_result_broadcast_observed = False

    def check_once(self, *, return_image, camera_role, return_source):
        assert return_image and return_source and camera_role == "top"
        # The first recorded top frame creates the candidate.  Do not consume
        # the finite departure fixture before that candidate has actually
        # entered real front-camera sampling and then broadcast its profile
        # result. Under a busy Windows runner, merely entering sampling leaves
        # it possible for the real collector to be canceled by departure
        # before it commits its public result.
        if self.poll_count == 1:
            if self.front_sampling_started is not None:
                assert _wait_for_recorded_fixture_event(self.front_sampling_started), (
                    "recorded departure fixture timed out before front sampling started"
                )
                self.front_sampling_observed = True
            if self.profile_result_broadcasted is not None:
                assert _wait_for_recorded_fixture_event(self.profile_result_broadcasted), (
                    "recorded departure fixture timed out before profile result broadcast"
                )
                self.profile_result_broadcast_observed = True
        image = self.source.read(warmup_frames=1)
        self.poll_count += 1
        if self.poll_count <= self.present_polls:
            proximity = {
                "present": True,
                "close": True,
                "rawCount": 1,
                "personCount": 1,
                "personPresent": True,
                "largestPersonBox": {
                    "centerX": 0.5,
                    "centerY": 0.5,
                    "width": 0.4,
                    "height": 0.6,
                },
                "largestPersonRatio": 0.24,
                "faceCount": 0,
                "facePresent": False,
                "bodyPresent": False,
                "topOccupancy": {"occupancy": "single", "confidence": 0.9},
            }
        else:
            proximity = {
                "present": False,
                "close": False,
                "rawCount": 0,
                "personCount": 0,
                "personPresent": False,
                "faceCount": 0,
                "facePresent": False,
                "bodyPresent": False,
                "topOccupancy": {"occupancy": "none", "confidence": 0.9},
            }
        return proximity, image, self.source.last_frame()

    def release(self):
        self.source.release()


def _reset_recorded_departure_state():
    gate = get_occupancy_gate()
    for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
        gate.mark_absent()
    reset_active_track()
    tracker = get_departure_tracker()
    tracker.active = False
    tracker.absent_count = 0
    tracker.departed_announced = False


def test_recorded_departure_fixture_watchdog_unblocks_asyncio_run():
    """An unmet test barrier must not strand asyncio's default executor."""
    unset = threading.Event()
    waiter_finished = threading.Event()

    async def scenario():
        def wait_for_unset_fixture_barrier():
            try:
                return _wait_for_recorded_fixture_event(unset, timeout=0.02)
            finally:
                waiter_finished.set()

        return await asyncio.to_thread(wait_for_unset_fixture_barrier)

    started = time.monotonic()
    assert asyncio.run(scenario()) is False
    assert waiter_finished.is_set()
    assert time.monotonic() - started < 0.5


def test_v2_ws_ping_and_cancel_stay_live_while_production_observer_blocks(monkeypatch):
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    frame = cv2.imread(str(Path(__file__).parents[1] / "fixtures/recorded-video/sources/person-man-front.png"))
    assert frame is not None
    monkeypatch.setattr(vision_app, "read_camera_with_source", lambda *_args, **_kwargs: (frame, {"source": "recorded_video"}))
    vision_app._acquisition_observer = AcquisitionObservationWorker(context=multiprocessing.get_context("spawn"), target=_permanently_blocking_acquisition_observer)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {"clientRole": "machine", "machineCode": "M001", "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"], "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"]}))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {"attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()), "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"}}))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                deadline = time.monotonic() + 1
                while vision_app._acquisition_observer.active_request_count == 0 and time.monotonic() < deadline:
                    time.sleep(0.002)
                assert vision_app._acquisition_observer.active_request_count == 1
                socket.send_json(_envelope("vision.ping", {}))
                assert socket.receive_json()["type"] == "vision.pong"
                socket.send_json(_envelope("vision.try_on.attempt.cancel", {"attemptId": attempt_id, "reason": "user"}))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.canceled"
                deadline = time.monotonic() + 1
                while not vision_app._acquisition_observer.assert_dead and time.monotonic() < deadline:
                    time.sleep(0.002)
                assert vision_app._acquisition_observer.assert_dead
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        vision_app._acquisition_observer = None


def test_public_recorded_top_departure_does_not_cancel_attempt_and_keeps_profile_events(monkeypatch):
    """Production top presence departure must not cancel a public WS attempt; the stream keeps reporting Vision facts."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    recorded_manifest = json.loads(
        (Path(__file__).parents[1] / "fixtures/recorded-video/expected-results.json").read_text(
            "utf-8"
        )
    )
    expected_polls = recorded_manifest["expected"]["top"]["departureWithinPolls"]
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", True)
    # The deterministic top boundary reaches the fixture's departure edge in
    # 23 polls while leaving the real front sampler time to publish its result.
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_INTERVAL_MS", 250)
    monkeypatch.setattr(vision_app, "_FAST_ATTEMPT_TIMEOUT_SECONDS", 10)
    monkeypatch.setattr(vision_app, "_ACQUISITION_POLL_SECONDS", 0.01)
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 1)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _ReadyFastBroker())
    monkeypatch.setattr(
        vision_app,
        "_acquisition_observer",
        AcquisitionObservationWorker(context=multiprocessing.get_context("spawn")),
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": True,
            "fastPoseReady": True,
            "acquisitionObserverReady": vision_app._acquisition_observer_ready(),
        },
    )
    _configure_recorded_top(monkeypatch)
    _configure_recorded_front(monkeypatch, "man-front.mp4")
    _reset_recorded_departure_state()
    front_sampling_started = threading.Event()
    profile_result_broadcasted = threading.Event()
    real_collect_best_profile_samples = profile_push.collect_best_profile_samples

    def collect_best_after_front_sampling_started(*args, **kwargs):
        front_sampling_started.set()
        return real_collect_best_profile_samples(*args, **kwargs)

    monkeypatch.setattr(
        profile_push,
        "collect_best_profile_samples",
        collect_best_after_front_sampling_started,
    )
    monitor = _RecordedTopDepartureMonitor(
        settings.TOP_CAMERA_CONFIG,
        present_polls=expected_polls - 2,
        front_sampling_started=front_sampling_started,
        profile_result_broadcasted=profile_result_broadcasted,
    )
    runtime = PresenceRuntime(monitor=monitor)
    monkeypatch.setattr(vision_app, "get_presence_runtime", lambda: runtime)
    monkeypatch.setattr(presence_runtime, "get_presence_runtime", lambda: runtime)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    completed = []
    departures = []
    seen_types = []
    post_departure_presence = False
    generating_seen = False
    front_idle_after_generating = False
    profile_after_generating = False
    ambient_payloads = []
    departure_seen = threading.Event()

    def missing_facts():
        missing = []
        if not departures:
            missing.append("person departure")
        if not generating_seen:
            missing.append("generating")
        if not completed:
            missing.append("completion after departure")
        if not post_departure_presence:
            missing.append("post-departure presence")
        if not front_idle_after_generating:
            missing.append("front camera release before generation")
        if not profile_after_generating:
            missing.append("profile after generation started")
        if not ambient_payloads:
            missing.append("ambient observation")
        return missing

    generating_started = threading.Event()
    real_broadcast_profile_update = vision_app.broadcast_profile_update

    async def broadcast_after_generating(update):
        if update["message_type"] == "vision.profile_result":
            assert await asyncio.to_thread(
                _wait_for_recorded_fixture_event, generating_started
            ), "recorded departure fixture timed out before attempt generation"
        await real_broadcast_profile_update(update)
        if update["message_type"] == "vision.profile_result":
            profile_result_broadcasted.set()

    async def render_until_departure(*_args, **_kwargs):
        generating_started.set()
        # Hold generation until the recorded departure edge has been observed
        # and broadcast: a top-camera departure must not cancel the active
        # front-camera attempt.
        assert await asyncio.to_thread(
            _wait_for_recorded_fixture_event, departure_seen, timeout=8
        ), "recorded departure edge did not arrive while the attempt was generating"
        return _png_bytes()

    monkeypatch.setattr(vision_app, "render_attempt_frame", render_until_departure)
    monkeypatch.setattr(vision_app, "broadcast_profile_update", broadcast_after_generating)
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {
                    "clientRole": "machine", "machineCode": "M001",
                    "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"],
                    "contractDigest": manifest["bundleDigest"],
                    "capabilities": [
                        "try_on_fast", "profile_push", "presence_status",
                        "person_departed", "ambient_light",
                    ],
                }))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {
                    "attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()),
                    "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"},
                }))

                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    message = socket.receive_json()
                    seen_types.append(message["type"])
                    if message["type"] == "vision.try_on.attempt.generating":
                        generating_seen = True
                        front_idle_after_generating = (
                            vision_app.get_front_camera_owner()["owner"] == "idle"
                        )
                    if generating_seen and message["type"] == "vision.profile_result":
                        profile_after_generating = True
                    if message["type"] == "vision.try_on.attempt.completed":
                        completed.append(message)
                    if message["type"] == "vision.person_departed":
                        departures.append(message)
                        departure_seen.set()
                    if completed and message["type"] == "vision.presence_status":
                        post_departure_presence = True
                    if message["type"] in {
                        "vision.presence_status",
                        "vision.person_departed",
                    } and "ambientLight" in message["payload"]:
                        ambient_payloads.append(message["payload"]["ambientLight"])
                    if not missing_facts():
                        break

                assert not missing_facts(), (
                    f"missing facts before deadline: {missing_facts()}; "
                    f"seen types: {seen_types}"
                )
                assert len(completed) == 1
                assert completed[0]["payload"]["attemptId"] == attempt_id
                assert isinstance(completed[0]["payload"].get("result"), dict)
                assert isinstance(
                    completed[0]["payload"]["result"].get("reference"), str
                )
                assert isinstance(completed[0]["payload"]["result"].get("digest"), str)
                assert "vision.try_on.attempt.canceled" not in seen_types
                assert "vision.try_on.attempt.failed" not in seen_types
                assert generating_seen
                assert front_idle_after_generating
                assert profile_after_generating
                assert len(departures) == 1
                assert departure_seen.is_set()
                assert "vision.presence_status" in seen_types
                assert "vision.profile_result" in seen_types
                assert ambient_payloads
                assert all(
                    "level" in payload
                    and isinstance(payload.get("sample"), dict)
                    and "lumaMean" in payload["sample"]
                    for payload in ambient_payloads
                )
                assert any(
                    message_type in seen_types
                    for message_type in ["vision.presence_status", "vision.profile_result"]
                )
                assert post_departure_presence
                assert monitor.front_sampling_observed
                assert monitor.profile_result_broadcast_observed
                assert monitor.poll_count >= expected_polls + 1
                assert monitor.source.frame_count == monitor.poll_count
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        # Release every fixture barrier before shutting down TestClient.  A
        # failing assertion must never leave an asyncio default-executor
        # worker blocked on an unset Event.
        front_sampling_started.set()
        profile_result_broadcasted.set()
        generating_started.set()
        server.shutdown()
        server.server_close()
        thread.join()
        monitor.release()
        camera_manager.release_all_cameras()
        monkeypatch.setattr(presence_runtime, "_runtime", None)
        if vision_app._acquisition_observer is not None:
            asyncio.run(vision_app._acquisition_observer.shutdown())
        vision_app._acquisition_observer = None


def test_v2_start_exposes_attempt_scoped_tokenized_acquisition_preview(monkeypatch):
    """A Machine can display, but never supply, the acquiring camera input."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {
                    "clientRole": "machine", "machineCode": "M001",
                    "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"],
                    "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"],
                }))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {
                    "attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()),
                    "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"},
                }))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                acquiring = socket.receive_json()
                assert acquiring["type"] == "vision.try_on.attempt.acquiring", acquiring
                # This recorded front fixture is a real multi-person scene;
                # production YOLO, not a pose stub, must keep it blocked.
                assert acquiring["payload"]["occupancy"] == "multiple"
                assert acquiring["payload"]["guidance"] == "multiple_people"
                assert acquiring["payload"]["manualCaptureAllowed"] is False
                preview = acquiring["payload"]["preview"]
                parsed = urlsplit(preview["reference"])
                assert parsed.path == "/v2/try-on/acquisition/preview.mjpeg"
                assert parsed.query.startswith("token=")
                assert preview["streamType"] == "mjpeg"
                assert client.head(preview["reference"]).status_code == 200
                assert client.get(f"{parsed.path}?token=wrong").status_code == 404
                assert client.get(parsed.path).status_code == 404
                assert client.get(f"{parsed.path}?{parsed.query}&extra=1").status_code == 404
                assert client.post(f"{parsed.path}?{parsed.query}").status_code == 405
                socket.send_json(_envelope("vision.try_on.attempt.cancel", {
                    "attemptId": attempt_id, "reason": "user",
                }))
                canceled = socket.receive_json()
                assert canceled["type"] == "vision.try_on.attempt.canceled"
                assert canceled["payload"] == {"attemptId": attempt_id, "reason": "user"}
                assert client.head(preview["reference"]).status_code == 404
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_manual_capture_bypasses_stability_but_not_single_person_alignment(monkeypatch):
    """Manual intent converts an unstable eligible recorded observation to generating."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 100)
    _configure_recorded_front(monkeypatch, "man-front.mp4")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {"clientRole": "machine", "machineCode": "M001", "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"], "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"]}))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {"attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()), "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"}}))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                acquiring = socket.receive_json()
                assert acquiring["payload"]["guidance"] == "hold_still"
                socket.send_json(_envelope("vision.try_on.attempt.capture", {"attemptId": attempt_id}))
                generating = socket.receive_json()
                assert generating["type"] == "vision.try_on.attempt.generating"
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    ("filename", "occupancy", "guidance"),
    [
        ("empty-front.mp4", "none", "no_person"),
        ("front.mp4", "multiple", "multiple_people"),
    ],
)
def test_v2_recorded_production_observation_truthfully_blocks_capture(
    monkeypatch, filename, occupancy, guidance
):
    """No detector, box, or pose stub may turn an unsafe source into capture."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch, filename)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {"clientRole": "machine", "machineCode": "M001", "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"], "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"]}))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {"attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()), "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"}}))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                acquiring = socket.receive_json()
                assert acquiring["type"] == "vision.try_on.attempt.acquiring"
                assert acquiring["payload"]["occupancy"] == occupancy
                assert acquiring["payload"]["guidance"] == guidance
                assert acquiring["payload"]["manualCaptureAllowed"] is False
                socket.send_json(_envelope("vision.try_on.attempt.capture", {"attemptId": attempt_id}))
                socket.send_json(_envelope("vision.try_on.attempt.cancel", {"attemptId": attempt_id, "reason": "user"}))
                canceled = socket.receive_json()
                assert canceled["type"] == "vision.try_on.attempt.canceled"
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_recorded_close_up_single_person_manual_capture_proceeds_when_aligned(
    monkeypatch,
):
    """A close-up single person (lower body cropped) is aligned and manual intent captures immediately."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 100)
    _configure_recorded_front(monkeypatch, "man-unaligned-front.mp4")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {"clientRole": "machine", "machineCode": "M001", "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"], "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"]}))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {"attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()), "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"}}))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                acquiring = socket.receive_json()
                assert acquiring["type"] == "vision.try_on.attempt.acquiring"
                assert acquiring["payload"]["occupancy"] == "single"
                assert acquiring["payload"]["guidance"] == "hold_still"
                assert acquiring["payload"]["manualCaptureAllowed"] is True
                socket.send_json(_envelope("vision.try_on.attempt.capture", {"attemptId": attempt_id}))
                generating = socket.receive_json()
                assert generating["type"] == "vision.try_on.attempt.generating"
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_recorded_single_person_auto_capture_uses_production_yolo_and_pose(monkeypatch):
    """A traceable recorded input automatically reaches generation without detector stubs."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True, "fastRenderReady": True, "fastPoseReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch, "man-front.mp4")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_envelope("vision.hello", {
                    "clientRole": "machine", "machineCode": "M001",
                    "schemaVersion": manifest["schemaVersion"], "bundleVersion": manifest["bundleVersion"],
                    "contractDigest": manifest["bundleDigest"], "capabilities": ["try_on_fast"],
                }))
                assert socket.receive_json()["type"] == "vision.ready"
                garment = _GarmentHandler.payload
                socket.send_json(_envelope("vision.try_on.attempt.start", {
                    "attemptId": attempt_id, "mode": "fast", "variantId": str(uuid4()),
                    "garment": {"assetId": str(uuid4()), "reference": f"http://127.0.0.1:{server.server_port}/garment?token=source-token", "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}", "contentType": "image/png", "byteSize": len(garment), "template": "tshirt_short_sleeve"},
                }))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                acquiring = socket.receive_json()
                assert acquiring["type"] == "vision.try_on.attempt.acquiring"
                assert acquiring["payload"]["occupancy"] == "single"
                assert acquiring["payload"]["guidance"] in {"hold_still", "ready"}
                while True:
                    message = socket.receive_json()
                    if message["type"] == "vision.try_on.attempt.generating":
                        break
                    assert message["type"] == "vision.try_on.attempt.acquiring"
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
                terminal = socket.receive_json()
                assert terminal["type"] == "vision.try_on.attempt.completed"
                assert terminal["payload"]["attemptId"] == attempt_id
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
