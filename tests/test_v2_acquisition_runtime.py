"""Public recorded-frame contract for V2 acquisition preview.

This is deliberately a websocket/HTTP test: it does not seed a registry or
reach into the preview store.  The test is the first Phase-B tracer bullet.
"""

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
from vision.config import settings
from vision.acquisition_observer import AcquisitionObservationWorker
from vision.acquisition_observer import AcquisitionObservation
from vision.profile_messages import profile_update


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


class _AcquiringOnlyObserver:
    ready = True
    fatal_error = None
    pid = None
    active_request_count = 0
    assert_dead = True

    async def start(self):
        return None

    async def observe(self, _frame, *, timeout=15.0):
        return AcquisitionObservation(b"jpeg", "single", False)

    async def wait_idle(self):
        return None

    async def shutdown(self):
        return None


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
        thread.join()
        vision_app._acquisition_observer = None


def test_public_recorded_top_departure_cancels_attempt_without_stopping_profile_events(monkeypatch):
    """Production top presence cancels a public WS attempt; the stream keeps reporting Vision facts."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", True)
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_INTERVAL_MS", 10)
    monkeypatch.setattr(vision_app, "_FAST_ATTEMPT_TIMEOUT_SECONDS", 8)
    monkeypatch.setattr(vision_app, "_ACQUISITION_POLL_SECONDS", 0.01)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _ReadyFastBroker())
    monkeypatch.setattr(vision_app, "_acquisition_observer", _AcquiringOnlyObserver())
    monkeypatch.setattr(
        vision_app,
        "collect_front_profile_update",
        lambda event_id, *_args, **_kwargs: profile_update(
            "vision.profile_result",
            {
                "eventId": event_id,
                "profile": {"presence": True, "age": 30, "gender": "unknown"},
                "source": "front",
            },
        ),
    )
    _configure_recorded_top(monkeypatch)
    _configure_recorded_front(monkeypatch, "man-front.mp4")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    attempt_id = str(uuid4())
    canceled = []
    departures = []
    seen_types = []
    post_cancel_presence = False
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
                    if message["type"] == "vision.try_on.attempt.canceled":
                        canceled.append(message)
                    if message["type"] == "vision.person_departed":
                        departures.append(message)
                    if canceled and message["type"] == "vision.presence_status":
                        post_cancel_presence = True
                    if canceled and departures and post_cancel_presence:
                        break

                assert [message["payload"] for message in canceled] == [
                    {"attemptId": attempt_id, "reason": "departure"}
                ]
                assert len(departures) == 1
                assert "vision.presence_status" in seen_types
                assert "vision.profile_result" in seen_types
                assert post_cancel_presence
                assert vision_app.get_front_camera_owner()["owner"] == "idle"
    finally:
        server.shutdown()
        thread.join()
        camera_manager.release_all_cameras()
        monkeypatch.setattr(presence_runtime, "_runtime", None)


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
        thread.join()


@pytest.mark.parametrize(
    ("filename", "occupancy", "guidance"),
    [
        ("empty-front.mp4", "none", "no_person"),
        ("front.mp4", "multiple", "multiple_people"),
        ("man-unaligned-front.mp4", "single", "align"),
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
        thread.join()
