import hashlib
import json
import threading
from urllib.parse import urlsplit
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as vision_app
from vision import camera_manager
from vision.config import settings


def _png_bytes():
    image = np.full((48, 36, 4), (20, 120, 220, 255), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _GarmentHandler(BaseHTTPRequestHandler):
    payload = _png_bytes()
    entered = threading.Event()
    release = threading.Event()

    def do_GET(self):
        self.entered.set()
        self.release.wait(timeout=5)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_):
        return


@pytest.fixture
def garment_reference():
    _GarmentHandler.entered.clear()
    _GarmentHandler.release.set()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/garment?token=source-token"
    server.shutdown()
    thread.join()


def _configure_recorded_front(monkeypatch):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "recorded-video"
    monkeypatch.setattr(
        settings,
        "FRONT_CAMERA_CONFIG",
        {
            "role": "profile_tryon",
            "source": "recorded_video",
            "video_path": str(fixture_root / "front.mp4"),
            "loop": True,
            "rotate": 0,
        },
    )
    camera_manager.release_all_cameras()


def _hello(manifest):
    return _envelope(
        "vision.hello",
        {
            "clientRole": "machine",
            "machineCode": "M001",
            "schemaVersion": manifest["schemaVersion"],
            "bundleVersion": manifest["bundleVersion"],
            "contractDigest": manifest["bundleDigest"],
            "capabilities": ["try_on_fast"],
        },
    )


def _start(attempt_id, reference):
    garment = _GarmentHandler.payload
    return _envelope(
        "vision.try_on.attempt.start",
        {
            "attemptId": attempt_id,
            "mode": "fast",
            "variantId": str(uuid4()),
            "garment": {
                "assetId": str(uuid4()),
                "reference": reference,
                "digest": f"sha256:{hashlib.sha256(garment).hexdigest()}",
                "contentType": "image/png",
                "byteSize": len(garment),
                "template": "tshirt_short_sleeve",
            },
        },
    )


def _envelope(message_type, payload):
    return {
        "protocol": "vem.vision.v2",
        "type": message_type,
        "messageId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


def test_v2_fast_attempt_accepts_generated_start_and_returns_tokenized_png(
    monkeypatch, garment_reference
):
    """The public V2 WS route owns one Fast attempt and returns a read grant."""
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    attempt_id = str(uuid4())
    garment = _GarmentHandler.payload
    hello = _hello(manifest)
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(hello)
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(start)
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = socket.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        result = completed["payload"]["result"]
        response = client.get(result["reference"])
        parsed_result = urlsplit(result["reference"])
        grant_path = f"{parsed_result.path}?{parsed_result.query}"
        head = client.head(grant_path)
        wrong_grant = client.get(f"{parsed_result.path}?token=wrong-token")
        missing_grant = client.get(parsed_result.path)
        extra_grant = client.get(f"{grant_path}&extra=true")
        duplicate_grant = client.get(f"{grant_path}&token=second")
        wrong_method = client.post(grant_path)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert head.status_code == 200
    assert head.headers["content-length"] == str(len(response.content))
    assert wrong_grant.status_code == missing_grant.status_code == 404
    assert extra_grant.status_code == duplicate_grant.status_code == 404
    assert wrong_method.status_code == 405
    assert cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is not None
    assert camera_manager.get_frame_source("front").status()["source"] == "recorded_video"


def test_v2_fast_attempt_keeps_ping_responsive_while_daemon_fetch_is_blocked(
    monkeypatch, garment_reference
):
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    attempt_id = str(uuid4())
    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            assert _GarmentHandler.entered.wait(timeout=2)
            socket.send_json(_envelope("vision.ping", {}))
            assert socket.receive_json()["type"] == "vision.pong"
            _GarmentHandler.release.set()
            assert socket.receive_json()["type"] == "vision.try_on.attempt.completed"


def test_v2_fast_attempt_replays_same_owner_active_attempt_without_new_terminal(
    monkeypatch, garment_reference
):
    """A transport retry of the same attempt joins the owner attempt instead of failing."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    attempt_id = str(uuid4())
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(start)
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            assert _GarmentHandler.entered.wait(timeout=2)

            socket.send_json(start)
            replayed = [socket.receive_json(), socket.receive_json()]

            assert [message["type"] for message in replayed] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.progress",
            ]
            assert all(message["payload"]["attemptId"] == attempt_id for message in replayed)

            _GarmentHandler.release.set()
            completed = socket.receive_json()

    assert completed["type"] == "vision.try_on.attempt.completed"
    assert completed["payload"]["attemptId"] == attempt_id


def test_v2_fast_attempt_second_socket_joins_and_both_receive_one_terminal(
    monkeypatch, garment_reference
):
    """A reconnecting transport is a subscriber, never a competing owner."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    attempt_id = str(uuid4())
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as owner, client.websocket_connect("/ws") as subscriber:
            owner.send_json(_hello(manifest))
            assert owner.receive_json()["type"] == "vision.ready"
            subscriber.send_json(_hello(manifest))
            assert subscriber.receive_json()["type"] == "vision.ready"

            owner.send_json(start)
            assert owner.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert owner.receive_json()["type"] == "vision.try_on.attempt.progress"
            assert _GarmentHandler.entered.wait(timeout=2)

            subscriber.send_json(start)
            replay = [subscriber.receive_json(), subscriber.receive_json()]
            assert [message["type"] for message in replay] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.progress",
            ]

            _GarmentHandler.release.set()
            owner_terminal = owner.receive_json()
            subscriber_terminal = subscriber.receive_json()

    assert owner_terminal["type"] == subscriber_terminal["type"] == "vision.try_on.attempt.completed"
    assert owner_terminal == subscriber_terminal


def test_v2_fast_attempt_second_socket_joins_without_cancelling_owner(
    monkeypatch, garment_reference
):
    """A retry on another WS is a subscriber, never a second owner or terminal."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    attempt_id = str(uuid4())
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as owner, client.websocket_connect("/ws") as retry:
            owner.send_json(_hello(manifest))
            retry.send_json(_hello(manifest))
            assert owner.receive_json()["type"] == "vision.ready"
            assert retry.receive_json()["type"] == "vision.ready"
            owner.send_json(start)
            assert owner.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert owner.receive_json()["type"] == "vision.try_on.attempt.progress"
            assert _GarmentHandler.entered.wait(timeout=2)

            retry.send_json(start)
            assert [retry.receive_json()["type"], retry.receive_json()["type"]] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.progress",
            ]
            # Losing the subscriber must not cancel the connection that owns work.
            retry.close()
            _GarmentHandler.release.set()
            completed = owner.receive_json()

    assert completed["type"] == "vision.try_on.attempt.completed"
    assert completed["payload"]["attemptId"] == attempt_id


def test_v2_fast_attempt_terminal_reconnect_replays_the_identical_grant(
    monkeypatch, garment_reference
):
    """Terminal records are canonical across a fresh WebSocket connection."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    attempt_id = str(uuid4())
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as owner:
            owner.send_json(_hello(manifest))
            assert owner.receive_json()["type"] == "vision.ready"
            owner.send_json(start)
            assert owner.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert owner.receive_json()["type"] == "vision.try_on.attempt.progress"
            terminal = owner.receive_json()

        with client.websocket_connect("/ws") as reconnect:
            reconnect.send_json(_hello(manifest))
            assert reconnect.receive_json()["type"] == "vision.ready"
            reconnect.send_json(start)
            replay = reconnect.receive_json()

    assert terminal["type"] == "vision.try_on.attempt.completed"
    assert replay == terminal


def test_v2_fast_unavailable_is_one_canonical_terminal_across_sockets_and_readiness_recovery(
    monkeypatch, garment_reference
):
    """A valid unavailable start is registered once, never rerun after recovery."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    readiness = {"cameraReady": False, "modelReady": True}
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: readiness)
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    attempt_id = str(uuid4())
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first.send_json(_hello(manifest))
            second.send_json(_hello(manifest))
            assert first.receive_json()["type"] == second.receive_json()["type"] == "vision.ready"
            first.send_json(start)
            terminal = first.receive_json()
            second.send_json(start)
            assert second.receive_json() == terminal

        readiness["cameraReady"] = True
        with client.websocket_connect("/ws") as recovered:
            recovered.send_json(_hello(manifest))
            assert recovered.receive_json()["type"] == "vision.ready"
            recovered.send_json(start)
            assert recovered.receive_json() == terminal

    assert terminal["type"] == "vision.try_on.attempt.failed"
    assert terminal["payload"] == {"attemptId": attempt_id, "reason": "fast_unavailable"}


def test_v2_fast_attempt_replacement_joins_old_worker_before_new_admission(
    monkeypatch, garment_reference
):
    """A different attempt cannot overtake its canceled worker's cleanup."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    first_id, second_id = str(uuid4()), str(uuid4())

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(first_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            assert _GarmentHandler.entered.wait(timeout=2)

            socket.send_json(_start(second_id, garment_reference))
            replaced = socket.receive_json()
            assert replaced["type"] == "vision.try_on.attempt.failed"
            assert replaced["payload"] == {"attemptId": first_id, "reason": "attempt_replaced"}

            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            _GarmentHandler.release.set()
            completed = socket.receive_json()

    assert completed["type"] == "vision.try_on.attempt.completed"
    assert completed["payload"]["attemptId"] == second_id


def test_v2_fast_result_store_rejects_self_too_large_without_publishing(monkeypatch):
    monkeypatch.setattr(vision_app, "_FAST_RESULT_MAX_BYTES", 8)
    image = _png_bytes()
    with pytest.raises(RuntimeError, match="fast_result_too_large"):
        vision_app._prepare_fast_result(str(uuid4()), image)
