import hashlib
import asyncio
import json
import multiprocessing
import os
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
from vision.directshow_broker import DirectShowCameraBroker
from vision.attempt_worker import FastRenderBroker


def _fast_block_first_broker_target(connection, config):
    counter = config["requestCounter"]
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            if command == "read":
                with counter.get_lock():
                    counter.value += 1
                    request_number = counter.value
                if request_number == 1:
                    while True:
                        threading.Event().wait(1.0)
                connection.send(("ok", {
                    "pid": os.getpid(),
                    "image": np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8),
                }))
    finally:
        connection.close()


def _fast_block_first_render_target(connection, counter):
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number == 1:
                while True:
                    threading.Event().wait(1.0)
            connection.send(("ok", _png_bytes()))
    finally:
        connection.close()


def _fast_pose_error_then_success_target(connection, counter):
    """A test-only worker fixture for public typed-attempt outcome coverage."""
    connection.send(("ready", {"pid": os.getpid(), "poseReady": True}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with counter.get_lock():
                counter.value += 1
                request_number = counter.value
            if request_number <= 3:
                connection.send(("pose_error", "PoseUnavailableError: C:\\internal\\model"))
            else:
                connection.send(("ok", _png_bytes()))
    finally:
        connection.close()


def _fast_block_then_fail_restart_target(connection, starts, requests):
    with starts.get_lock():
        starts.value += 1
        start_number = starts.value
    if start_number > 1:
        connection.close()
        return
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with requests.get_lock():
                requests.value += 1
            while True:
                threading.Event().wait(1.0)
    finally:
        connection.close()


def _fast_block_then_barrier_restart_target(
    connection,
    starts,
    requests,
    restart_entered,
    restart_release,
    restart_fails,
):
    with starts.get_lock():
        starts.value += 1
        start_number = starts.value
    if start_number > 1:
        restart_entered.set()
        restart_release.wait()
        if restart_fails:
            connection.close()
            return
    connection.send(("ready", {"pid": os.getpid()}))
    try:
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                connection.send(("ok", None))
                return
            with requests.get_lock():
                requests.value += 1
                request_number = requests.value
            if request_number == 1:
                while True:
                    threading.Event().wait(1.0)
            connection.send(("ok", _png_bytes()))
    finally:
        connection.close()


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
    """Public V2 completes a max garment against a recorded 720p frame."""
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
    garment_image = np.zeros((4096, 4096, 4), dtype=np.uint8)
    garment_image[384:3712, 640:3456] = (20, 120, 220, 220)
    ok, encoded_garment = cv2.imencode(".png", garment_image)
    assert ok
    garment = encoded_garment.tobytes()
    assert len(garment) <= 8 * 1024 * 1024
    monkeypatch.setattr(_GarmentHandler, "payload", garment)

    recorded_dimensions = []
    original_recorded_read = camera_manager.RecordedVideoFrameSource.read

    def read_recorded_720p(source, warmup_frames=None):
        recorded = original_recorded_read(source, warmup_frames=warmup_frames)
        recorded_dimensions.append(recorded.shape)
        return cv2.resize(recorded, (1280, 720), interpolation=cv2.INTER_LINEAR)

    monkeypatch.setattr(
        camera_manager.RecordedVideoFrameSource,
        "read",
        read_recorded_720p,
    )
    attempt_id = str(uuid4())
    hello = _hello(manifest)
    start = _start(attempt_id, garment_reference)

    with TestClient(vision_app.app) as client:
        assert vision_app._fast_render_broker.ready
        render_pid = vision_app._fast_render_broker.pid
        assert render_pid is not None
        with client.websocket_connect("/ws") as socket:
            socket.send_json(hello)
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(start)
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = socket.receive_json()
        assert vision_app._fast_render_broker.pid == render_pid

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

    assert vision_app._fast_render_broker.pid is None

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert head.status_code == 200
    assert head.headers["content-length"] == str(len(response.content))
    assert wrong_grant.status_code == missing_grant.status_code == 404
    assert extra_grant.status_code == duplicate_grant.status_code == 404
    assert wrong_method.status_code == 405
    result_image = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert result_image is not None
    assert result_image.shape == (720, 1280, 3)
    assert recorded_dimensions == [(360, 640, 3)]
    assert camera_manager.get_frame_source("front").status()["source"] == "recorded_video"


def test_v2_fast_completed_envelope_failure_has_only_failed_replay_and_no_result_grant(
    monkeypatch, garment_reference
):
    """A post-render contract failure cannot retain a staged Fast capability."""
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
    sentinel_token = "sentinel-result-token"
    original_prepare = vision_app._prepare_fast_result
    original_envelope = vision_app._generated_v2_envelope

    def prepare_with_sentinel(attempt_id, image):
        stored, public = original_prepare(attempt_id, image)
        reference = vision_app._fast_result_reference(attempt_id, sentinel_token)
        stored.update(token=sentinel_token, reference=reference)
        public.update(reference=reference)
        return stored, public

    def reject_completed(message_type, payload):
        if message_type == "vision.try_on.attempt.completed":
            raise ValueError("forced_completed_contract_failure")
        return original_envelope(message_type, payload)

    monkeypatch.setattr(vision_app, "_prepare_fast_result", prepare_with_sentinel)
    monkeypatch.setattr(vision_app, "_generated_v2_envelope", reject_completed)
    attempt_id = str(uuid4())

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            failed = socket.receive_json()

        assert failed["type"] == "vision.try_on.attempt.failed"
        assert failed["payload"] == {"attemptId": attempt_id, "reason": "fast_failed"}
        assert client.get(
            f"/v2/try-on/results/{attempt_id}?token={sentinel_token}"
        ).status_code == 404

        with client.websocket_connect("/ws") as replay_socket:
            replay_socket.send_json(_hello(manifest))
            assert replay_socket.receive_json()["type"] == "vision.ready"
            replay_socket.send_json(_start(attempt_id, garment_reference))
            assert replay_socket.receive_json()["type"] == "vision.try_on.attempt.failed"


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


def test_v2_fast_pose_failures_are_stable_terminals_without_worker_recovery(
    monkeypatch, garment_reference
):
    """Public failed attempts expose no result and retain the warmed worker."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=_fast_pose_error_then_success_target,
        target_args=(counter,),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)

    with TestClient(vision_app.app) as client:
        pid = broker.pid
        assert pid is not None
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            for _ in range(3):
                socket.send_json(_start(str(uuid4()), garment_reference))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
                failed = socket.receive_json()
                assert failed["type"] == "vision.try_on.attempt.failed"
                assert failed["payload"]["reason"] == "fast_failed"
                assert "result" not in failed["payload"]
                assert broker.pid == pid
            attempt_id = str(uuid4())
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = socket.receive_json()
            assert completed["type"] == "vision.try_on.attempt.completed"
            assert completed["payload"]["attemptId"] == attempt_id
            assert broker.pid == pid
            assert counter.value == 4

    assert broker.pid is None


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


def test_v2_replacement_restarts_render_then_next_attempts_complete(
    monkeypatch, garment_reference
):
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
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_first_render_target,
        target_args=(counter,),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    first_id, second_id, third_id = str(uuid4()), str(uuid4()), str(uuid4())

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        assert first_pid is not None
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(first_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            deadline = threading.Event()
            for _ in range(200):
                if counter.value == 1:
                    break
                deadline.wait(0.01)
            assert counter.value == 1

            socket.send_json(_start(second_id, garment_reference))
            replaced = socket.receive_json()
            assert replaced["payload"] == {
                "attemptId": first_id,
                "reason": "attempt_replaced",
            }
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            replacement_pid = broker.pid
            assert broker.ready
            assert replacement_pid is not None and replacement_pid != first_pid
            assert broker.active_request_count == 0
            assert {child.pid for child in multiprocessing.active_children()} == {
                replacement_pid
            }
            completed = socket.receive_json()

            assert completed["type"] == "vision.try_on.attempt.completed"
            assert completed["payload"]["attemptId"] == second_id

            socket.send_json(_start(third_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = socket.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == third_id
        assert broker.pid == replacement_pid
        assert counter.value == 3
        assert broker.active_request_count == 0

    assert broker.pid is None


def test_v2_restart_failure_is_live_stable_unavailable_without_second_worker(
    monkeypatch, garment_reference
):
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    starts = context.Value("i", 0)
    requests = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_then_fail_restart_target,
        target_args=(starts, requests),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": broker.ready,
        },
    )
    first_id, second_id, third_id, fourth_id = (
        str(uuid4()),
        str(uuid4()),
        str(uuid4()),
        str(uuid4()),
    )

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        assert first_pid is not None
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["payload"]["fastReady"] is True
            socket.send_json(_start(first_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            waiter = threading.Event()
            for _ in range(200):
                if requests.value == 1:
                    break
                waiter.wait(0.01)
            assert requests.value == 1

            socket.send_json(_start(second_id, garment_reference))
            replaced = socket.receive_json()
            assert replaced["payload"] == {
                "attemptId": first_id,
                "reason": "attempt_replaced",
            }
            unavailable = socket.receive_json()
            assert unavailable["type"] == "vision.try_on.attempt.failed"
            assert unavailable["payload"] == {
                "attemptId": second_id,
                "reason": "fast_unavailable",
            }

            assert not broker.ready
            assert broker.pid is None
            assert broker.active_request_count == 0
            assert multiprocessing.active_children() == []
            assert starts.value == 2

            socket.send_json(_start(third_id, garment_reference))
            unavailable = socket.receive_json()
            assert unavailable["payload"] == {
                "attemptId": third_id,
                "reason": "fast_unavailable",
            }
            assert starts.value == 2

        with client.websocket_connect("/ws") as fresh:
            fresh.send_json(_hello(manifest))
            ready = fresh.receive_json()
            assert ready["payload"]["fastReady"] is False
            assert ready["payload"]["businessReadinessDiagnostic"] == (
                "camera_unavailable"
            )
            fresh.send_json(_start(fourth_id, garment_reference))
            unavailable = fresh.receive_json()
            assert unavailable["payload"] == {
                "attemptId": fourth_id,
                "reason": "fast_unavailable",
            }

        assert starts.value == 2
        assert multiprocessing.active_children() == []

    assert broker.pid is None


def test_v2_duplicate_waits_for_atomic_failed_replacement_admission(
    monkeypatch, garment_reference
):
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    starts = context.Value("i", 0)
    requests = context.Value("i", 0)
    restart_entered = context.Event()
    restart_release = context.Event()
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_then_barrier_restart_target,
        target_args=(
            starts,
            requests,
            restart_entered,
            restart_release,
            True,
        ),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": broker.ready,
        },
    )
    first_id, replacement_id = str(uuid4()), str(uuid4())
    owner_messages = []
    duplicate_messages = []
    replaced_seen = threading.Event()
    owner_done = threading.Event()
    duplicate_done = threading.Event()

    with TestClient(vision_app.app) as client:
        with (
            client.websocket_connect("/ws") as owner,
            client.websocket_connect("/ws") as duplicate,
        ):
            owner.send_json(_hello(manifest))
            duplicate.send_json(_hello(manifest))
            assert owner.receive_json()["payload"]["fastReady"] is True
            assert duplicate.receive_json()["payload"]["fastReady"] is True
            owner.send_json(_start(first_id, garment_reference))
            assert owner.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert owner.receive_json()["type"] == "vision.try_on.attempt.progress"
            waiter = threading.Event()
            for _ in range(200):
                if requests.value == 1:
                    break
                waiter.wait(0.01)
            assert requests.value == 1

            def receive_owner():
                owner_messages.append(owner.receive_json())
                replaced_seen.set()
                owner_messages.append(owner.receive_json())
                owner_done.set()

            owner_reader = threading.Thread(target=receive_owner, daemon=True)
            owner_reader.start()
            owner.send_json(_start(replacement_id, garment_reference))
            assert restart_entered.wait(timeout=2)
            try:
                assert replaced_seen.wait(timeout=0.5)
                duplicate.send_json(_start(replacement_id, garment_reference))

                def receive_duplicate():
                    duplicate_messages.append(duplicate.receive_json())
                    duplicate_done.set()

                duplicate_reader = threading.Thread(
                    target=receive_duplicate, daemon=True
                )
                duplicate_reader.start()
            finally:
                restart_release.set()

            assert owner_done.wait(timeout=3)
            assert duplicate_done.wait(timeout=3)
            owner_reader.join(timeout=1)
            duplicate_reader.join(timeout=1)

            owner.send_json(_start(replacement_id, garment_reference))
            replayed_terminal = owner.receive_json()

        expected_replaced = {
            "attemptId": first_id,
            "reason": "attempt_replaced",
        }
        assert owner_messages[0]["payload"] == expected_replaced
        expected_unavailable = owner_messages[1]
        assert expected_unavailable["type"] == "vision.try_on.attempt.failed"
        assert expected_unavailable["payload"] == {
            "attemptId": replacement_id,
            "reason": "fast_unavailable",
        }
        assert duplicate_messages == [expected_unavailable]
        assert replayed_terminal == expected_unavailable
        assert starts.value == 2
        assert requests.value == 1
        assert broker.pid is None
        assert broker.active_request_count == 0
        assert multiprocessing.active_children() == []


def test_v2_duplicate_replays_only_after_atomic_ready_replacement_admission(
    monkeypatch, garment_reference
):
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    starts = context.Value("i", 0)
    requests = context.Value("i", 0)
    restart_entered = context.Event()
    restart_release = context.Event()
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_then_barrier_restart_target,
        target_args=(
            starts,
            requests,
            restart_entered,
            restart_release,
            False,
        ),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": broker.ready,
        },
    )
    first_id, replacement_id = str(uuid4()), str(uuid4())
    owner_messages = []
    duplicate_messages = []
    replaced_seen = threading.Event()
    owner_done = threading.Event()
    duplicate_message_seen = threading.Event()
    duplicate_done = threading.Event()

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        with (
            client.websocket_connect("/ws") as owner,
            client.websocket_connect("/ws") as duplicate,
        ):
            owner.send_json(_hello(manifest))
            duplicate.send_json(_hello(manifest))
            assert owner.receive_json()["payload"]["fastReady"] is True
            assert duplicate.receive_json()["payload"]["fastReady"] is True
            owner.send_json(_start(first_id, garment_reference))
            assert owner.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert owner.receive_json()["type"] == "vision.try_on.attempt.progress"
            waiter = threading.Event()
            for _ in range(200):
                if requests.value == 1:
                    break
                waiter.wait(0.01)
            assert requests.value == 1

            def receive_owner():
                for _ in range(4):
                    owner_messages.append(owner.receive_json())
                    if len(owner_messages) == 1:
                        replaced_seen.set()
                owner_done.set()

            owner_reader = threading.Thread(target=receive_owner, daemon=True)
            owner_reader.start()
            owner.send_json(_start(replacement_id, garment_reference))
            assert restart_entered.wait(timeout=2)
            assert replaced_seen.wait(timeout=0.5)
            duplicate.send_json(_start(replacement_id, garment_reference))

            def receive_duplicate():
                for _ in range(3):
                    duplicate_messages.append(duplicate.receive_json())
                    duplicate_message_seen.set()
                duplicate_done.set()

            duplicate_reader = threading.Thread(
                target=receive_duplicate, daemon=True
            )
            duplicate_reader.start()
            assert not duplicate_message_seen.wait(timeout=0.1)
            restart_release.set()

            assert owner_done.wait(timeout=5)
            assert duplicate_done.wait(timeout=5)
            owner_reader.join(timeout=1)
            duplicate_reader.join(timeout=1)

        assert owner_messages[0]["payload"] == {
            "attemptId": first_id,
            "reason": "attempt_replaced",
        }
        assert [message["type"] for message in owner_messages[1:]] == [
            "vision.try_on.attempt.accepted",
            "vision.try_on.attempt.progress",
            "vision.try_on.attempt.completed",
        ]
        assert duplicate_messages == owner_messages[1:]
        replacement_pid = broker.pid
        assert replacement_pid is not None and replacement_pid != first_pid
        assert starts.value == 2
        assert requests.value == 2
        assert broker.active_request_count == 0
        assert {child.pid for child in multiprocessing.active_children()} == {
            replacement_pid
        }

    assert broker.pid is None


def test_v2_disconnect_restarts_render_and_new_connection_completes(
    monkeypatch, garment_reference
):
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
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_first_render_target,
        target_args=(counter,),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    retry_id = str(uuid4())

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        assert first_pid is not None
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(str(uuid4()), garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            waiter = threading.Event()
            for _ in range(200):
                if counter.value == 1:
                    break
                waiter.wait(0.01)
            assert counter.value == 1
            socket.close()

        waiter = threading.Event()
        for _ in range(300):
            if broker.ready and broker.active_request_count == 0:
                break
            waiter.wait(0.01)
        replacement_pid = broker.pid
        assert broker.ready
        assert replacement_pid is not None and replacement_pid != first_pid
        assert broker.active_request_count == 0

        with client.websocket_connect("/ws") as retry:
            retry.send_json(_hello(manifest))
            ready = retry.receive_json()
            assert ready["type"] == "vision.ready"
            assert ready["payload"]["fastReady"] is True
            retry.send_json(_start(retry_id, garment_reference))
            assert retry.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert retry.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = retry.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == retry_id
        assert broker.pid == replacement_pid
        assert counter.value == 2

    assert broker.pid is None


def test_v2_timeout_restarts_render_and_new_connection_completes(
    monkeypatch, garment_reference
):
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": vision_app._fast_render_broker.ready,
        },
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_FAST_ATTEMPT_TIMEOUT_SECONDS", 0.1)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=_fast_block_first_render_target,
        target_args=(counter,),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    timed_out_id, retry_id = str(uuid4()), str(uuid4())

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        assert first_pid is not None
        with client.websocket_connect("/ws") as first:
            first.send_json(_hello(manifest))
            assert first.receive_json()["payload"]["fastReady"] is True
            first.send_json(_start(timed_out_id, garment_reference))
            assert first.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert first.receive_json()["type"] == "vision.try_on.attempt.progress"
            failed = first.receive_json()

        assert failed["type"] == "vision.try_on.attempt.failed"
        assert failed["payload"] == {
            "attemptId": timed_out_id,
            "reason": "fast_failed",
        }
        replacement_pid = broker.pid
        assert broker.ready
        assert replacement_pid is not None and replacement_pid != first_pid
        assert broker.active_request_count == 0
        assert {child.pid for child in multiprocessing.active_children()} == {
            replacement_pid
        }

        with client.websocket_connect("/ws") as retry:
            retry.send_json(_hello(manifest))
            ready = retry.receive_json()
            assert ready["payload"]["fastReady"] is True
            retry.send_json(_start(retry_id, garment_reference))
            assert retry.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert retry.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = retry.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == retry_id
        assert broker.pid == replacement_pid
        assert counter.value == 2

    assert broker.pid is None


def test_v2_fast_attempt_reads_front_frame_in_parent_process(monkeypatch, garment_reference):
    """Fast must not spawn a child that opens the front camera device."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    parent_pid = os.getpid()
    read_pids = []

    def read_front(role, warmup_frames=None):
        assert vision_app.get_front_camera_owner()["owner"] == "tryon_frontend"
        read_pids.append((os.getpid(), role, warmup_frames))
        return np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8), {"source": "dshow"}

    async def render(frame, garment_png, *, digest, template, timeout, broker):
        assert os.getpid() == parent_pid
        assert garment_png == _GarmentHandler.payload
        assert digest.startswith("sha256:")
        assert template == "tshirt_short_sleeve"
        assert broker is vision_app._fast_render_broker
        return _png_bytes()

    monkeypatch.setattr(vision_app, "read_camera_with_source", read_front)
    monkeypatch.setattr(vision_app, "render_attempt_frame", render)
    attempt_id = str(uuid4())

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            assert socket.receive_json()["type"] == "vision.try_on.attempt.progress"
            completed = socket.receive_json()

    assert completed["type"] == "vision.try_on.attempt.completed"
    assert read_pids == [(parent_pid, "front", 1)]
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


def test_v2_fast_attempt_does_not_release_legacy_tryon_owner(monkeypatch):
    """Fast uses its attempt lease and cannot reuse the legacy try-on owner token."""
    owner = vision_app.acquire_front_camera(
        "tryon_frontend",
        reason="try_on_start:legacy-session",
        lease_token="tryon_frontend",
    )
    assert owner["ok"]

    async def is_current(_receipt):
        return True

    monkeypatch.setattr(vision_app._fast_attempt_registry, "is_current", is_current)
    receipt = vision_app.AttemptReceipt(
        attempt_id=str(uuid4()),
        owner_token="owner-token",
        generation=7,
    )

    try:
        with pytest.raises(vision_app.GarmentFetchError, match="front_camera_busy"):
            asyncio.run(vision_app._read_fast_front_frame(receipt, timeout=0.1))

        assert vision_app.get_front_camera_owner()["owner"] == "tryon_frontend"
        assert vision_app.get_front_camera_owner()["leaseToken"] == "tryon_frontend"
    finally:
        vision_app.release_front_camera(
            "tryon_frontend",
            reason="test_cleanup",
            lease_token="tryon_frontend",
        )


def test_v2_fast_attempt_uses_camera_manager_dshow_broker_not_app_worker(monkeypatch):
    """The production dshow branch enters camera_manager and opens one broker boundary."""
    parent_pid = os.getpid()
    events = []

    class Candidate:
        index = 9
        backend = "dshow"
        stable_id = "front-stable"

    class Maintenance:
        def resolve(self, role):
            assert role == "front"
            return Candidate()

        def refresh_after_read_failure(self):
            raise AssertionError("happy path should not refresh")

    class Broker:
        def __init__(self, role, config):
            events.append(("open", os.getpid(), role, config["stableId"]))
            self.config = dict(config)
            self.config["logicalRole"] = role
            self.last_pid = 4242

        def read(self, warmup_frames=None, *, timeout=None):
            events.append(("read", os.getpid(), warmup_frames, timeout))
            return np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8)

        async def read_async(self, warmup_frames=None, *, timeout=None):
            return self.read(warmup_frames=warmup_frames, timeout=timeout)

        def last_frame(self):
            return {"source": "dshow", "brokerPid": self.last_pid}

        def release(self):
            events.append(("release", os.getpid()))
            return True

        def assert_dead(self):
            return True

    async def is_current(_receipt):
        return True

    monkeypatch.setattr(vision_app._fast_attempt_registry, "is_current", is_current)
    monkeypatch.setattr(vision_app.settings, "FRONT_CAMERA_CONFIG", {
        "role": "profile_tryon",
        "source": "dshow",
        "keep_open": True,
    })
    monkeypatch.setattr(camera_manager, "get_camera_maintenance", lambda: Maintenance())
    monkeypatch.setattr(camera_manager, "DirectShowCameraBroker", Broker)
    camera_manager.release_all_cameras()

    receipt = vision_app.AttemptReceipt(
        attempt_id=str(uuid4()),
        owner_token="owner-token",
        generation=8,
    )

    try:
        frame, source = asyncio.run(vision_app._read_fast_front_frame(receipt, timeout=1.0))
    finally:
        vision_app.release_front_camera(
            "tryon_frontend",
            reason="test_cleanup",
            lease_token=f"fast:{receipt.attempt_id}:{receipt.generation}:{receipt.owner_token}",
        )
        camera_manager.release_all_cameras()

    assert frame.shape == (80, 60, 3)
    assert source == {"source": "dshow", "brokerPid": 4242}
    assert events[0] == ("open", parent_pid, "front", "front-stable")
    assert events[1][0:3] == ("read", parent_pid, 1)
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


def test_fast_blocked_production_broker_cancel_keeps_loop_live_joins_and_restarts(monkeypatch):
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = DirectShowCameraBroker(
        "front",
        {
            "role": "profile_tryon",
            "index": 9,
            "backend": "dshow",
            "stableId": "front-stable",
            "keep_open": True,
            "requestCounter": counter,
        },
        context=context,
        target=_fast_block_first_broker_target,
    )

    class Candidate:
        index = 9
        backend = "dshow"
        stable_id = "front-stable"

    class Maintenance:
        def resolve(self, role):
            assert role == "front"
            return Candidate()

        def refresh_after_read_failure(self):
            return None

    class Registry:
        def __init__(self):
            self.cancel_event = None

        async def is_current(self, _receipt):
            return not self.cancel_event.is_set()

        async def cancel_event_for(self, _receipt):
            return self.cancel_event

    registry = Registry()
    monkeypatch.setattr(vision_app, "_fast_attempt_registry", registry)
    monkeypatch.setattr(vision_app.settings, "FRONT_CAMERA_CONFIG", {
        "role": "profile_tryon", "source": "dshow", "keep_open": True,
    })
    monkeypatch.setattr(camera_manager, "get_camera_maintenance", lambda: Maintenance())
    monkeypatch.setattr(camera_manager, "DirectShowCameraBroker", lambda _role, _config: broker)
    camera_manager.release_all_cameras()

    async def scenario():
        registry.cancel_event = asyncio.Event()
        first = vision_app.AttemptReceipt(str(uuid4()), "owner-1", 1)
        read_task = asyncio.create_task(
            vision_app._read_fast_front_frame(first, timeout=15.0)
        )
        deadline = asyncio.get_running_loop().time() + 1.0
        while counter.value < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.002)
        ticks = 0
        tick_deadline = asyncio.get_running_loop().time() + 0.05
        while asyncio.get_running_loop().time() < tick_deadline:
            ticks += 1
            await asyncio.sleep(0.002)
        registry.cancel_event.set()
        with pytest.raises(vision_app.GarmentFetchError, match="attempt_replaced"):
            await asyncio.wait_for(read_task, timeout=1.0)
        assert ticks >= 10
        assert broker.assert_dead()
        assert broker.active_request_count == 0

        registry.cancel_event = asyncio.Event()
        second = vision_app.AttemptReceipt(str(uuid4()), "owner-2", 2)
        frame, source = await vision_app._read_fast_front_frame(second, timeout=1.0)
        assert frame.shape == (80, 60, 3)
        assert source["brokerPid"] is not None

    try:
        asyncio.run(scenario())
    finally:
        broker.release()
        with camera_manager._streams_lock:
            camera_manager._dshow_brokers.pop("front", None)
    assert broker.assert_dead()
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


def test_v2_fast_result_store_rejects_self_too_large_without_publishing(monkeypatch):
    monkeypatch.setattr(vision_app, "_FAST_RESULT_MAX_BYTES", 8)
    image = _png_bytes()
    with pytest.raises(RuntimeError, match="fast_result_too_large"):
        vision_app._prepare_fast_result(str(uuid4()), image)
