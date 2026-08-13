import hashlib
import asyncio
import ast
from decimal import Decimal
import importlib
import inspect
import json
import multiprocessing
import os
import threading
import time
from urllib.parse import urlsplit
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests import fast_worker_fixture

import app as vision_app
from vision import camera_manager
from vision.config import settings
from vision.directshow_broker import DirectShowCameraBroker
from vision.attempt_worker import FastRenderBroker
from vision.acquisition_observer import AcquisitionObservation


def test_fast_worker_fixture_is_spawn_safe():
    """Broker child targets must not import the application test module on spawn."""
    module = importlib.import_module("tests.fast_worker_fixture")
    tree = ast.parse(Path(module.__file__).read_text("utf-8"))
    top_level_imports = {
        alias.name.split(".")[0]
        for statement in tree.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
        for alias in statement.names
    }
    assert top_level_imports <= {"base64", "os", "threading", "time"}
    assert {
        "block_first_render",
        "block_first_directshow",
        "block_then_fail_restart_render",
        "block_then_barrier_restart_render",
        "block_then_ready_barrier_directshow",
    } <= set(vars(module))
    directshow_source = inspect.getsource(module.block_first_directshow)
    assert directshow_source.index('connection.send(("ready"') < directshow_source.index(
        "import numpy as np"
    )


def _active_child_pids_except_acquisition_observer():
    observer = vision_app._acquisition_observer
    observer_pid = observer.pid if observer is not None else None
    return {
        child.pid
        for child in multiprocessing.active_children()
        if child.pid != observer_pid
    }


async def _wait_for_raw_value(value, *, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while not value.value:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        await asyncio.sleep(min(0.01, remaining))


def test_raw_value_wait_timeout_returns_from_asyncio_run_without_executor(monkeypatch):
    """A missed child-entry signal must not strand asyncio.run on executor shutdown."""
    entered = multiprocessing.get_context("spawn").RawValue("b", 0)

    async def unexpected_to_thread(*_args, **_kwargs):
        raise AssertionError("entry wait must not use the default executor")

    monkeypatch.setattr(asyncio, "to_thread", unexpected_to_thread)

    async def scenario():
        await _wait_for_raw_value(entered, timeout=0.02)

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(scenario())
    assert time.monotonic() - started < 0.5


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
    server.server_close()
    thread.join()


def _configure_recorded_front(monkeypatch):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "recorded-video"
    monkeypatch.setattr(
        settings,
        "FRONT_CAMERA_CONFIG",
        {
            "role": "profile_fast_try_on",
            "source": "recorded_video",
            "video_path": str(fixture_root / "man-front.mp4"),
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


def _hello_with_capabilities(manifest, capabilities):
    message = _hello(manifest)
    message["payload"]["capabilities"] = list(capabilities)
    return message


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


class _ReadyFastBroker:
    ready = True
    pose_ready = True

    async def start(self):
        return None

    def quiesce(self):
        return None

    async def shutdown(self):
        return None


class _SingleAlignedObserver:
    ready = True
    pose_ready = True
    fatal_error = None
    pid = None
    active_request_count = 0
    assert_dead = True

    async def start(self):
        return None

    async def observe(self, _frame, *, timeout=15.0):
        return AcquisitionObservation(b"jpeg", "single", True)

    async def wait_idle(self, *, timeout=None):
        return True

    async def shutdown(self):
        return None


class _FatalAcquisitionObserver:
    ready = False
    fatal_error = "spawn_failed"
    pid = None
    active_request_count = 0
    assert_dead = True

    async def start(self):
        return None

    async def observe(self, _frame, *, timeout=15.0):
        raise AssertionError("fatal observer must be rechecked before accepted admission")

    async def wait_idle(self):
        return None

    async def shutdown(self):
        return None


class _DeadlineRecordingObserver:
    ready = True
    fatal_error = None
    pid = None
    active_request_count = 0
    assert_dead = True

    def __init__(self):
        self.timeouts = []

    async def start(self):
        return None

    async def observe(self, _frame, *, timeout):
        self.timeouts.append(timeout)
        return AcquisitionObservation(b"jpeg", "single", False)

    async def wait_idle(self):
        return None

    async def shutdown(self):
        return None


def _envelope(message_type, payload):
    return {
        "protocol": "vem.vision.v2",
        "type": message_type,
        "messageId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


def _await_generating(socket):
    """Preserve Fast terminal assertions while crossing the V2 acquisition stage."""
    while True:
        message = socket.receive_json()
        if message["type"] == "vision.try_on.attempt.generating":
            return message
        assert message["type"] == "vision.try_on.attempt.acquiring"


def await_no_active_fast_attempt():
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if asyncio.run(vision_app._fast_attempt_registry.active_attempt_id()) is None:
            return True
        time.sleep(0.002)
    return False


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
            _await_generating(socket)
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
    # Acquisition consumes three stable production source frames before the
    # fixed captured frame enters rendering; the preview never supplies it.
    assert len(recorded_dimensions) >= 3
    assert set(recorded_dimensions) == {(768, 512, 3)}
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
            _await_generating(socket)
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
            _await_generating(socket)
            assert _GarmentHandler.entered.wait(timeout=2)
            socket.send_json(_envelope("vision.ping", {}))
            assert socket.receive_json()["type"] == "vision.pong"
            _GarmentHandler.release.set()
            assert socket.receive_json()["type"] == "vision.try_on.attempt.completed"


def test_v2_top_departure_cancels_active_generated_attempt_without_late_completion(
    monkeypatch, garment_reference
):
    """A production presence departure cancels the active attempt through the public WS."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", True)
    monkeypatch.setattr(vision_app.settings, "MOCK_SCENARIO", "departure-test")
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_INTERVAL_MS", 5)
    _configure_recorded_front(monkeypatch)
    _GarmentHandler.release.clear()
    depart = threading.Event()
    departed_once = threading.Event()

    def collect_departure(_status, _ambient, include_departure):
        if depart.is_set() and include_departure and not departed_once.is_set():
            departed_once.set()
            return {
                "message_type": "vision.person_departed",
                "payload": {
                    "source": "top",
                    "reason": "no_person",
                    "occupancy": {"state": "none"},
                },
            }
        return None

    monkeypatch.setattr(vision_app, "collect_profile_update", collect_departure)
    attempt_id = str(uuid4())
    messages = []

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(
                _hello_with_capabilities(
                    manifest, ["try_on_fast", "person_departed"]
                )
            )
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(socket)
            assert _GarmentHandler.entered.wait(timeout=2)

            depart.set()
            while len(messages) < 2:
                messages.append(socket.receive_json())
                if any(
                    message["type"] == "vision.try_on.attempt.canceled"
                    for message in messages
                ) and any(
                    message["type"] == "vision.person_departed"
                    for message in messages
                ):
                    break

            _GarmentHandler.release.set()

        assert await_no_active_fast_attempt()
        assert vision_app.get_front_camera_owner()["owner"] == "idle"

    canceled = [
        message for message in messages
        if message["type"] == "vision.try_on.attempt.canceled"
    ]
    assert canceled == [
        {
            **canceled[0],
            "payload": {"attemptId": attempt_id, "reason": "departure"},
        }
    ]
    assert [message["type"] for message in messages].count(
        "vision.try_on.attempt.completed"
    ) == 0
    assert departed_once.is_set()


def test_v2_route_leave_cancel_during_generation_fences_late_result_and_releases_resources(
    monkeypatch, garment_reference
):
    """Machine route leave is an explicit public cancel, not a hidden disconnect."""
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
            _await_generating(socket)
            assert _GarmentHandler.entered.wait(timeout=2)
            socket.send_json(
                _envelope(
                    "vision.try_on.attempt.cancel",
                    {"attemptId": attempt_id, "reason": "route_leave"},
                )
            )
            canceled = socket.receive_json()
            _GarmentHandler.release.set()

        assert await_no_active_fast_attempt()
        assert vision_app.get_front_camera_owner()["owner"] == "idle"

        with client.websocket_connect("/ws") as replay:
            replay.send_json(_hello(manifest))
            assert replay.receive_json()["type"] == "vision.ready"
            replay.send_json(_start(attempt_id, garment_reference))
            replayed = replay.receive_json()

    assert canceled["type"] == "vision.try_on.attempt.canceled"
    assert canceled["payload"] == {"attemptId": attempt_id, "reason": "route_leave"}
    assert replayed == canceled


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
                _await_generating(socket)
                failed = socket.receive_json()
                assert failed["type"] == "vision.try_on.attempt.failed"
                assert failed["payload"]["reason"] == "fast_failed"
                assert "result" not in failed["payload"]
                assert broker.pid == pid
            attempt_id = str(uuid4())
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(socket)
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
            _await_generating(socket)
            assert _GarmentHandler.entered.wait(timeout=2)

            socket.send_json(start)
            replayed = [socket.receive_json(), socket.receive_json()]

            assert [message["type"] for message in replayed] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.generating",
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
            _await_generating(owner)
            assert _GarmentHandler.entered.wait(timeout=2)

            subscriber.send_json(start)
            replay = [subscriber.receive_json(), subscriber.receive_json()]
            assert [message["type"] for message in replay] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.generating",
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
            _await_generating(owner)
            assert _GarmentHandler.entered.wait(timeout=2)

            retry.send_json(start)
            assert [retry.receive_json()["type"], retry.receive_json()["type"]] == [
                "vision.try_on.attempt.accepted",
                "vision.try_on.attempt.generating",
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
            _await_generating(owner)
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
            _await_generating(socket)
            assert _GarmentHandler.entered.wait(timeout=2)

            socket.send_json(_start(second_id, garment_reference))
            replaced = socket.receive_json()
            assert replaced["type"] == "vision.try_on.attempt.canceled"
            assert replaced["payload"] == {"attemptId": first_id, "reason": "replaced"}

            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(socket)
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
        target=fast_worker_fixture.block_first_render,
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
            _await_generating(socket)
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
                "reason": "replaced",
            }
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(socket)
            replacement_pid = broker.pid
            assert broker.ready
            assert replacement_pid is not None and replacement_pid != first_pid
            assert broker.active_request_count == 0
            assert _active_child_pids_except_acquisition_observer() == {
                replacement_pid
            }
            completed = socket.receive_json()

            assert completed["type"] == "vision.try_on.attempt.completed"
            assert completed["payload"]["attemptId"] == second_id

            socket.send_json(_start(third_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(socket)
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
        target=fast_worker_fixture.block_then_fail_restart_render,
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
            _await_generating(socket)
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
                "reason": "replaced",
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
            assert _active_child_pids_except_acquisition_observer() == set()
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
        assert _active_child_pids_except_acquisition_observer() == set()

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
        target=fast_worker_fixture.block_then_barrier_restart_render,
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
            _await_generating(owner)
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
            "reason": "replaced",
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
        assert _active_child_pids_except_acquisition_observer() == set()


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
        target=fast_worker_fixture.block_then_barrier_restart_render,
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
            _await_generating(owner)
            waiter = threading.Event()
            for _ in range(200):
                if requests.value == 1:
                    break
                waiter.wait(0.01)
            assert requests.value == 1

            def receive_owner():
                for _ in range(6):
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
                for _ in range(5):
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
            "reason": "replaced",
        }
        replacement_trace = [message["type"] for message in owner_messages[1:]]
        assert replacement_trace[0] == "vision.try_on.attempt.accepted"
        assert replacement_trace[-1] == "vision.try_on.attempt.completed"
        assert replacement_trace.count("vision.try_on.attempt.generating") == 1
        assert all(
            message_type == "vision.try_on.attempt.acquiring"
            for message_type in replacement_trace[1:-2]
        )
        assert duplicate_messages == owner_messages[1:]
        replacement_pid = broker.pid
        assert replacement_pid is not None and replacement_pid != first_pid
        assert starts.value == 2
        assert requests.value == 2
        assert broker.active_request_count == 0
        assert _active_child_pids_except_acquisition_observer() == {
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
        target=fast_worker_fixture.block_first_render,
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
            _await_generating(socket)
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
            _await_generating(retry)
            completed = retry.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == retry_id
        assert broker.pid == replacement_pid
        assert counter.value == 2

    assert broker.pid is None


def test_v2_start_rechecks_acquisition_observer_after_ready_before_accepting_attempt(
    monkeypatch, garment_reference
):
    """A cached hello cannot admit an attempt after the observer has gone fatal."""
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _ReadyFastBroker())
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": True,
            "fastPoseReady": True,
            "acquisitionObserverReady": True,
        },
    )
    vision_app._acquisition_observer = _FatalAcquisitionObserver()
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello(manifest))
                ready = socket.receive_json()
                assert ready["payload"]["fastReady"] is True

                socket.send_json(_start(str(uuid4()), garment_reference))
                rejected = socket.receive_json()

        assert rejected["type"] == "vision.try_on.attempt.failed"
        assert rejected["payload"]["reason"] == "fast_unavailable"
    finally:
        vision_app._acquisition_observer = None


def test_v2_acquisition_observer_uses_remaining_attempt_deadline(monkeypatch, garment_reference):
    """Observation must share the attempt deadline instead of adding its own 15s window."""
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _ReadyFastBroker())
    monkeypatch.setattr(vision_app, "_FAST_ATTEMPT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(vision_app, "_ACQUISITION_POLL_SECONDS", 0.005)
    observer = _DeadlineRecordingObserver()
    vision_app._acquisition_observer = observer
    frame = np.zeros((64, 48, 3), dtype=np.uint8)
    monkeypatch.setattr(
        vision_app,
        "read_camera_with_source",
        lambda *_args, **_kwargs: (frame, {"source": "recorded_video"}),
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": True,
            "fastPoseReady": True,
            "acquisitionObserverReady": True,
        },
    )

    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello(manifest))
                assert socket.receive_json()["type"] == "vision.ready"

                socket.send_json(_start(str(uuid4()), garment_reference))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                while True:
                    terminal = socket.receive_json()
                    if terminal["type"] in {
                        "vision.try_on.attempt.canceled",
                        "vision.try_on.attempt.failed",
                    }:
                        break

        assert terminal["type"] == "vision.try_on.attempt.canceled"
        assert terminal["payload"]["reason"] == "timeout"
        assert observer.timeouts
        requested_deadline = Decimal("0.05")
        float_clock_epsilon = Decimal("0.000000001")
        assert all(
            Decimal(str(timeout)) <= requested_deadline + float_clock_epsilon
            for timeout in observer.timeouts
        )
    finally:
        vision_app._acquisition_observer = None


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
    # Keep real acquisition on its normal deadline.  The render phase itself
    # gets a deliberately short first-worker deadline, so this test never
    # mistakes hosted spawn/load time for an attempt timeout.
    monkeypatch.setattr(vision_app, "_FAST_ATTEMPT_TIMEOUT_SECONDS", 15.0)
    _configure_recorded_front(monkeypatch)
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    broker = FastRenderBroker(
        context=context,
        target=fast_worker_fixture.block_first_render,
        target_args=(counter,),
    )
    monkeypatch.setattr(vision_app, "_fast_render_broker", broker)
    real_render = vision_app.render_attempt_frame

    async def render_with_first_worker_deadline(*args, **kwargs):
        if counter.value == 0:
            kwargs["timeout"] = 0.1
        return await real_render(*args, **kwargs)

    monkeypatch.setattr(vision_app, "render_attempt_frame", render_with_first_worker_deadline)
    timed_out_id, retry_id = str(uuid4()), str(uuid4())

    with TestClient(vision_app.app) as client:
        first_pid = broker.pid
        assert first_pid is not None
        with client.websocket_connect("/ws") as first:
            first.send_json(_hello(manifest))
            assert first.receive_json()["payload"]["fastReady"] is True
            first.send_json(_start(timed_out_id, garment_reference))
            assert first.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(first)
            failed = first.receive_json()

        assert failed["type"] == "vision.try_on.attempt.canceled"
        assert failed["payload"] == {
            "attemptId": timed_out_id,
            "reason": "timeout",
        }
        replacement_pid = broker.pid
        assert broker.ready
        assert replacement_pid is not None and replacement_pid != first_pid
        assert broker.active_request_count == 0
        assert _active_child_pids_except_acquisition_observer() == {
            replacement_pid
        }

        with client.websocket_connect("/ws") as retry:
            retry.send_json(_hello(manifest))
            ready = retry.receive_json()
            assert ready["payload"]["fastReady"] is True
            retry.send_json(_start(retry_id, garment_reference))
            assert retry.receive_json()["type"] == "vision.try_on.attempt.accepted"
            _await_generating(retry)
            completed = retry.receive_json()

        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == retry_id
        assert broker.pid == replacement_pid
        assert counter.value == 2

    assert broker.pid is None


def test_v2_fast_attempt_reads_front_frame_in_parent_process(monkeypatch, garment_reference):
    """Acquisition must not spawn a child that opens the front camera device."""
    manifest = json.loads((Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text("utf-8"))
    monkeypatch.setattr(vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True})
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    test_broker = _ReadyFastBroker()
    monkeypatch.setattr(vision_app, "_fast_render_broker", test_broker)
    parent_pid = os.getpid()
    read_pids = []

    def read_front(role, warmup_frames=None):
        assert vision_app.get_front_camera_owner()["owner"] == "try_on_attempt"
        read_pids.append((os.getpid(), role, warmup_frames))
        frame = cv2.imread(
            str(Path(__file__).parents[1] / "fixtures/recorded-video/sources/person-man-front.png")
        )
        assert frame is not None
        return cv2.resize(frame, (512, 768)), {"source": "dshow"}

    async def render(frame, garment_png, *, digest, template, timeout, broker):
        assert os.getpid() == parent_pid
        assert garment_png == _GarmentHandler.payload
        assert digest.startswith("sha256:")
        assert template == "tshirt_short_sleeve"
        assert broker is test_broker
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
            _await_generating(socket)
            completed = socket.receive_json()

    assert completed["type"] == "vision.try_on.attempt.completed"
    assert read_pids == [(parent_pid, "front", 1)] * vision_app._ACQUISITION_STABLE_FRAMES
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


def test_v2_fast_attempt_respects_attempt_owner_lease(monkeypatch):
    """Acquisition cannot replace an existing attempt camera lease."""
    owner = vision_app.acquire_front_camera(
        "try_on_attempt",
        reason="try_on_acquisition:existing",
        lease_token="try-on:existing",
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
            asyncio.run(vision_app._read_attempt_front_frame(receipt, timeout=0.1))

        assert vision_app.get_front_camera_owner()["owner"] == "try_on_attempt"
        assert vision_app.get_front_camera_owner()["leaseToken"] == "try-on:existing"
    finally:
        vision_app.release_front_camera(
            "try_on_attempt",
            reason="test_cleanup",
            lease_token="try-on:existing",
        )


def test_v2_capture_preview_close_failure_releases_lease_and_commits_one_failed_terminal(
    monkeypatch, garment_reference
):
    """Public WS capture cleanup cannot let preview close failure skip lease release/terminal."""
    manifest = json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _ReadyFastBroker())
    monkeypatch.setattr(vision_app, "_acquisition_observer", _SingleAlignedObserver())
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 1)
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {
            "cameraReady": True,
            "modelReady": True,
            "fastRenderReady": True,
            "fastPoseReady": True,
        },
    )
    monkeypatch.setattr(
        vision_app,
        "read_camera_with_source",
        lambda *_args, **_kwargs: (
            np.zeros((80, 60, 3), dtype=np.uint8),
            {"source": "recorded_video"},
        ),
    )

    close_calls = []

    async def stubborn_close(attempt_id=None, *, timeout=1.0):
        close_calls.append((attempt_id, timeout))
        if attempt_id is not None:
            raise RuntimeError("acquisition_preview_stubborn_readers")

    async def render_should_not_run(*_args, **_kwargs):
        raise AssertionError("render must wait for successful acquisition cleanup")

    monkeypatch.setattr(vision_app._acquisition_previews, "close", stubborn_close)
    monkeypatch.setattr(vision_app, "render_attempt_frame", render_should_not_run)
    attempt_id = str(uuid4())
    received = []
    done = threading.Event()

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello(manifest))
            assert socket.receive_json()["type"] == "vision.ready"
            socket.send_json(_start(attempt_id, garment_reference))
            assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"

            def receive_until_terminal():
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    message = socket.receive_json()
                    received.append(message)
                    if message["type"] in {
                        "vision.try_on.attempt.completed",
                        "vision.try_on.attempt.failed",
                        "vision.try_on.attempt.canceled",
                    }:
                        break
                done.set()

            reader = threading.Thread(target=receive_until_terminal, daemon=True)
            reader.start()
            assert done.wait(timeout=3.0)
            reader.join(timeout=1.0)

    terminals = [
        message for message in received
        if message["type"] in {
            "vision.try_on.attempt.completed",
            "vision.try_on.attempt.failed",
            "vision.try_on.attempt.canceled",
        }
    ]
    assert terminals == [
        {
            **terminals[0],
            "type": "vision.try_on.attempt.failed",
            "payload": {"attemptId": attempt_id, "reason": "fast_failed"},
        }
    ]
    assert len(close_calls) >= 1
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


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

    async def cancel_event_for(_receipt):
        return asyncio.Event()

    monkeypatch.setattr(vision_app._fast_attempt_registry, "is_current", is_current)
    monkeypatch.setattr(
        vision_app._fast_attempt_registry, "cancel_event_for", cancel_event_for
    )
    monkeypatch.setattr(vision_app.settings, "FRONT_CAMERA_CONFIG", {
        "role": "profile_fast_try_on",
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
        frame, source = asyncio.run(vision_app._read_attempt_front_frame(receipt, timeout=1.0))
    finally:
        vision_app.release_front_camera(
            "try_on_attempt",
            reason="test_cleanup",
            lease_token=f"try-on:{receipt.attempt_id}:{receipt.generation}:{receipt.owner_token}",
        )
        camera_manager.release_all_cameras()

    assert frame.shape == (80, 60, 3)
    assert source == {"source": "dshow", "brokerPid": 4242}
    assert events[0] == ("open", parent_pid, "front", "front-stable")
    assert events[1][0:3] == ("read", parent_pid, 1)
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


@pytest.mark.parametrize("stability_round", [1, 2])
def test_fast_blocked_production_broker_cancel_keeps_loop_live_joins_and_restarts(
    monkeypatch, stability_round
):
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    blocked_read_entered = context.RawValue("b", 0)
    broker = DirectShowCameraBroker(
        "front",
        {
            "role": "profile_fast_try_on",
            "source": "dshow",
            "index": 9,
            "backend": "dshow",
            "stableId": "front-stable",
            "keep_open": True,
            "_brokerReadyHandshake": True,
            "requestCounter": counter,
            "blockedReadEntered": blocked_read_entered,
        },
        context=context,
        target=fast_worker_fixture.block_first_directshow,
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
        "role": "profile_fast_try_on", "source": "dshow", "keep_open": True,
    })
    monkeypatch.setattr(vision_app.settings, "CAMERA_READ_RETRY_COUNT", 0)
    monkeypatch.setattr(camera_manager, "get_camera_maintenance", lambda: Maintenance())
    monkeypatch.setattr(camera_manager, "DirectShowCameraBroker", lambda _role, _config: broker)
    camera_manager.release_all_cameras()

    async def scenario():
        registry.cancel_event = asyncio.Event()
        first = vision_app.AttemptReceipt(str(uuid4()), "owner-1", 1)
        read_task = asyncio.create_task(
            vision_app._read_attempt_front_frame(first, timeout=15.0)
        )
        await _wait_for_raw_value(blocked_read_entered, timeout=15.0)
        loop_yielded = asyncio.Event()

        async def prove_loop_responsive():
            await asyncio.sleep(0)
            loop_yielded.set()

        responsiveness_task = asyncio.create_task(prove_loop_responsive())
        await loop_yielded.wait()
        assert responsiveness_task.done()
        registry.cancel_event.set()
        with pytest.raises(vision_app.GarmentFetchError, match="attempt_canceled"):
            await asyncio.wait_for(read_task, timeout=15.0)
        assert not broker.assert_dead()
        assert broker.active_request_count == 0

        for generation in range(2, 12):
            registry.cancel_event = asyncio.Event()
            receipt = vision_app.AttemptReceipt(
                str(uuid4()), f"owner-{generation}", generation
            )
            frame, source = await vision_app._read_attempt_front_frame(
                receipt, timeout=1.0
            )
            assert frame.shape == (80, 60, 3)
            assert source["brokerPid"] is not None
            assert broker.active_request_count == 0
        assert stability_round in {1, 2}

    try:
        asyncio.run(scenario())
    finally:
        assert asyncio.run(broker.abort_async(reason="test_cleanup"))
        with camera_manager._streams_lock:
            camera_manager._dshow_brokers.pop("front", None)
    assert broker.assert_dead()
    assert vision_app.get_front_camera_owner()["owner"] == "idle"


@pytest.mark.parametrize("prestart", [False, True])
def test_fast_cancel_joins_replacement_ready_before_next_generation_budget(monkeypatch, prestart):
    """A new generation receives a warmed command loop, not a spawn deadline."""
    context = multiprocessing.get_context("spawn")
    starts = context.Value("i", 0)
    blocked_read_entered = context.RawValue("b", 0)
    restart_ready_entered = context.RawValue("b", 0)
    restart_ready_release = context.RawValue("b", 0)
    broker = DirectShowCameraBroker(
        "front",
        {
            "role": "profile_fast_try_on",
            "source": "dshow",
            "index": 9,
            "backend": "dshow",
            "stableId": "front-stable",
            "keep_open": True,
            "_brokerReadyHandshake": True,
            "starts": starts,
            "blockedReadEntered": blocked_read_entered,
            "restartReadyEntered": restart_ready_entered,
            "restartReadyRelease": restart_ready_release,
        },
        context=context,
        target=fast_worker_fixture.block_then_ready_barrier_directshow,
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
            self.cancel_event = asyncio.Event()

        async def is_current(self, _receipt):
            return not self.cancel_event.is_set()

        async def cancel_event_for(self, _receipt):
            return self.cancel_event

    registry = Registry()
    monkeypatch.setattr(vision_app, "_fast_attempt_registry", registry)
    monkeypatch.setattr(vision_app.settings, "FRONT_CAMERA_CONFIG", {
        "role": "profile_fast_try_on", "source": "dshow", "keep_open": True,
    })
    monkeypatch.setattr(vision_app.settings, "CAMERA_READ_RETRY_COUNT", 0)
    monkeypatch.setattr(camera_manager, "get_camera_maintenance", lambda: Maintenance())
    monkeypatch.setattr(camera_manager, "DirectShowCameraBroker", lambda _role, _config: broker)
    if not prestart:
        async def skip_restart(_role):
            return True

        monkeypatch.setattr(vision_app, "restart_camera_request", skip_restart)
    camera_manager.release_all_cameras()

    async def scenario():
        first = vision_app.AttemptReceipt(str(uuid4()), "owner-1", 1)
        blocked = asyncio.create_task(
            vision_app._read_attempt_front_frame(first, timeout=15.0)
        )
        await _wait_for_raw_value(blocked_read_entered, timeout=15.0)
        registry.cancel_event.set()
        if prestart:
            await _wait_for_raw_value(restart_ready_entered, timeout=15.0)
            restart_ready_release.value = 1
        with pytest.raises(vision_app.GarmentFetchError, match="attempt_canceled"):
            await asyncio.wait_for(blocked, timeout=15.0)

        assert starts.value == (2 if prestart else 1)
        registry.cancel_event = asyncio.Event()
        second = vision_app.AttemptReceipt(str(uuid4()), "owner-2", 2)
        if not prestart:
            # Without cleanup prestart, the next generation pays replacement
            # startup from its own frame budget and times out.  Hosted spawn
            # may consume that budget before the child reaches this test's
            # ready barrier, so only assert the caller-visible result.
            with pytest.raises(asyncio.TimeoutError):
                await vision_app._read_attempt_front_frame(second, timeout=1.0)
            return
        # Cleanup only returned after the replacement ready barrier was
        # released, so the next generation's one-second budget covers a read.
        assert broker.active_request_count == 0
        frame, _source = await vision_app._read_attempt_front_frame(second, timeout=1.0)
        assert frame.shape == (80, 60, 3)

    try:
        asyncio.run(scenario())
    finally:
        # Never strand a child in its synchronization barrier when a prior
        # assertion fails; the barrier is test-only and not the behavior
        # being asserted.
        restart_ready_release.value = 1
        assert asyncio.run(broker.abort_async(reason="test_cleanup"))
        with camera_manager._streams_lock:
            camera_manager._dshow_brokers.pop("front", None)
    assert broker.assert_dead()


def test_front_read_cancellation_wins_when_waiter_scheduling_lags(monkeypatch):
    """A fenced attempt cannot return a frame merely because its waiter lags."""

    class DelayedCancellation:
        def is_set(self):
            return True

        async def wait(self):
            await asyncio.Event().wait()

    class Registry:
        async def is_current(self, _receipt):
            return True

        async def cancel_event_for(self, _receipt):
            return DelayedCancellation()

    async def blocked_read(*_args, **_kwargs):
        await asyncio.Event().wait()

    async def aborted(*_args, **_kwargs):
        return True

    async def restart(_role):
        return True

    monkeypatch.setattr(vision_app, "_fast_attempt_registry", Registry())
    monkeypatch.setattr(vision_app, "read_camera_with_source_async", blocked_read)
    monkeypatch.setattr(vision_app, "abort_camera_request", aborted)
    monkeypatch.setattr(vision_app, "restart_camera_request", restart)
    receipt = vision_app.AttemptReceipt(str(uuid4()), "owner", 1)

    with pytest.raises(vision_app.GarmentFetchError, match="attempt_canceled"):
        asyncio.run(
            vision_app._read_attempt_front_frame(
                receipt, timeout=0.01, lease_token="test-cancel-fence"
            )
        )


def test_front_read_current_receipt_fence_wins_after_completed_frame(monkeypatch):
    """A replacement after a read completes still fences that old frame."""

    class Registry:
        def __init__(self):
            self.current_checks = 0

        async def is_current(self, _receipt):
            self.current_checks += 1
            return self.current_checks == 1

        async def cancel_event_for(self, _receipt):
            return asyncio.Event()

    async def completed_read(*_args, **_kwargs):
        return np.zeros((80, 60, 3), dtype=np.uint8), {"source": "dshow"}

    aborted = []

    async def abort(role, *, reason):
        aborted.append((role, reason))
        return True

    async def restart(_role):
        return True

    monkeypatch.setattr(vision_app, "_fast_attempt_registry", Registry())
    monkeypatch.setattr(vision_app, "read_camera_with_source_async", completed_read)
    monkeypatch.setattr(vision_app, "abort_camera_request", abort)
    monkeypatch.setattr(vision_app, "restart_camera_request", restart)
    receipt = vision_app.AttemptReceipt(str(uuid4()), "owner", 1)

    with pytest.raises(vision_app.GarmentFetchError, match="attempt_canceled"):
        asyncio.run(
            vision_app._read_attempt_front_frame(
                receipt, timeout=0.01, lease_token="test-current-fence"
            )
        )

    assert aborted == [("front", "try_on_attempt_canceled")]


def test_render_cancellation_wins_when_waiter_scheduling_lags(monkeypatch):
    """A fenced render result cannot win because the waiter has not run yet."""

    class DelayedCancellation:
        def is_set(self):
            return True

        async def wait(self):
            await asyncio.Event().wait()

    class Registry:
        async def is_current(self, _receipt):
            return True

        async def cancel_event_for(self, _receipt):
            return DelayedCancellation()

    async def completed_render():
        return b"rendered"

    monkeypatch.setattr(vision_app, "_fast_attempt_registry", Registry())
    receipt = vision_app.AttemptReceipt(str(uuid4()), "owner", 1)

    with pytest.raises(vision_app.GarmentFetchError, match="attempt_canceled"):
        asyncio.run(
            vision_app._run_owned_attempt_step(
                receipt, completed_render(), timeout=0.01
            )
        )


def test_v2_fast_result_store_rejects_self_too_large_without_publishing(monkeypatch):
    monkeypatch.setattr(vision_app, "_FAST_RESULT_MAX_BYTES", 8)
    image = _png_bytes()
    with pytest.raises(RuntimeError, match="fast_result_too_large"):
        vision_app._prepare_fast_result(str(uuid4()), image)
