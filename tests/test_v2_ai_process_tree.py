import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as vision_app
import vision.ai_attempt_process as ai_attempt_process_module
from vision.ai_attempt_process import AiAttemptProcess
from vision.ai_model_pack import OfficialAiReadinessSnapshot


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group tracer")
ROOT = Path(__file__).parents[1]
WORKER = Path(__file__).with_name("ai_process_tree_worker.py")


class _GarmentHandler(BaseHTTPRequestHandler):
    image = np.full((36, 48, 4), (180, 40, 90, 255), dtype=np.uint8)
    encoded, payload_buffer = cv2.imencode(".png", image)
    assert encoded
    payload = payload_buffer.tobytes()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args):
        return


class _SingleAlignedObserver:
    ready = True
    fatal_error = None
    pid = None
    active_request_count = 0

    async def start(self):
        return None

    async def observe(self, _frame, *, timeout=15.0):
        from vision.acquisition_observer import AcquisitionObservation

        return AcquisitionObservation(b"jpeg", "single", True)

    async def wait_idle(self, *, timeout=None):
        return True

    async def shutdown(self):
        return None


def _envelope(message_type: str, payload: dict) -> dict:
    return {
        "protocol": "vem.vision.v2",
        "type": message_type,
        "messageId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


def _hello(*additional_capabilities: str) -> dict:
    manifest = json.loads(
        (ROOT / "contracts/vem_vision_v2/manifest.json").read_text("utf-8")
    )
    return _envelope(
        "vision.hello",
        {
            "clientRole": "machine",
            "machineCode": "PROCESS-TREE-TEST",
            "schemaVersion": manifest["schemaVersion"],
            "bundleVersion": manifest["bundleVersion"],
            "contractDigest": manifest["bundleDigest"],
            "capabilities": ["try_on_ai", *additional_capabilities],
        },
    )


def _start(attempt_id: str, reference: str) -> dict:
    garment = _GarmentHandler.payload
    return _envelope(
        "vision.try_on.attempt.start",
        {
            "attemptId": attempt_id,
            "mode": "ai",
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


def _configure_public_ai(
    monkeypatch,
    pack: Path,
    pid_file: Path | None,
    mode: str | None,
    *,
    command_by_attempt: dict[str, tuple[Path, str]] | None = None,
    success_png: Path | None = None,
) -> None:
    monkeypatch.setenv("VEM_AI_MODEL_PACK", str(pack))
    monkeypatch.setattr(
        vision_app,
        "official_ai_readiness_snapshot",
        lambda root: OfficialAiReadinessSnapshot(
            root=str(pack),
            identity=("test-owned-process-tree",),
            ready=root == str(pack),
            diagnostic="ready" if root == str(pack) else "model_pack_missing",
        ),
    )
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", AiAttemptProcess)
    monkeypatch.setattr(vision_app, "_ai_attempt_execution_lock", asyncio.Lock())
    monkeypatch.setattr(
        vision_app,
        "_fast_attempt_registry",
        vision_app.FastAttemptRegistry(
            terminal_ttl_seconds=vision_app._FAST_RESULT_TTL_SECONDS,
            result_max_count=vision_app._FAST_RESULT_MAX_COUNT,
            result_max_bytes=vision_app._FAST_RESULT_MAX_TOTAL_BYTES,
            result_single_max_bytes=vision_app._FAST_RESULT_MAX_BYTES,
        ),
    )
    monkeypatch.setattr(
        vision_app, "get_runtime_status", lambda: {"cameraReady": True, "modelReady": True}
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_acquisition_observer", _SingleAlignedObserver())
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 1)
    monkeypatch.setattr(
        vision_app,
        "read_camera_with_source",
        lambda *_args, **_kwargs: (
            np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8),
            {"source": "recorded_video"},
        ),
    )

    def test_worker_command(_model_pack, **kwargs):
        selected_pid_file = pid_file
        selected_mode = mode
        if command_by_attempt is not None:
            output_png = str(kwargs["output_png"])
            matches = [
                command
                for attempt_id, command in command_by_attempt.items()
                if attempt_id in output_png
            ]
            assert len(matches) == 1
            selected_pid_file, selected_mode = matches[0]
        assert selected_pid_file is not None
        assert selected_mode is not None
        command = [
            sys.executable,
            str(WORKER),
            "--role",
            "leader",
            "--mode",
            selected_mode,
            "--pid-file",
            str(selected_pid_file),
        ]
        if selected_mode == "success":
            assert success_png is not None
            command.extend(
                ["--success-png", str(success_png), "--output", str(kwargs["output_png"])]
            )
        return command

    monkeypatch.setattr(
        ai_attempt_process_module, "ai_attempt_worker_command", test_worker_command
    )


def _serve_garment():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return (
        server,
        thread,
        f"http://127.0.0.1:{server.server_port}/garment?token=tree-test",
    )


def _receive_until_generating(socket) -> list[dict]:
    trace = []
    while True:
        message = socket.receive_json()
        trace.append(message)
        if (
            message["type"] == "vision.try_on.attempt.generating"
            and message["payload"]["stage"] == "generating"
        ):
            return trace


def _receive_until_terminal(socket) -> tuple[list[dict], dict]:
    trace = []
    while True:
        message = socket.receive_json()
        trace.append(message)
        if message["type"] in {
            "vision.try_on.attempt.completed",
            "vision.try_on.attempt.failed",
            "vision.try_on.attempt.canceled",
        }:
            return trace, message


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        state = (Path("/proc") / str(pid) / "stat").read_text("utf-8").split()[2]
        return state != "Z"
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError):
        return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (PermissionError, ProcessLookupError):
        return False


def _wait_pid_tree(pid_file: Path) -> dict[str, int]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            tree = json.loads(pid_file.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if set(tree) == {"leader", "child", "grandchild"}:
            return {name: int(pid) for name, pid in tree.items()}
    raise AssertionError("process tree PID evidence was not published")


def _assert_tree_dead(tree: dict[str, int]) -> None:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and any(_pid_exists(pid) for pid in tree.values()):
        time.sleep(0.025)
    assert {name: pid for name, pid in tree.items() if _pid_exists(pid)} == {}


def _staging_paths(attempt_id: str) -> list[Path]:
    return list(Path(tempfile.gettempdir()).glob(f"vem-ai-attempt-{attempt_id}-*"))


def _assert_staging_clear(attempt_id: str) -> None:
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and _staging_paths(attempt_id):
        time.sleep(0.025)
    assert _staging_paths(attempt_id) == []


def _success_png(path: Path) -> None:
    image = np.full((42, 54, 4), (25, 190, 75, 255), dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    path.write_bytes(payload.tobytes())


def test_public_ai_replacement_joins_real_tree_before_next_worker_and_completes(
    tmp_path, monkeypatch
):
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    first_id, second_id = str(uuid4()), str(uuid4())
    first_pid_file = tmp_path / "first-tree.json"
    second_pid_file = tmp_path / "second-tree.json"
    success_png = tmp_path / "success.png"
    _success_png(success_png)
    _configure_public_ai(
        monkeypatch,
        pack,
        None,
        None,
        command_by_attempt={
            first_id: (first_pid_file, "block"),
            second_id: (second_pid_file, "success"),
        },
        success_png=success_png,
    )
    server, thread, reference = _serve_garment()
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(first_id, reference))
                first_trace = _receive_until_generating(socket)
                first_tree = _wait_pid_tree(first_pid_file)
                assert all(_pid_alive(pid) for pid in first_tree.values())

                socket.send_json(_start(second_id, reference))
                replacement_trace = []
                while True:
                    message = socket.receive_json()
                    replacement_trace.append(message)
                    if (
                        message["type"] == "vision.try_on.attempt.accepted"
                        and message["payload"]["attemptId"] == second_id
                    ):
                        break

                assert {
                    name: pid
                    for name, pid in first_tree.items()
                    if _pid_exists(pid)
                } == {}
                second_trace = _receive_until_generating(socket)
                second_tree = _wait_pid_tree(second_pid_file)
                completion_trace, completed = _receive_until_terminal(socket)
                _assert_tree_dead(second_tree)
                socket.send_json(_envelope("vision.ping", {}))
                late_messages = []
                while True:
                    message = socket.receive_json()
                    if message["type"] == "vision.pong":
                        break
                    late_messages.append(message)

            first_missing = client.get(
                f"/v2/try-on/results/{first_id}?token=no-result"
            )
            result = completed["payload"]["result"]
            parsed = urlsplit(result["reference"])
            grant = f"{parsed.path}?{parsed.query}"
            get = client.get(grant)
            head = client.head(grant)

        trace = first_trace + replacement_trace + second_trace + completion_trace
        first_terminals = [
            message
            for message in trace + late_messages
            if message["type"]
            in {
                "vision.try_on.attempt.completed",
                "vision.try_on.attempt.failed",
                "vision.try_on.attempt.canceled",
            }
            and message["payload"]["attemptId"] == first_id
        ]
        assert len(first_terminals) == 1
        assert first_terminals[0]["type"] == "vision.try_on.attempt.canceled"
        assert first_terminals[0]["payload"] == {
            "attemptId": first_id,
            "reason": "replaced",
        }
        assert completed["type"] == "vision.try_on.attempt.completed"
        assert completed["payload"]["attemptId"] == second_id
        assert first_missing.status_code == 404
        assert get.status_code == 200
        assert get.headers["content-type"] == "image/png"
        assert head.status_code == 200
        assert int(head.headers["content-length"]) == len(get.content)
        _assert_staging_clear(first_id)
        _assert_staging_clear(second_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_ai_timeout_kills_real_tree_without_result_and_keeps_public_ws_alive(
    tmp_path, monkeypatch
):
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "timeout-tree.json"
    _configure_public_ai(monkeypatch, pack, pid_file, "block")
    monkeypatch.setattr(vision_app, "_AI_ATTEMPT_TIMEOUT_SECONDS", 0.6)
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", True)
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_INTERVAL_MS", 10)
    monkeypatch.setattr(vision_app.settings, "MOCK_SCENARIO", "success")
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello("presence_status"))
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference))
                trace = _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                assert all(_pid_alive(pid) for pid in tree.values())
                tail, terminal = _receive_until_terminal(socket)
                trace.extend(tail)
                _assert_tree_dead(tree)
                _assert_staging_clear(attempt_id)

                socket.send_json(_envelope("vision.ping", {}))
                pong = None
                presence_seen = any(
                    message["type"] == "vision.presence_status" for message in trace
                )
                while pong is None or not presence_seen:
                    message = socket.receive_json()
                    trace.append(message)
                    if message["type"] == "vision.pong":
                        pong = message
                    if message["type"] == "vision.presence_status":
                        presence_seen = True

            missing = client.get(
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            )

        terminal_types = {
            "vision.try_on.attempt.completed",
            "vision.try_on.attempt.failed",
            "vision.try_on.attempt.canceled",
        }
        attempt_terminals = [
            message
            for message in trace
            if message["type"] in terminal_types
            and message["payload"]["attemptId"] == attempt_id
        ]
        assert len(attempt_terminals) == 1
        assert terminal["type"] == "vision.try_on.attempt.canceled"
        assert terminal["payload"] == {
            "attemptId": attempt_id,
            "reason": "timeout",
        }
        assert all(
            message["type"] != "vision.try_on.attempt.completed" for message in trace
        )
        assert missing.status_code == 404
        assert pong is not None and pong["type"] == "vision.pong"
        assert presence_seen is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_ai_leader_crash_kills_real_descendants_and_emits_one_failure(
    tmp_path, monkeypatch
):
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "tree.json"
    _configure_public_ai(monkeypatch, pack, pid_file, "crash")
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    assert _staging_paths(attempt_id) == []
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference))
                trace = _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                tail, terminal = _receive_until_terminal(socket)
                trace.extend(tail)
                socket.send_json(_envelope("vision.ping", {}))
                pong = socket.receive_json()

            assert client.get(
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            ).status_code == 404

        assert terminal["type"] == "vision.try_on.attempt.failed"
        assert terminal["payload"] == {
            "attemptId": attempt_id,
            "reason": "ai_failed",
        }
        assert sum(message["type"] == terminal["type"] for message in trace) == 1
        assert all(message["type"] != "vision.try_on.attempt.completed" for message in trace)
        assert pong["type"] == "vision.pong"
        _assert_tree_dead(tree)
        _assert_staging_clear(attempt_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_ai_cancel_kills_real_tree_without_late_terminal_or_result(
    tmp_path, monkeypatch
):
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "tree.json"
    _configure_public_ai(monkeypatch, pack, pid_file, "block")
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    assert _staging_paths(attempt_id) == []
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference))
                trace = _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                assert all(_pid_alive(pid) for pid in tree.values())

                started = time.monotonic()
                socket.send_json(
                    _envelope(
                        "vision.try_on.attempt.cancel",
                        {"attemptId": attempt_id, "reason": "route_leave"},
                    )
                )
                tail, terminal = _receive_until_terminal(socket)
                elapsed = time.monotonic() - started
                trace.extend(tail)
                _assert_tree_dead(tree)
                _assert_staging_clear(attempt_id)

                time.sleep(0.1)
                socket.send_json(_envelope("vision.ping", {}))
                late_messages = []
                while True:
                    message = socket.receive_json()
                    if message["type"] == "vision.pong":
                        pong = message
                        break
                    late_messages.append(message)

            assert client.get(
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            ).status_code == 404

        assert elapsed < 2
        assert terminal["type"] == "vision.try_on.attempt.canceled"
        assert terminal["payload"] == {
            "attemptId": attempt_id,
            "reason": "route_leave",
        }
        terminal_types = {
            "vision.try_on.attempt.completed",
            "vision.try_on.attempt.failed",
            "vision.try_on.attempt.canceled",
        }
        assert sum(message["type"] in terminal_types for message in trace) == 1
        assert all(message["type"] not in terminal_types for message in late_messages)
        assert pong["type"] == "vision.pong"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
