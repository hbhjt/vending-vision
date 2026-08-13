import hashlib
import inspect
import json
import asyncio
import queue
import threading
import tempfile
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
from PIL import Image

import app as vision_app
from vision.ai_model_pack import OfficialAiReadinessSnapshot


def _receive_json_with_timeout(socket, *, timeout=10.0):
    """Bound a test-client receive without changing production deadlines."""
    result = queue.Queue(maxsize=1)

    def receive():
        try:
            result.put((True, socket.receive_json()))
        except BaseException as exc:
            result.put((False, exc))

    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    try:
        succeeded, value = result.get(timeout=timeout)
    except queue.Empty:
        socket.close()
        reader.join(timeout=1)
        pytest.fail(f"websocket receive exceeded {timeout:.1f}s test deadline")
    if succeeded:
        return value
    raise value


def _png_bytes(color=(20, 120, 220, 255), *, size=(48, 36)):
    image = np.full((size[1], size[0], 4), color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class _GarmentHandler(BaseHTTPRequestHandler):
    payload = _png_bytes((180, 40, 90, 255))

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


def _manifest():
    return json.loads(
        (Path(__file__).parents[1] / "contracts/vem_vision_v2/manifest.json").read_text(
            "utf-8"
        )
    )


def _hello():
    manifest = _manifest()
    return _envelope(
        "vision.hello",
        {
            "clientRole": "machine",
            "machineCode": "M001",
            "schemaVersion": manifest["schemaVersion"],
            "bundleVersion": manifest["bundleVersion"],
            "contractDigest": manifest["bundleDigest"],
            "capabilities": ["try_on_fast", "try_on_ai"],
        },
    )


def _start(attempt_id, reference, *, mode):
    garment = _GarmentHandler.payload
    return _envelope(
        "vision.try_on.attempt.start",
        {
            "attemptId": attempt_id,
            "mode": mode,
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


class _CollectingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


def test_v2_ai_backpressure_rejects_with_ai_unavailable_not_fast_unavailable():
    attempt_id = str(uuid4())
    socket = _CollectingWebSocket()
    asyncio.run(
        vision_app.reject_v2_fast_attempt_for_backpressure(
            socket,
            asyncio.Lock(),
            _start(attempt_id, "http://127.0.0.1/garment.png?token=t", mode="ai"),
        )
    )

    assert socket.messages[-1]["type"] == "vision.try_on.attempt.failed"
    assert socket.messages[-1]["payload"] == {
        "attemptId": attempt_id,
        "reason": "ai_unavailable",
    }


def test_ai_parent_validation_and_result_failure_paths_never_emit_fast_failed():
    ai_source = inspect.getsource(vision_app.run_v2_ai_attempt)
    backpressure_source = inspect.getsource(vision_app.reject_v2_fast_attempt_for_backpressure)

    assert '"fast_failed"' not in ai_source
    assert '"ai_failed"' in ai_source
    assert '"ai_unavailable" if mode == "ai"' in backpressure_source


def _write_official_pack(root: Path):
    model = root / "CatVTON" / "attention.safetensors"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"official-weight")
    manifest = {
        "schemaVersion": "vem-catvton-model-pack/v1",
        "upstream": {"repository": "zhengchong/CatVTON", "revision": "9f415fa"},
        "files": [
            {
                "path": "CatVTON/attention.safetensors",
                "byteSize": model.stat().st_size,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / "ai-model-manifest.json").write_text(json.dumps(manifest), "utf-8")


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


class _CountingFastBroker:
    ready = True
    pose_ready = True
    pid = None

    def __init__(self):
        self.calls = 0

    async def start(self):
        return None

    def quiesce(self):
        return None

    async def shutdown(self):
        return None


class _PrepareBarrierRegistry(vision_app.FastAttemptRegistry):
    def __init__(self):
        super().__init__(
            terminal_ttl_seconds=vision_app._FAST_RESULT_TTL_SECONDS,
            result_max_count=vision_app._FAST_RESULT_MAX_COUNT,
            result_max_bytes=vision_app._FAST_RESULT_MAX_TOTAL_BYTES,
            result_single_max_bytes=vision_app._FAST_RESULT_MAX_BYTES,
        )
        self.prepared = threading.Event()
        self.release = threading.Event()

    async def prepare_admission(self, **kwargs):
        preparation = await super().prepare_admission(**kwargs)
        self.prepared.set()
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        return preparation


class _DeterministicAiChild:
    calls = 0

    def __init__(self, model_pack):
        self.model_pack = Path(model_pack)
        self.closed = False

    async def run(
        self,
        *,
        person_png: Path,
        garment_png: Path,
        output_png: Path,
        timeout: float,
        template: str = "tshirt_short_sleeve",
        regional_evidence_output: Path | None = None,
        captured_source: dict | None = None,
    ):
        type(self).calls += 1
        assert self.model_pack.exists()
        assert person_png.exists()
        assert garment_png.read_bytes() == _GarmentHandler.payload
        assert template in {"tshirt_short_sleeve", "tshirt_long_sleeve"}
        output_png.write_bytes(_png_bytes((5, 80, 140, 255), size=(64, 64)))
        if regional_evidence_output is not None:
            _write_test_regional_sidecar(
                regional_evidence_output,
                person_png,
                garment_png,
                output_png,
                captured_source,
            )

    async def close(self):
        self.closed = True


class _BlockingAiChild:
    instances = []
    block_first_cleanup = False

    def __init__(self, model_pack):
        self.model_pack = Path(model_pack)
        self.run_entered = threading.Event()
        self.cleanup_entered = threading.Event()
        self.loop = None
        self._run_release = None
        self._cleanup_release = None
        self._block_cleanup = bool(type(self).block_first_cleanup and not type(self).instances)
        self.closed = False
        self.tree_dead = False
        self.staging_dir = None
        self.output_path = None
        type(self).instances.append(self)

    async def run(self, *, person_png: Path, garment_png: Path, output_png: Path, timeout: float, template: str = "tshirt_short_sleeve", regional_evidence_output: Path | None = None, captured_source: dict | None = None):
        self.loop = asyncio.get_running_loop()
        self._run_release = asyncio.Event()
        self.staging_dir = output_png.parent
        self.output_path = output_png
        self.run_entered.set()
        try:
            await self._run_release.wait()
            output_png.write_bytes(_png_bytes((9, 90, 160, 255), size=(64, 64)))
            if regional_evidence_output is not None:
                _write_test_regional_sidecar(regional_evidence_output, person_png, garment_png, output_png, captured_source)
        except asyncio.CancelledError:
            try:
                output_png.write_bytes(_png_bytes((200, 20, 40, 255), size=(64, 64)))
            except OSError:
                pass
            raise

    async def close(self):
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        if self._run_release is not None:
            self._run_release.set()
        self._cleanup_release = asyncio.Event()
        if not self._block_cleanup:
            self._cleanup_release.set()
        self.cleanup_entered.set()
        await self._cleanup_release.wait()
        self.closed = True
        self.tree_dead = True

    def release_run(self):
        if self.loop is None or self._run_release is None:
            raise RuntimeError("AI child has not entered run")
        self.loop.call_soon_threadsafe(self._run_release.set)

    def release_cleanup(self):
        if self.loop is None or self._cleanup_release is None:
            raise RuntimeError("AI child has not entered cleanup")
        self.loop.call_soon_threadsafe(self._cleanup_release.set)


class _OutputValidationAiChild:
    instances = []
    mode = "undecodable"

    def __init__(self, model_pack):
        self.model_pack = Path(model_pack)
        self.closed = False
        self.tree_dead = False
        self.staging_dir = None
        self.escape_path = None
        type(self).instances.append(self)

    async def run(self, *, person_png: Path, garment_png: Path, output_png: Path, timeout: float, template: str = "tshirt_short_sleeve", regional_evidence_output: Path | None = None, captured_source: dict | None = None):
        self.staging_dir = output_png.parent
        mode = type(self).mode
        if mode == "undecodable":
            output_png.write_bytes(b"\x89PNG\r\n\x1a\nnot-decodable")
        elif mode == "oversize-bytes":
            output_png.write_bytes(_png_bytes((1, 2, 3, 255), size=(32, 32)))
        elif mode == "oversize-dimensions":
            output_png.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + (13).to_bytes(4, "big")
                + b"IHDR"
                + (8193).to_bytes(4, "big")
                + (1).to_bytes(4, "big")
                + b"\x08\x06\x00\x00\x00"
                + b"\x00\x00\x00\x00"
            )
        elif mode == "wrong-format":
            output_png.write_bytes(b"GIF89a-not-a-png")
        elif mode == "input-copy":
            output_png.write_bytes(person_png.read_bytes())
        elif mode == "garment-copy":
            output_png.write_bytes(garment_png.read_bytes())
        elif mode == "missing-file":
            return
        elif mode == "extra-file":
            output_png.write_bytes(_png_bytes((5, 80, 140, 255), size=(64, 64)))
            (output_png.parent / "debug-extra.png").write_bytes(b"extra")
        elif mode == "path-escape":
            escaped = output_png.parent.parent / f"{output_png.parent.name}-escaped.png"
            escaped.write_bytes(_png_bytes((5, 80, 140, 255), size=(64, 64)))
            output_png.symlink_to(escaped)
            self.escape_path = escaped
        elif mode == "worker-error":
            raise RuntimeError("deterministic worker failure")
        else:
            raise AssertionError(mode)
        if regional_evidence_output is not None and output_png.is_file() and not output_png.is_symlink() and mode != "extra-file":
            _write_test_regional_sidecar(regional_evidence_output, person_png, garment_png, output_png, captured_source)

    async def close(self):
        self.closed = True
        self.tree_dead = True


def _configure_stage1_runtime(monkeypatch, pack: Path, *, ai_ready: bool = True):
    monkeypatch.setenv("VEM_AI_MODEL_PACK", str(pack))
    monkeypatch.setattr(
        vision_app,
        "official_ai_readiness_snapshot",
        lambda value: OfficialAiReadinessSnapshot(
            root=str(pack),
            identity=None,
            ready=bool(ai_ready and value == str(pack)),
            diagnostic="ready"
            if ai_ready and value == str(pack)
            else "model_pack_missing",
        ),
    )
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
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )
    monkeypatch.setattr(vision_app.settings, "PROFILE_PUSH_ENABLED", False)
    monkeypatch.setattr(vision_app, "_acquisition_observer", _SingleAlignedObserver())
    monkeypatch.setattr(vision_app, "_ACQUISITION_STABLE_FRAMES", 1)
    monkeypatch.setattr(
        vision_app,
        "read_camera_with_source",
        lambda *_args, **_kwargs: (
            np.full((80, 60, 3), (235, 220, 205), dtype=np.uint8),
            _recorded_front_source(),
        ),
    )


def _recorded_front_source():
    return {
        "adapter": "recorded_video",
        "configSha256": "7" * 64,
        "decodedFrameCount": 42,
        "fixtureSha256": "8" * 64,
        "frameIndex": 7,
        "relabeled": False,
        "role": "front",
        "synthetic": False,
    }


def _write_test_regional_sidecar(path, person, garment, output, captured_source):
    with Image.open(output) as image:
        width, height = image.size
    value = {
        "attempt": {
            "acquisitionSource": "direct_recorded_frame",
            "decodedHeight": height,
            "decodedWidth": width,
            "garmentSha256": hashlib.sha256(garment.read_bytes()).hexdigest(),
            "inputSha256": hashlib.sha256(person.read_bytes()).hexdigest(),
            "recordedFixtureSha256": captured_source["fixtureSha256"],
            "resultSha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sourceCamera": "front",
        },
        "evaluator": {},
        "kind": "regional-evidence",
        "masks": {},
        "measurements": {},
        "policy": {},
        "schemaVersion": "vem-ai-regional-evidence/v1",
        "verdict": "regional_check_failed",
    }
    path.write_bytes(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )


def _serve_garment():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reference = f"http://127.0.0.1:{server.server_port}/garment?token=source-token"
    return server, thread, reference


def _receive_until_completed(socket):
    trace = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        message = _receive_json_with_timeout(
            socket, timeout=max(0.001, deadline - time.monotonic())
        )
        trace.append(message)
        if message["type"] == "vision.try_on.attempt.completed":
            return trace, message
    raise AssertionError(f"attempt did not complete: {[message['type'] for message in trace]}")


def _receive_until_ai_child_running(socket):
    trace = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        message = _receive_json_with_timeout(
            socket, timeout=max(0.001, deadline - time.monotonic())
        )
        trace.append(message)
        if (
            message["type"] == "vision.try_on.attempt.generating"
            and message["payload"]["stage"] == "generating"
        ):
            return trace
    raise AssertionError(f"AI child did not start: {[message['type'] for message in trace]}")


def _receive_until_terminal(socket):
    trace = []
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        message = _receive_json_with_timeout(
            socket, timeout=max(0.001, deadline - time.monotonic())
        )
        trace.append(message)
        if message["type"] in {
            "vision.try_on.attempt.completed",
            "vision.try_on.attempt.failed",
            "vision.try_on.attempt.canceled",
        }:
            return trace, message
    raise AssertionError(f"terminal not received: {[message['type'] for message in trace]}")


def _assert_canceled_cleanup_and_no_result(client, attempt_id: str, child: _BlockingAiChild, reason: str, terminal):
    assert terminal["type"] == "vision.try_on.attempt.canceled"
    assert terminal["payload"] == {"attemptId": attempt_id, "reason": reason}
    assert child.closed is True
    assert child.tree_dead is True
    assert child.staging_dir is not None
    assert not child.staging_dir.exists()
    assert client.get(f"/v2/try-on/results/{attempt_id}?token=late-token").status_code == 404


def _start_blocking_ai(monkeypatch):
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _BlockingAiChild)
    _BlockingAiChild.instances.clear()
    _BlockingAiChild.block_first_cleanup = False


def _start_output_validation_ai(monkeypatch, mode: str):
    _OutputValidationAiChild.instances.clear()
    _OutputValidationAiChild.mode = mode
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _OutputValidationAiChild)


def _wait_for_blocking_child(index: int = 0) -> _BlockingAiChild:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if len(_BlockingAiChild.instances) > index:
            child = _BlockingAiChild.instances[index]
            if child.run_entered.wait(timeout=0.01):
                return child
        time.sleep(0.01)
    raise AssertionError(f"AI child {index} did not enter run")


def _assert_failed_cleanup_and_no_result(client, attempt_id: str, child, terminal):
    assert terminal["type"] == "vision.try_on.attempt.failed"
    assert terminal["payload"] == {"attemptId": attempt_id, "reason": "ai_failed"}
    assert "result" not in terminal["payload"]
    if child is not None:
        assert child.closed is True
        assert child.tree_dead is True
        assert child.staging_dir is not None
        assert not child.staging_dir.exists()
    assert client.get(f"/v2/try-on/results/{attempt_id}?token=invalid-token").status_code == 404


def test_v2_ai_attempt_uses_official_boundary_not_fast_and_publishes_tokenized_result(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    fast = _CountingFastBroker()
    monkeypatch.setattr(vision_app, "_fast_render_broker", fast)
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _DeterministicAiChild)
    acceptance_root = tmp_path / "acceptance-regional"
    acceptance_root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(acceptance_root))
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                ready = socket.receive_json()
                assert ready["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                trace, completed = _receive_until_completed(socket)

            result = completed["payload"]["result"]
            parsed = urlsplit(result["reference"])
            grant_path = f"{parsed.path}?{parsed.query}"
            get = client.get(grant_path)
            head = client.head(grant_path)

        assert [message["type"] for message in trace][0] == "vision.try_on.attempt.accepted"
        assert "vision.try_on.attempt.acquiring" in [message["type"] for message in trace]
        assert "vision.try_on.attempt.generating" in [message["type"] for message in trace]
        assert get.status_code == 200
        assert get.headers["content-type"] == "image/png"
        assert head.status_code == 200
        assert int(head.headers["content-length"]) == len(get.content)
        assert fast.calls == 0
        assert _DeterministicAiChild.calls == 1
        exported = acceptance_root / f"{attempt_id}.regional-evidence.json"
        assert exported.is_file()
        assert json.loads(exported.read_text("utf-8"))["attempt"]["resultSha256"] == result[
            "digest"
        ].removeprefix("sha256:")
        assert set(completed["payload"]) == {"attemptId", "result"}
    finally:
        _DeterministicAiChild.calls = 0
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_admission_rechecks_current_root_after_prepare_barrier_becomes_unset(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    registry = _PrepareBarrierRegistry()
    monkeypatch.setattr(vision_app, "_fast_attempt_registry", registry)
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _DeterministicAiChild)
    server, thread, reference = _serve_garment()
    toggler_errors = []

    def unset_root_at_barrier():
        if not registry.prepared.wait(timeout=5):
            toggler_errors.append("prepare barrier was not reached")
            registry.release.set()
            return
        monkeypatch.delenv("VEM_AI_MODEL_PACK", raising=False)
        registry.release.set()

    toggler = threading.Thread(target=unset_root_at_barrier)
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert _receive_json_with_timeout(socket)["payload"]["aiReady"] is True
                toggler.start()
                start = _start(attempt_id, reference, mode="ai")
                socket.send_json(start)
                terminal = _receive_json_with_timeout(socket)
                socket.send_json(start)
                replay = _receive_json_with_timeout(socket)

        assert terminal == {
            "protocol": "vem.vision.v2",
            "type": "vision.try_on.attempt.failed",
            "messageId": terminal["messageId"],
            "timestamp": terminal["timestamp"],
            "payload": {"attemptId": attempt_id, "reason": "ai_unavailable"},
        }
        assert replay == terminal
        assert _DeterministicAiChild.calls == 0
        assert toggler_errors == []
    finally:
        registry.release.set()
        if toggler.ident is not None:
            toggler.join(timeout=2)
        _DeterministicAiChild.calls = 0
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_admission_rechecks_current_root_after_prepare_barrier_becomes_valid(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    monkeypatch.delenv("VEM_AI_MODEL_PACK", raising=False)
    registry = _PrepareBarrierRegistry()
    monkeypatch.setattr(vision_app, "_fast_attempt_registry", registry)
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _DeterministicAiChild)
    server, thread, reference = _serve_garment()
    toggler_errors = []

    def restore_root_at_barrier():
        if not registry.prepared.wait(timeout=5):
            toggler_errors.append("prepare barrier was not reached")
            registry.release.set()
            return
        monkeypatch.setenv("VEM_AI_MODEL_PACK", str(pack))
        registry.release.set()

    toggler = threading.Thread(target=restore_root_at_barrier)
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is False
                toggler.start()
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                trace, terminal = _receive_until_terminal(socket)

        assert trace[0]["type"] == "vision.try_on.attempt.accepted"
        assert terminal["type"] == "vision.try_on.attempt.completed"
        assert _DeterministicAiChild.calls == 1
        assert toggler_errors == []
    finally:
        registry.release.set()
        if toggler.ident is not None:
            toggler.join(timeout=2)
        _DeterministicAiChild.calls = 0
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_fast_attempt_never_calls_ai_boundary(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack, ai_ready=False)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _CountingFastBroker())

    ai_calls = []

    def forbidden_ai_factory(*_args):
        ai_calls.append("called")
        raise AssertionError("AI child must not start for Fast")

    async def render(frame, garment_png, *, digest, template, timeout, broker):
        assert garment_png == _GarmentHandler.payload
        return _png_bytes((40, 120, 40, 255), size=(64, 64))

    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", forbidden_ai_factory)
    monkeypatch.setattr(vision_app, "render_attempt_frame", render)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["fastReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="fast"))
                _trace, completed = _receive_until_completed(socket)

        assert completed["payload"]["attemptId"] == attempt_id
        assert ai_calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_unready_fails_closed_before_acquisition_child_or_staging_while_fast_still_runs(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    (pack / "CatVTON" / "attention.safetensors").write_bytes(b"tampered")
    _configure_stage1_runtime(monkeypatch, pack, ai_ready=False)
    monkeypatch.setattr(vision_app, "_fast_render_broker", _CountingFastBroker())
    observe_calls = []

    class ForbiddenObserver(_SingleAlignedObserver):
        async def observe(self, _frame, *, timeout=15.0):
            observe_calls.append("called")
            raise AssertionError("AI-unready attempt must not acquire")

    ai_calls = []

    def forbidden_ai_factory(*_args):
        ai_calls.append("called")
        raise AssertionError("AI-unready attempt must not start child")

    async def render(frame, garment_png, *, digest, template, timeout, broker):
        return _png_bytes((40, 120, 40, 255), size=(64, 64))

    monkeypatch.setattr(vision_app, "_acquisition_observer", ForbiddenObserver())
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", forbidden_ai_factory)
    monkeypatch.setattr(vision_app, "render_attempt_frame", render)
    before_staging = set(Path(tempfile.gettempdir()).glob("vem-ai-attempt-*"))
    server, thread, reference = _serve_garment()
    try:
        ai_attempt_id, fast_attempt_id = str(uuid4()), str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                ready = socket.receive_json()
                assert ready["payload"]["aiReady"] is False
                assert ready["payload"]["fastReady"] is True

                socket.send_json(_start(ai_attempt_id, reference, mode="ai"))
                ai_terminal = socket.receive_json()
                assert ai_terminal["type"] == "vision.try_on.attempt.failed"
                assert ai_terminal["payload"] == {
                    "attemptId": ai_attempt_id,
                    "reason": "ai_unavailable",
                }
                assert observe_calls == []
                monkeypatch.setattr(vision_app, "_acquisition_observer", _SingleAlignedObserver())

                socket.send_json(_start(fast_attempt_id, reference, mode="fast"))
                _trace, fast_completed = _receive_until_completed(socket)

            assert client.get(
                f"/v2/try-on/results/{ai_attempt_id}?token=none"
            ).status_code == 404
            assert fast_completed["type"] == "vision.try_on.attempt.completed"

        after_staging = set(Path(tempfile.gettempdir()).glob("vem-ai-attempt-*"))
        assert after_staging == before_staging
        assert observe_calls == []
        assert ai_calls == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "corrupt",
        "tamper",
        "extra",
        "wrong-revision",
        "duplicate-key",
        "test-fake-env",
    ],
)
def test_v2_hello_ai_readiness_comes_only_from_verified_official_pack_without_affecting_core(
    tmp_path, monkeypatch, case
):
    pack = tmp_path / "pack"
    if case not in {"missing", "test-fake-env"}:
        _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack, ai_ready=False)

    if case == "corrupt":
        (pack / "ai-model-manifest.json").write_text("{not json", "utf-8")
    elif case == "tamper":
        (pack / "CatVTON" / "attention.safetensors").write_bytes(b"tampered")
    elif case == "extra":
        (pack / "extra-weight.safetensors").write_bytes(b"extra")
    elif case == "wrong-revision":
        manifest = json.loads((pack / "ai-model-manifest.json").read_text("utf-8"))
        manifest["upstream"]["revision"] = "wrong"
        (pack / "ai-model-manifest.json").write_text(json.dumps(manifest), "utf-8")
    elif case == "duplicate-key":
        (pack / "ai-model-manifest.json").write_text(
            '{"schemaVersion":"vem-catvton-model-pack/v1","schemaVersion":"duplicate","upstream":{"repository":"zhengchong/CatVTON","revision":"9f415fa"},"files":[]}',
            "utf-8",
        )
    elif case == "test-fake-env":
        monkeypatch.setenv("VEM_AI_FAKE_READY", "1")
        monkeypatch.setenv("VEM_AI_FAKE_CHILD", "1")
        monkeypatch.setenv("VEM_AI_WORKER_TARGET", "tests.fake_worker")

    with TestClient(vision_app.app) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json(_hello())
            ready = socket.receive_json()

    assert ready["type"] == "vision.ready"
    assert ready["payload"]["aiReady"] is False
    assert ready["payload"]["fastReady"] is True
    assert ready["payload"]["cameraReady"] is True
    assert ready["payload"]["visionBusinessReady"] is True
    assert ready["payload"]["businessReadinessDiagnostic"] == "ready"
    assert "try_on_fast" in ready["payload"]["capabilities"]
    assert "try_on_ai" not in ready["payload"]["capabilities"]


def test_v2_ai_cancel_route_leave_kills_child_removes_staging_and_fences_late_result(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    acceptance_root = tmp_path / "acceptance-regional"
    acceptance_root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(acceptance_root))
    _start_blocking_ai(monkeypatch)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _receive_until_ai_child_running(socket)
                child = _wait_for_blocking_child(0)

                socket.send_json(
                    _envelope(
                        "vision.try_on.attempt.cancel",
                        {"attemptId": attempt_id, "reason": "route_leave"},
                    )
                )
                _trace, terminal = _receive_until_terminal(socket)

            _assert_canceled_cleanup_and_no_result(
                client, attempt_id, child, "route_leave", terminal
            )
            assert list(acceptance_root.iterdir()) == []

            with client.websocket_connect("/ws") as replay:
                replay.send_json(_hello())
                assert replay.receive_json()["type"] == "vision.ready"
                replay.send_json(_start(attempt_id, reference, mode="ai"))
                assert replay.receive_json() == terminal
    finally:
        _BlockingAiChild.instances.clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_replacement_waits_for_child_cleanup_before_next_attempt(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    _start_blocking_ai(monkeypatch)
    _BlockingAiChild.block_first_cleanup = True
    server, thread, reference = _serve_garment()
    try:
        first_id, second_id = str(uuid4()), str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(first_id, reference, mode="ai"))
                _receive_until_ai_child_running(socket)
                first_child = _wait_for_blocking_child(0)
                assert first_child._block_cleanup is True

                socket.send_json(_start(second_id, reference, mode="ai"))
                _trace, replaced = _receive_until_terminal(socket)
                assert replaced["payload"] == {"attemptId": first_id, "reason": "replaced"}
                assert first_child.cleanup_entered.wait(timeout=1)

                next_messages = []
                next_done = threading.Event()

                def read_next():
                    next_messages.append(socket.receive_json())
                    next_done.set()

                reader = threading.Thread(target=read_next, daemon=True)
                reader.start()
                if next_done.wait(timeout=0.15):
                    raise AssertionError(
                        f"message arrived before cleanup finished: {next_messages[0]}"
                    )
                first_child.release_cleanup()
                assert next_done.wait(timeout=2)
                reader.join(timeout=1)
                assert next_messages[0]["type"] == "vision.try_on.attempt.accepted"
                assert next_messages[0]["payload"] == {
                    "attemptId": second_id,
                    "mode": "ai",
                }
                _receive_until_ai_child_running(socket)
                second_child = _wait_for_blocking_child(1)
                second_child.release_run()
                _second_trace, completed = _receive_until_completed(socket)

            _assert_canceled_cleanup_and_no_result(
                client, first_id, first_child, "replaced", replaced
            )
            assert completed["type"] == "vision.try_on.attempt.completed"
            assert completed["payload"]["attemptId"] == second_id
    finally:
        _BlockingAiChild.instances.clear()
        _BlockingAiChild.block_first_cleanup = False
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_disconnect_kills_child_removes_staging_and_replays_one_terminal(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    _start_blocking_ai(monkeypatch)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _receive_until_ai_child_running(socket)
                child = _wait_for_blocking_child(0)
                socket.close()

            deadline = time.monotonic() + 2
            while not child.closed and time.monotonic() < deadline:
                time.sleep(0.01)
            with client.websocket_connect("/ws") as replay:
                replay.send_json(_hello())
                assert replay.receive_json()["type"] == "vision.ready"
                replay.send_json(_start(attempt_id, reference, mode="ai"))
                terminal = replay.receive_json()

            _assert_canceled_cleanup_and_no_result(
                client, attempt_id, child, "disconnect", terminal
            )
    finally:
        _BlockingAiChild.instances.clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_departure_kills_child_removes_staging_and_publishes_one_terminal(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    _start_blocking_ai(monkeypatch)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _receive_until_ai_child_running(socket)
                child = _wait_for_blocking_child(0)

                asyncio.run(vision_app._cancel_active_attempt("departure"))
                _trace, terminal = _receive_until_terminal(socket)

            _assert_canceled_cleanup_and_no_result(
                client, attempt_id, child, "departure", terminal
            )
    finally:
        _BlockingAiChild.instances.clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_timeout_kills_child_removes_staging_and_next_attempt_completes(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    _start_blocking_ai(monkeypatch)
    monkeypatch.setattr(vision_app, "_AI_ATTEMPT_TIMEOUT_SECONDS", 0.1)
    server, thread, reference = _serve_garment()
    try:
        timed_out_id, retry_id = str(uuid4()), str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(timed_out_id, reference, mode="ai"))
                _receive_until_ai_child_running(socket)
                first_child = _wait_for_blocking_child(0)
                _trace, timed_out = _receive_until_terminal(socket)

                socket.send_json(_start(retry_id, reference, mode="ai"))
                assert socket.receive_json()["type"] == "vision.try_on.attempt.accepted"
                _receive_until_ai_child_running(socket)
                retry_child = _wait_for_blocking_child(1)
                retry_child.release_run()
                _retry_trace, completed = _receive_until_completed(socket)

            _assert_canceled_cleanup_and_no_result(
                client, timed_out_id, first_child, "timeout", timed_out
            )
            assert completed["type"] == "vision.try_on.attempt.completed"
            assert completed["payload"]["attemptId"] == retry_id
    finally:
        _BlockingAiChild.instances.clear()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "mode",
    [
        "undecodable",
        "oversize-bytes",
        "oversize-dimensions",
        "wrong-format",
        "input-copy",
        "garment-copy",
        "missing-file",
        "extra-file",
        "path-escape",
        "worker-error",
    ],
)
def test_v2_ai_invalid_private_staging_outputs_fail_without_result_or_orphan(
    tmp_path, monkeypatch, mode
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    acceptance_root = tmp_path / "acceptance-regional"
    acceptance_root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(acceptance_root))
    if mode == "oversize-bytes":
        monkeypatch.setattr(vision_app, "_FAST_RESULT_MAX_BYTES", 8)
    _start_output_validation_ai(monkeypatch, mode)
    server, thread, reference = _serve_garment()
    child = None
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _trace, terminal = _receive_until_terminal(socket)
            child = _OutputValidationAiChild.instances[0]
            _assert_failed_cleanup_and_no_result(client, attempt_id, child, terminal)
            assert list(acceptance_root.iterdir()) == []

            with client.websocket_connect("/ws") as replay:
                replay.send_json(_hello())
                assert replay.receive_json()["type"] == "vision.ready"
                replay.send_json(_start(attempt_id, reference, mode="ai"))
                replayed = replay.receive_json()
            assert replayed == terminal
            assert len(_OutputValidationAiChild.instances) == 1
    finally:
        if child is not None and child.escape_path is not None:
            try:
                child.escape_path.unlink()
            except FileNotFoundError:
                pass
        _OutputValidationAiChild.instances.clear()
        server.shutdown()
        server.server_close()
        thread.join()


def test_ai_sparse_oversize_output_is_rejected_before_reading(monkeypatch, tmp_path):
    output = tmp_path / "output.png"
    with output.open("wb") as handle:
        handle.truncate(vision_app._FAST_RESULT_MAX_BYTES + 1)

    def fail_if_read(_path):
        raise AssertionError("oversize AI output must be rejected from stat before read_bytes")

    monkeypatch.setattr(Path, "read_bytes", fail_if_read)

    with pytest.raises(RuntimeError, match="ai_result_too_large"):
        asyncio.run(vision_app._read_ai_output_bytes(output))


def test_v2_ai_completed_terminal_encode_failure_has_no_orphan_result(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    acceptance_root = tmp_path / "acceptance-regional"
    acceptance_root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(acceptance_root))
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _DeterministicAiChild)
    original_envelope = vision_app._generated_v2_envelope

    def reject_completed(message_type, payload):
        if message_type == "vision.try_on.attempt.completed":
            raise ValueError("forced_completed_contract_failure")
        return original_envelope(message_type, payload)

    monkeypatch.setattr(vision_app, "_generated_v2_envelope", reject_completed)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _trace, terminal = _receive_until_terminal(socket)

            assert terminal["type"] == "vision.try_on.attempt.failed"
            assert terminal["payload"] == {"attemptId": attempt_id, "reason": "ai_failed"}
            assert client.get(
                f"/v2/try-on/results/{attempt_id}?token=sentinel"
            ).status_code == 404
            assert list(acceptance_root.iterdir()) == []

            with client.websocket_connect("/ws") as replay:
                replay.send_json(_hello())
                assert replay.receive_json()["type"] == "vision.ready"
                replay.send_json(_start(attempt_id, reference, mode="ai"))
                assert replay.receive_json() == terminal
    finally:
        _DeterministicAiChild.calls = 0
        server.shutdown()
        server.server_close()
        thread.join()


def test_v2_ai_result_admission_failure_does_not_export_uncommitted_sidecar(
    tmp_path, monkeypatch
):
    pack = tmp_path / "pack"
    _write_official_pack(pack)
    _configure_stage1_runtime(monkeypatch, pack)
    acceptance_root = tmp_path / "acceptance-regional"
    acceptance_root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(acceptance_root))
    monkeypatch.setattr(vision_app, "_ai_attempt_process_factory", _DeterministicAiChild)
    monkeypatch.setattr(vision_app._fast_attempt_registry._results, "single_max_bytes", 1)
    server, thread, reference = _serve_garment()
    try:
        attempt_id = str(uuid4())
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference, mode="ai"))
                _trace, terminal = _receive_until_terminal(socket)

            assert terminal["type"] == "vision.try_on.attempt.failed"
            assert terminal["payload"] == {
                "attemptId": attempt_id,
                "reason": "ai_failed",
            }
            assert client.get(
                f"/v2/try-on/results/{attempt_id}?token=sentinel"
            ).status_code == 404
            assert list(acceptance_root.iterdir()) == []
    finally:
        _DeterministicAiChild.calls = 0
        server.shutdown()
        server.server_close()
        thread.join()
