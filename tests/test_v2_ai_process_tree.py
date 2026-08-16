import asyncio
import contextlib
import hashlib
import json
import math
import os
import socket as socket_module
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
import cv2
import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import app as vision_app
from vision import camera_manager, presence_runtime
import vision.ai_attempt_process as ai_attempt_process_module
from vision.ai_attempt_process import AiAttemptProcess
from vision.ai_model_pack import OfficialAiReadinessSnapshot
from vision.config import settings
from vision.profile_state import get_occupancy_gate, reset_active_track


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group tracer")
ROOT = Path(__file__).parents[1]
WORKER = Path(__file__).with_name("ai_process_tree_worker.py")
RECORDED_FIXTURES = ROOT / "fixtures" / "recorded-video"


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
    worker_args: list[str] | None = None,
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
            {
                "adapter": "recorded_video",
                "configSha256": "7" * 64,
                "decodedFrameCount": 42,
                "fixtureSha256": "8" * 64,
                "frameIndex": 7,
                "relabeled": False,
                "role": "front",
                "synthetic": False,
            },
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
                [
                    "--success-png",
                    str(success_png),
                    "--output",
                    str(kwargs["output_png"]),
                ]
            )
            if kwargs["regional_evidence_output"] is not None:
                command.extend(
                    [
                        "--regional-evidence-output",
                        str(kwargs["regional_evidence_output"]),
                        "--person",
                        str(kwargs["person_png"]),
                        "--garment",
                        str(kwargs["garment_png"]),
                        "--captured-source",
                        json.dumps(kwargs["captured_source"], sort_keys=True, separators=(",", ":")),
                    ]
                )
        command.extend(worker_args or [])
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
        message = _receive_json_deadline(socket)
        trace.append(message)
        if (
            message["type"] == "vision.try_on.attempt.generating"
            and message["payload"]["stage"] == "generating"
        ):
            return trace


def _receive_until_terminal(socket) -> tuple[list[dict], dict]:
    trace = []
    while True:
        message = _receive_json_deadline(socket)
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


def _receive_json_deadline(socket, *, timeout: float = 3.0):
    portal = getattr(socket, "portal", None)
    receive_stream = getattr(socket, "_send_rx", None)
    raise_on_close = getattr(socket, "_raise_on_close", None)
    if portal is None or receive_stream is None or raise_on_close is None:
        raise AssertionError("unsupported Starlette WebSocketTestSession boundary")

    async def receive_message():
        with anyio.fail_after(timeout):
            return await receive_stream.receive()

    try:
        message = portal.call(receive_message)
    except TimeoutError as exc:
        raise TimeoutError("websocket receive deadline exceeded") from exc
    raise_on_close(message)
    return json.loads(message["text"])


def _http_get_deadline(base_url: str, path: str, *, timeout: float = 3.0):
    try:
        return httpx.get(
            f"{base_url}{path}",
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            trust_env=False,
        )
    except httpx.TimeoutException as exc:
        raise TimeoutError(f"HTTP GET {path} deadline exceeded") from exc


@contextlib.contextmanager
def _production_http_loopback(*, extra_routes=()):
    production_paths = {
        "/",
        "/health",
        "/v2/try-on/results/{attempt_id}",
    }
    production_routes = [
        route
        for route in vision_app.app.routes
        if getattr(route, "path", None) in production_paths
    ]
    assert {route.path for route in production_routes} == production_paths
    router = FastAPI()
    router.router.routes = [*production_routes, *extra_routes]
    listener = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    listener.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    host, port = listener.getsockname()
    config = uvicorn.Config(
        router,
        host=host,
        port=port,
        loop="asyncio",
        http="h11",
        lifespan="off",
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_errors = []

    def serve():
        try:
            server.run(sockets=[listener])
        except BaseException as exc:
            server_errors.append(exc)

    server_thread = threading.Thread(
        target=serve,
        name=f"uvicorn-loopback-{uuid4()}",
    )
    server_thread.start()
    startup_deadline = time.monotonic() + 3.0
    while (
        not server.started
        and server_thread.is_alive()
        and time.monotonic() < startup_deadline
    ):
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        server_thread.join(2.0)
        listener.close()
        raise AssertionError(f"loopback Uvicorn failed startup: {server_errors}")
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        server_thread.join(3.0)
        if server_thread.is_alive():
            server.force_exit = True
            server_thread.join(2.0)
        listener.close()
        assert not server_thread.is_alive(), "loopback Uvicorn failed bounded shutdown"
        assert server_errors == []


def _deadline_probe_app(exits: dict[str, int]) -> FastAPI:
    probe = FastAPI()

    @probe.websocket("/ws")
    async def silent(socket: WebSocket):
        await socket.accept()
        try:
            await socket.receive()
        finally:
            exits["websocket"] += 1

    @probe.get("/health")
    def health():
        return {"status": "ok"}

    return probe


def test_real_public_websocket_receive_deadline_cancels_without_residue():
    exits = {"websocket": 0, "http": 0}
    app = _deadline_probe_app(exits)

    with TestClient(app) as client:
        for expected_exits in range(1, 6):
            with client.websocket_connect("/ws") as socket:
                started = time.monotonic()
                with pytest.raises(TimeoutError, match="websocket receive deadline"):
                    _receive_json_deadline(socket, timeout=0.05)
                assert time.monotonic() - started < 0.5
            assert exits["websocket"] == expected_exits
        assert client.get("/health").json() == {"status": "ok"}

    assert exits["websocket"] == 5
    assert not any(
        thread.name.startswith("deadline-") for thread in threading.enumerate()
    )


def test_real_blocked_http_deadline_cancels_and_keeps_client_usable():
    exits = {"websocket": 0, "http": 0}
    handler_started = threading.Event()
    handler_finished = threading.Event()
    request_threads = []
    health_route = next(
        route for route in vision_app.app.routes if getattr(route, "path", None) == "/health"
    )

    def blocked_health():
        request_threads.append(threading.current_thread())
        handler_started.set()
        try:
            time.sleep(0.6)
            return health_route.endpoint()
        finally:
            exits["http"] += 1
            handler_finished.set()

    with TestClient(vision_app.app) as client:
        with _production_http_loopback(
            extra_routes=[APIRoute("/blocked", blocked_health, methods=["GET"])]
        ) as base_url:
            for expected_exits in range(1, 6):
                handler_started.clear()
                handler_finished.clear()
                started = time.monotonic()
                with pytest.raises(TimeoutError, match="HTTP GET /blocked deadline"):
                    _http_get_deadline(base_url, "/blocked", timeout=0.05)
                assert time.monotonic() - started < 0.5
                assert handler_started.wait(0.5)
                assert handler_finished.wait(1.0)
                assert exits["http"] == expected_exits
                assert _http_get_deadline(base_url, "/health").status_code == 200
        assert client.get("/health").status_code == 200

    assert exits["http"] == 5
    assert all(not thread.is_alive() for thread in request_threads)
    assert not any(
        thread.name.startswith("uvicorn-loopback-") for thread in threading.enumerate()
    )


def _process_usage(pid: int) -> dict[str, int]:
    stat = (Path("/proc") / str(pid) / "stat").read_text("utf-8")
    fields = stat[stat.rfind(")") + 2 :].split()
    status = (Path("/proc") / str(pid) / "status").read_text("utf-8")
    rss_kib = next(
        int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:")
    )
    return {
        "cpuTicks": int(fields[11]) + int(fields[12]),
        "nice": int(fields[16]),
        "rssKiB": rss_kib,
    }


def _latency_distribution(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def test_latency_distribution_uses_nearest_rank_for_small_samples():
    distribution = _latency_distribution([0.01] * 7 + [9.0])

    assert distribution["max"] == 9.0
    assert distribution["p95"] == 9.0


def _success_png(path: Path) -> None:
    image = np.full((42, 54, 4), (25, 190, 75, 255), dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", image)
    assert encoded
    path.write_bytes(payload.tobytes())


def _configure_recorded_presence(monkeypatch, tmp_path: Path) -> tuple[dict, str]:
    manifest = json.loads(
        (RECORDED_FIXTURES / "expected-results.json").read_text("utf-8")
    )
    top = manifest["recordings"]["top"]
    front = manifest["recordings"]["manFront"]
    top_config = {
        "role": "presence",
        "source": "recorded_video",
        "video_path": str(RECORDED_FIXTURES / top["file"]),
        "loop": top["loop"],
        "rotate": 0,
    }
    front_config = {
        "role": "profile_fast_try_on",
        "source": "recorded_video",
        "video_path": str(RECORDED_FIXTURES / front["file"]),
        "loop": front["loop"],
        "rotate": 0,
    }
    managed = {
        "schemaVersion": "vending-vision-site-config/v1",
        "host": "127.0.0.1",
        "port": 7892,
        "allowed_origins": ["http://127.0.0.1:7892"],
        "cameras": {"top": top_config, "front": front_config},
    }
    managed_bytes = (json.dumps(managed, indent=2) + "\n").encode("utf-8")
    managed_path = tmp_path / "recorded-site.json"
    managed_path.write_bytes(managed_bytes)
    monkeypatch.setenv("VISION_CONFIG_FILE", str(managed_path))
    monkeypatch.setattr(settings, "TOP_CAMERA_CONFIG", top_config)
    monkeypatch.setattr(settings, "FRONT_CAMERA_CONFIG", front_config)
    monkeypatch.setattr(presence_runtime, "_runtime", None)
    gate = get_occupancy_gate()
    for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
        gate.mark_absent()
    reset_active_track()
    camera_manager.release_all_cameras()
    return manifest, hashlib.sha256(managed_bytes).hexdigest()


def test_public_ai_bounded_cpu_rss_pressure_keeps_ws_presence_profile_and_core_live(
    tmp_path, monkeypatch
):
    production_get_runtime_status = vision_app.get_runtime_status
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "pressure-tree.json"
    _configure_public_ai(
        monkeypatch,
        pack,
        pid_file,
        "stress",
        worker_args=["--stress-mib", "16", "--stress-seconds", "8"],
    )
    _configure_recorded_presence(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "PROFILE_PUSH_ENABLED", True)
    monkeypatch.setattr(settings, "PROFILE_PUSH_INTERVAL_MS", 300)
    monkeypatch.setattr(vision_app, "get_runtime_status", production_get_runtime_status)
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    try:
        with (
            TestClient(vision_app.app) as client,
            _production_http_loopback() as http_base_url,
        ):
            with client.websocket_connect("/ws") as socket:
                socket.send_json(
                    _hello(
                        "profile_push",
                        "presence_status",
                        "person_departed",
                        "ambient_light",
                    )
                )
                ready = _receive_json_deadline(socket)
                initial_health = _http_get_deadline(http_base_url, "/health")
                socket.send_json(_start(attempt_id, reference))
                trace = _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                assert all(_pid_alive(pid) for pid in tree.values())

                initial_usage = {
                    name: _process_usage(pid) for name, pid in tree.items()
                }
                peak_usage = {
                    name: dict(usage) for name, usage in initial_usage.items()
                }
                sampler_errors = []

                def sample_process_tree():
                    deadline = time.monotonic() + 1.5
                    try:
                        while time.monotonic() < deadline:
                            for name, pid in tree.items():
                                usage = _process_usage(pid)
                                peak_usage[name]["cpuTicks"] = max(
                                    peak_usage[name]["cpuTicks"], usage["cpuTicks"]
                                )
                                peak_usage[name]["rssKiB"] = max(
                                    peak_usage[name]["rssKiB"], usage["rssKiB"]
                                )
                                peak_usage[name]["nice"] = usage["nice"]
                            time.sleep(0.025)
                    except Exception as exc:
                        sampler_errors.append(exc)

                core_latencies = []
                core_statuses = []
                core_errors = []

                def exercise_core():
                    try:
                        for path in ("/", "/health") * 4:
                            started = time.monotonic()
                            response = _http_get_deadline(http_base_url, path)
                            core_latencies.append(time.monotonic() - started)
                            core_statuses.append((path, response.status_code))
                    except Exception as exc:
                        core_errors.append(exc)

                sampler = threading.Thread(target=sample_process_tree)
                core_worker = threading.Thread(target=exercise_core)
                sampler.start()
                core_worker.start()

                ping_latencies = []
                presence_count = sum(
                    message["type"] == "vision.presence_status" for message in trace
                )
                profile_seen = any(
                    message["type"] == "vision.profile_result" for message in trace
                )
                response_deadline = time.monotonic() + 3.5
                ping_pacer = threading.Event()
                while time.monotonic() < response_deadline:
                    started = time.monotonic()
                    socket.send_json(_envelope("vision.ping", {}))
                    while True:
                        message = _receive_json_deadline(socket)
                        trace.append(message)
                        if message["type"] == "vision.pong":
                            ping_latencies.append(time.monotonic() - started)
                            break
                        if message["type"] == "vision.presence_status":
                            presence_count += 1
                        elif message["type"] == "vision.profile_result":
                            profile_seen = True
                    if (
                        len(ping_latencies) >= 5
                        and presence_count >= 2
                        and profile_seen
                        and not sampler.is_alive()
                        and not core_worker.is_alive()
                    ):
                        break
                    # Pacing is not the correctness clock: the outer deadline,
                    # sampler completion and public events decide success.
                    ping_pacer.wait(
                        min(0.075, settings.PROFILE_PUSH_INTERVAL_MS / 1000.0 / 4)
                    )

                sampler.join(timeout=2)
                core_worker.join(timeout=2)
                assert sampler.is_alive() is False
                assert core_worker.is_alive() is False
                assert sampler_errors == []
                assert core_errors == []

                socket.send_json(
                    _envelope(
                        "vision.try_on.attempt.cancel",
                        {"attemptId": attempt_id, "reason": "user"},
                    )
                )
                terminal_trace, terminal = _receive_until_terminal(socket)
                trace.extend(terminal_trace)
                _assert_tree_dead(tree)
                _assert_staging_clear(attempt_id)
                socket.send_json(_envelope("vision.ping", {}))
                while True:
                    message = _receive_json_deadline(socket)
                    trace.append(message)
                    if message["type"] == "vision.pong":
                        final_pong = message
                        break

            missing = _http_get_deadline(
                http_base_url,
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            )
            final_health = _http_get_deadline(http_base_url, "/health")
            with client.websocket_connect("/ws") as final_socket:
                final_socket.send_json(_hello())
                final_ready = _receive_json_deadline(final_socket)

        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
        cpu_deltas = {
            name: peak_usage[name]["cpuTicks"] - initial_usage[name]["cpuTicks"]
            for name in tree
        }
        rss_peaks = {name: peak_usage[name]["rssKiB"] for name in tree}
        observed_nice = {name: peak_usage[name]["nice"] for name in tree}
        ping_distribution = _latency_distribution(ping_latencies)
        core_distribution = _latency_distribution(core_latencies)
        latency_limit = settings.PROFILE_PUSH_INTERVAL_MS / 1000.0 * 10
        pressure_evidence = {
            "cpuSeconds": {
                name: round(delta / clock_ticks, 3) for name, delta in cpu_deltas.items()
            },
            "rssPeakKiB": rss_peaks,
            # Windows is the target priority boundary.  Linux is deliberately
            # observation-only until a safe post-spawn priority policy exists.
            "nice": {"main": os.getpriority(os.PRIO_PROCESS, 0), **observed_nice},
            "pingLatencySeconds": ping_distribution,
            "coreLatencySeconds": core_distribution,
        }
        print(json.dumps({"pressureEvidence": pressure_evidence}, sort_keys=True))

        assert ready["payload"]["fastReady"] is True
        assert ready["payload"]["visionBusinessReady"] is True
        assert {
            key: initial_health.json()[key]
            for key in ("status", "cameraReady", "modelReady", "aiReady")
        } == {
            key: final_health.json()[key]
            for key in ("status", "cameraReady", "modelReady", "aiReady")
        } == {
            "status": "ok",
            "cameraReady": True,
            "modelReady": True,
            "aiReady": True,
        }
        assert len(ping_latencies) >= 5
        assert presence_count >= 2
        assert profile_seen is True
        assert all(0 <= sample < latency_limit for sample in ping_latencies), ping_latencies
        assert all(0 <= sample < latency_limit for sample in core_latencies), core_latencies
        assert ping_distribution["p95"] < latency_limit, ping_distribution
        assert ping_distribution["max"] < latency_limit, ping_distribution
        assert core_distribution["p95"] < latency_limit, core_distribution
        assert core_distribution["max"] < latency_limit, core_distribution
        assert core_statuses == [("/", 200), ("/health", 200)] * 4
        assert all(delta >= clock_ticks // 5 for delta in cpu_deltas.values()), cpu_deltas
        assert all(16 * 1024 <= rss <= 64 * 1024 for rss in rss_peaks.values()), rss_peaks
        assert sum(rss_peaks.values()) <= 192 * 1024, rss_peaks
        assert all(-20 <= nice <= 19 for nice in observed_nice.values()), observed_nice
        assert terminal["type"] == "vision.try_on.attempt.canceled"
        assert terminal["payload"] == {"attemptId": attempt_id, "reason": "user"}
        assert final_pong["type"] == "vision.pong"
        assert missing.status_code == 404
        assert final_health.status_code == 200
        assert final_ready["payload"]["fastReady"] is True
        assert final_ready["payload"]["visionBusinessReady"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        camera_manager.release_all_cameras()


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


def test_public_ai_disconnect_joins_real_tree_and_replays_only_disconnect_terminal(
    tmp_path, monkeypatch
):
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "disconnect-tree.json"
    _configure_public_ai(monkeypatch, pack, pid_file, "block")
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(_hello())
                assert socket.receive_json()["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference))
                _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                assert all(_pid_alive(pid) for pid in tree.values())
                socket.close()

            _assert_tree_dead(tree)
            _assert_staging_clear(attempt_id)
            missing = client.get(
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            )

            with client.websocket_connect("/ws") as replay_socket:
                replay_socket.send_json(_hello())
                ready = replay_socket.receive_json()
                replay_socket.send_json(_envelope("vision.ping", {}))
                pong_before = replay_socket.receive_json()
                replay_socket.send_json(_start(attempt_id, reference))
                replay = replay_socket.receive_json()
                replay_socket.send_json(_envelope("vision.ping", {}))
                late_messages = []
                while True:
                    message = replay_socket.receive_json()
                    if message["type"] == "vision.pong":
                        pong_after = message
                        break
                    late_messages.append(message)

        assert ready["type"] == "vision.ready"
        assert pong_before["type"] == "vision.pong"
        assert replay["type"] == "vision.try_on.attempt.canceled"
        assert replay["payload"] == {
            "attemptId": attempt_id,
            "reason": "disconnect",
        }
        assert all(
            message["type"]
            not in {
                "vision.try_on.attempt.completed",
                "vision.try_on.attempt.failed",
                "vision.try_on.attempt.canceled",
            }
            for message in late_messages
        )
        assert pong_after["type"] == "vision.pong"
        assert missing.status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_public_recorded_departure_does_not_cancel_real_ai_tree_and_keeps_core_live(
    tmp_path, monkeypatch
):
    """A production presence departure must not cancel the AI attempt; explicit user cancel still kills the real tree."""
    pack = tmp_path / "test-owned-pack"
    pack.mkdir()
    pid_file = tmp_path / "departure-tree.json"
    _configure_public_ai(monkeypatch, pack, pid_file, "block")
    manifest, managed_config_sha = _configure_recorded_presence(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "PROFILE_PUSH_ENABLED", True)
    monkeypatch.setattr(settings, "PROFILE_PUSH_INTERVAL_MS", 167)
    server, thread, reference = _serve_garment()
    attempt_id = str(uuid4())
    try:
        with TestClient(vision_app.app) as client:
            with client.websocket_connect("/ws") as socket:
                socket.send_json(
                    _hello(
                        "profile_push",
                        "presence_status",
                        "person_departed",
                        "ambient_light",
                    )
                )
                ready = socket.receive_json()
                assert ready["payload"]["aiReady"] is True
                socket.send_json(_start(attempt_id, reference))
                trace = _receive_until_generating(socket)
                tree = _wait_pid_tree(pid_file)
                assert all(_pid_alive(pid) for pid in tree.values())

                canceled = None
                departure = next(
                    (
                        message
                        for message in trace
                        if message["type"] == "vision.person_departed"
                    ),
                    None,
                )
                profile_seen = any(
                    message["type"] == "vision.profile_result" for message in trace
                )
                while departure is None or not profile_seen:
                    message = socket.receive_json()
                    trace.append(message)
                    if message["type"] == "vision.try_on.attempt.canceled":
                        canceled = message
                    elif message["type"] == "vision.person_departed":
                        departure = message
                    elif message["type"] == "vision.profile_result":
                        profile_seen = True

                # The departure edge must not have canceled the attempt and the
                # real AI process tree must still be alive and owned.
                assert canceled is None
                assert all(_pid_alive(pid) for pid in tree.values())

                # Explicit user cancel is the bounded terminal that still kills
                # the whole owned process tree and clears the staged output.
                socket.send_json(
                    _envelope(
                        "vision.try_on.attempt.cancel",
                        {"attemptId": attempt_id, "reason": "user"},
                    )
                )
                while True:
                    message = socket.receive_json()
                    trace.append(message)
                    if message["type"] == "vision.try_on.attempt.canceled":
                        canceled = message
                        break

                _assert_tree_dead(tree)
                _assert_staging_clear(attempt_id)
                socket.send_json(_envelope("vision.ping", {}))
                while True:
                    message = socket.receive_json()
                    trace.append(message)
                    if message["type"] == "vision.pong":
                        pong = message
                        break

            missing = client.get(
                f"/v2/try-on/results/{attempt_id}?token=no-result"
            )
            core = client.get("/")

        assert canceled is not None
        assert canceled["payload"] == {"attemptId": attempt_id, "reason": "user"}
        terminals = [
            message
            for message in trace
            if message["type"]
            in {
                "vision.try_on.attempt.completed",
                "vision.try_on.attempt.failed",
                "vision.try_on.attempt.canceled",
            }
            and message["payload"]["attemptId"] == attempt_id
        ]
        assert terminals == [canceled]
        assert departure is not None
        source_frame = departure["payload"]["sourceFrame"]
        assert source_frame["adapter"] == "recorded_video"
        assert source_frame["role"] == "top"
        assert source_frame["fixtureSha256"] == manifest["recordings"]["top"]["sha256"]
        assert source_frame["configSha256"] == managed_config_sha
        assert source_frame["synthetic"] is False
        assert source_frame["relabeled"] is False
        assert profile_seen is True
        assert any(
            message["type"] == "vision.presence_status" for message in trace
        )
        assert pong["type"] == "vision.pong"
        assert missing.status_code == 404
        assert core.status_code == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        camera_manager.release_all_cameras()


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
