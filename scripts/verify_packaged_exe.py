from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from urllib.error import HTTPError
from datetime import datetime, timezone
from pathlib import Path

import websockets


CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "contracts" / "vem_vision_v2"
PROFILE_FIELDS = {
    "personPresent",
    "heightCm",
    "shoulderWidthCm",
    "ageRange",
    "gender",
    "bodyType",
    "upperColor",
    "confidence",
}


def create_managed_maintenance_fixture(temp_dir, *, port):
    """Create a managed config for the plain loopback v2 maintenance smoke."""
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    config_path = temp_dir / "managed-site.json"
    config_path.write_text(json.dumps({
        "schemaVersion": "vending-vision-site-config/v1",
        "host": "127.0.0.1",
        "port": port,
        "allowed_origins": [f"http://127.0.0.1:{port}", "http://tauri.localhost"],
        "cameras": {
            "top": {"backend": "dshow", "role": "presence", "keep_open": True, "rotate": 0},
            "front": {"backend": "dshow", "role": "profile_tryon", "keep_open": True, "rotate": 0},
        },
    }), encoding="utf-8")
    return config_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start a packaged vending-vision EXE and verify its runtime contract."
    )
    parser.add_argument(
        "exe",
        nargs="?",
        default="dist/vending-vision/vending-vision.exe",
    )
    parser.add_argument("--port", type=int, default=17892)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--expected-version")
    return parser.parse_args()


def message(message_type, message_id, payload=None):
    return {
        "protocol": v2_runtime_identity()["protocol"],
        "type": message_type,
        "messageId": message_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload or {},
    }


def v2_runtime_identity():
    manifest = json.loads((CONTRACT_ROOT / "manifest.json").read_text("utf-8"))
    return {
        "protocol": manifest["protocol"],
        "schemaVersion": manifest["schemaVersion"],
        "bundleVersion": manifest["bundleVersion"],
        "contractDigest": manifest["bundleDigest"],
    }


def v2_handshake_identity():
    """Identity fields permitted inside the vision.hello payload."""
    identity = v2_runtime_identity()
    return {
        "schemaVersion": identity["schemaVersion"],
        "bundleVersion": identity["bundleVersion"],
        "contractDigest": identity["contractDigest"],
    }


def http_get_json(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise AssertionError(f"GET {url} returned {response.status}")
        return json.loads(response.read().decode("utf-8"))


def http_get_text(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise AssertionError(f"GET {url} returned {response.status}")
        return response.read().decode("utf-8")


def http_status_json(url, timeout=5.0, *, method="GET", headers=None):
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def http_status(url, timeout=5.0, *, method="GET", headers=None):
    request = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def wait_for_http(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"packaged process exited early: {process.returncode}")
        try:
            return http_get_json(f"{base_url}/health", timeout=2.0)
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise TimeoutError(f"service did not start within {timeout}s: {last_error}")


def assert_v2_contract_resources(contract_root):
    expected_contract_resources = {
        "manifest.json",
        "__init__.py",
        "python/__init__.py",
        "python/vision_v2_models.py",
        "vision-v2.schema.json",
        "fixtures/valid.json",
        "fixtures/invalid.json",
    }
    if not contract_root.is_dir():
        raise AssertionError(f"missing packaged V2 contract directory: {contract_root}")
    actual_contract_resources = {
        path.relative_to(contract_root).as_posix()
        for path in contract_root.rglob("*")
        if path.is_file()
    }
    if actual_contract_resources != expected_contract_resources:
        raise AssertionError(
            "packaged V2 contract resources are not exact: "
            f"{sorted(actual_contract_resources)}",
        )
    if any(path.suffix == ".pyc" for path in contract_root.rglob("*")):
        raise AssertionError("packaged V2 contract resources must not contain Python bytecode")


def assert_bundled_resources(exe_path):
    internal = exe_path.parent / "_internal"
    contract_root = internal / "contracts" / "vem_vision_v2"
    required = [
        internal / "config.json",
        internal / "dashboard" / "profile_dashboard.html",
        internal / "config" / "vending-vision-camera-maintenance-v2.schema.json",
        internal / "config" / "vending-vision-camera-maintenance-v2.requests.schema.json",
        internal / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json",
        contract_root / "manifest.json",
        contract_root / "__init__.py",
        contract_root / "python" / "__init__.py",
        contract_root / "python" / "vision_v2_models.py",
        contract_root / "vision-v2.schema.json",
        contract_root / "fixtures" / "valid.json",
        contract_root / "fixtures" / "invalid.json",
        internal / "models" / "person_detection" / "person_yolov8n.onnx",
        internal / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx",
        internal / "models" / "age_gender" / "age_net.caffemodel",
        internal / "models" / "age_gender" / "gender_net.caffemodel",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"missing packaged resources: {missing}")
    assert_v2_contract_resources(contract_root)
    retired = [
        internal / "config" / "vending-vision-camera-maintenance-v1.schema.json",
    ]
    present_retired = [str(path) for path in retired if path.exists()]
    if present_retired:
        raise AssertionError(f"retired packaged resources must not be shipped: {present_retired}")
    camera_adapter = internal / "cv2_enumerate_cameras"
    if sys.platform == "win32" and not any(camera_adapter.glob("_windows_backend*.pyd")):
        raise AssertionError("missing packaged cv2-enumerate-cameras Windows DirectShow adapter")


async def verify_websocket(port):
    uri = f"ws://127.0.0.1:{port}/ws"
    async with websockets.connect(uri, origin=f"http://127.0.0.1:{port}") as websocket:
        await websocket.send(json.dumps(message("vision.ping", "pre-hello")))
        pre_hello = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5.0))
        if pre_hello.get("type") != "vision.error":
            raise AssertionError("business message before hello was not rejected")
        if pre_hello.get("payload", {}).get("code") != "invalid_message":
            raise AssertionError(f"unexpected pre-hello error: {pre_hello}")

        await websocket.send(
            json.dumps(
                message(
                    "vision.hello",
                    "hello-packaged",
                    {
                        **v2_handshake_identity(),
                        "capabilities": [
                            "profile_push",
                            "presence_status",
                            "person_departed",
                            "ambient_light",
                            "try_on_fast",
                        ],
                        "clientRole": "machine",
                        "machineCode": "PACKAGED-TEST",
                    },
                )
            )
        )
        ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=8.0))
        if ready.get("type") != "vision.ready":
            raise AssertionError(f"expected vision.ready, got: {ready}")
        capabilities = set(ready.get("payload", {}).get("capabilities") or [])
        if "try_on_fast" not in capabilities or "profile_push" not in capabilities:
            raise AssertionError(f"server capabilities are incomplete: {capabilities}")

        await websocket.send(json.dumps(message("vision.ping", "ping-packaged")))
        received_types = []
        profile_result = None
        deadline = asyncio.get_running_loop().time() + 10.0
        while asyncio.get_running_loop().time() < deadline:
            event = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5.0))
            received_types.append(event.get("type"))
            if event.get("type") == "vision.profile_result":
                profile_result = event
                break
        if "vision.pong" not in received_types:
            raise AssertionError(f"vision.pong was not received: {received_types}")
        if profile_result is None:
            raise AssertionError(f"mock profile_result was not received: {received_types}")
        profile = profile_result.get("payload", {}).get("profile") or {}
        if set(profile) != PROFILE_FIELDS:
            raise AssertionError(f"profile contract mismatch: {sorted(profile)}")
        occupancy = profile_result.get("payload", {}).get("occupancy", {}).get("state")
        if occupancy != "single":
            raise AssertionError(f"profile occupancy is not single: {occupancy}")


def ensure_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"port {port} is already in use") from exc


def terminate_packaged_process(process, process_log, *, verification_failed=False):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    process_log.seek(0)
    output = process_log.read()
    if output and (verification_failed or process.returncode not in {0, 1, -15}):
        print(output, file=sys.stderr)


def verify_plain_camera_maintenance_contract(base_url):
    """Exercise the real packaged adapter through plain loopback v2 routes."""
    read_status, contract = http_status_json(f"{base_url}/maintenance/cameras")
    if read_status != 200:
        raise AssertionError(f"plain camera read failed: {read_status} {contract}")
    if contract.get("contractVersion") != "vem.vision.camera-maintenance/v2":
        raise AssertionError(f"plain camera contract version mismatch: {contract}")
    if not isinstance(contract.get("candidates"), list) or not isinstance(contract.get("roles"), dict):
        raise AssertionError(f"plain camera contract is incomplete: {contract}")

    refresh_status, refreshed = http_status_json(
        f"{base_url}/maintenance/cameras/refresh",
        method="POST",
    )
    if refresh_status != 200:
        raise AssertionError(f"plain camera refresh failed: {refresh_status} {refreshed}")
    if refreshed.get("contractVersion") != "vem.vision.camera-maintenance/v2":
        raise AssertionError(f"plain refreshed camera contract version mismatch: {refreshed}")


def verify_managed_production_surface(exe_path, *, port, startup_timeout, temp_dir):
    """Run managed production mode, not the supplier development/dashboard mode."""
    config_path = create_managed_maintenance_fixture(temp_dir, port=port)
    env = os.environ.copy()
    env.update({
        "VISION_OPEN_BROWSER": "0",
        "VISION_MOCK_SCENARIO": "off",
        "VISION_DEVELOPMENT_DASHBOARD": "true",  # managed mode must still hide it
        "VISION_WORKDIR": str(temp_dir),
    })
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as process_log:
        process = subprocess.Popen(
            [str(exe_path), "--no-browser", "--config", str(config_path)],
            cwd=str(exe_path.parent),
            env=env,
            stdout=process_log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        verification_failed = False
        try:
            base_url = f"http://127.0.0.1:{port}"
            wait_for_http(base_url, process, startup_timeout)
            verify_plain_camera_maintenance_contract(base_url)
            for legacy_url in ("/dashboard", "/camera/top/snapshot.jpg", "/camera/top/reopen"):
                method = "POST" if legacy_url.endswith("/reopen") else "GET"
                status = http_status(f"{base_url}{legacy_url}", method=method)
                if status != 404:
                    raise AssertionError(f"managed production unexpectedly exposed {legacy_url}: {status}")
        except BaseException:
            verification_failed = True
            raise
        finally:
            terminate_packaged_process(
                process, process_log, verification_failed=verification_failed
            )


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = root / exe_path
    exe_path = exe_path.resolve()
    if not exe_path.is_file():
        raise FileNotFoundError(exe_path)

    assert_bundled_resources(exe_path)
    probe = subprocess.run(
        [str(exe_path), "--verify-v2-contract-bundle"],
        cwd=str(exe_path.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if probe.returncode != 0 or "V2 contract bundle probe passed" not in probe.stdout:
        raise AssertionError(
            f"packaged V2 contract probe failed: {probe.stdout}{probe.stderr}",
        )
    ensure_port_available(args.port)
    managed_port = args.port + 1
    ensure_port_available(managed_port)

    with tempfile.TemporaryDirectory(prefix="vending-vision-package-") as temp_dir:
        temp_dir = Path(temp_dir)
        # Exercise the supported supplier-development config path; managed
        # production is verified separately below and must keep this route hidden.
        dev_config_path = temp_dir / "config.json"
        dev_config_path.write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": args.port,
                    "development_dashboard": True,
                }
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            {
                "VISION_MOCK_SCENARIO": "success",
                "VISION_OPEN_BROWSER": "0",
                "VISION_CONFIG_FILE": str(dev_config_path),
                "VISION_WORKDIR": str(temp_dir),
            }
        )
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as process_log:
            process = subprocess.Popen(
                [str(exe_path)],
                cwd=str(temp_dir),
                env=env,
                stdout=process_log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            verification_failed = False
            try:
                base_url = f"http://127.0.0.1:{args.port}"
                health = wait_for_http(base_url, process, args.startup_timeout)
                if not health.get("checks", {}).get("pose", {}).get("ok"):
                    raise AssertionError(f"packaged pose check failed: {health}")
                if not health.get("checks", {}).get("face", {}).get("ok"):
                    raise AssertionError(f"packaged face check failed: {health}")
                version = http_get_json(f"{base_url}/version")
                if version.get("protocol") != v2_runtime_identity()["protocol"]:
                    raise AssertionError(f"protocol mismatch: {version}")
                if args.expected_version and version.get("version") != args.expected_version:
                    raise AssertionError(
                        f"release version mismatch: expected {args.expected_version}, got {version}"
                    )
                metrics = http_get_json(f"{base_url}/metrics")
                if not isinstance(metrics, dict):
                    raise AssertionError("metrics endpoint did not return an object")
                verify_plain_camera_maintenance_contract(base_url)
                asyncio.run(verify_websocket(args.port))
            except BaseException:
                verification_failed = True
                raise
            finally:
                terminate_packaged_process(
                    process, process_log, verification_failed=verification_failed
                )
        verify_managed_production_surface(
            exe_path,
            port=managed_port,
            startup_timeout=args.startup_timeout,
            temp_dir=Path(temp_dir) / "managed-production",
        )
        print("PACKAGED_EXE_VERIFICATION=PASS")
        print(f"EXE={exe_path}")
        print(f"SERVER_VERSION={version.get('version')}")
        print(f"AGE_GENDER_MODE={health.get('ageGenderMode')}")

if __name__ == "__main__":
    main()
