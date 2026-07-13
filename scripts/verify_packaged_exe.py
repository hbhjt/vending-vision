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
from datetime import datetime, timezone
from pathlib import Path

import websockets


PROTOCOL = "vem.vision.v1"
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
        "protocol": PROTOCOL,
        "type": message_type,
        "messageId": message_id,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload or {},
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


def assert_bundled_resources(exe_path):
    internal = exe_path.parent / "_internal"
    required = [
        internal / "config.json",
        internal / "dashboard" / "profile_dashboard.html",
        internal / "models" / "person_detection" / "person_yolov8n.onnx",
        internal / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx",
        internal / "models" / "age_gender" / "age_net.caffemodel",
        internal / "models" / "age_gender" / "gender_net.caffemodel",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"missing packaged resources: {missing}")


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
                        "protocolVersion": 1,
                        "capabilities": [
                            "profile_push",
                            "presence_status",
                            "person_departed",
                            "ambient_light",
                            "try_on_session",
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
        if "try_on_session" not in capabilities or "profile_push" not in capabilities:
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
    ensure_port_available(args.port)

    with tempfile.TemporaryDirectory(prefix="vending-vision-package-") as temp_dir:
        env = os.environ.copy()
        env.update(
            {
                "VISION_HOST": "127.0.0.1",
                "VISION_PORT": str(args.port),
                "VISION_MOCK_SCENARIO": "success",
                "VISION_OPEN_BROWSER": "0",
                "VISION_WORKDIR": temp_dir,
            }
        )
        process = subprocess.Popen(
            [str(exe_path)],
            cwd=str(exe_path.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            health = wait_for_http(base_url, process, args.startup_timeout)
            if not health.get("checks", {}).get("pose", {}).get("ok"):
                raise AssertionError(f"packaged pose check failed: {health}")
            if not health.get("checks", {}).get("face", {}).get("ok"):
                raise AssertionError(f"packaged face check failed: {health}")
            version = http_get_json(f"{base_url}/version")
            if version.get("protocol") != PROTOCOL:
                raise AssertionError(f"protocol mismatch: {version}")
            if args.expected_version and version.get("version") != args.expected_version:
                raise AssertionError(
                    f"release version mismatch: expected {args.expected_version}, got {version}"
                )
            dashboard = http_get_text(f"{base_url}/dashboard")
            if "WebSocket" not in dashboard and "profile" not in dashboard.lower():
                raise AssertionError("packaged dashboard content is incomplete")
            metrics = http_get_json(f"{base_url}/metrics")
            if not isinstance(metrics, dict):
                raise AssertionError("metrics endpoint did not return an object")
            asyncio.run(verify_websocket(args.port))
            print("PACKAGED_EXE_VERIFICATION=PASS")
            print(f"EXE={exe_path}")
            print(f"SERVER_VERSION={version.get('version')}")
            print(f"AGE_GENDER_MODE={health.get('ageGenderMode')}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            output = process.stdout.read() if process.stdout else ""
            if process.returncode not in {0, 1, -15} and output:
                print(output, file=sys.stderr)


if __name__ == "__main__":
    main()
