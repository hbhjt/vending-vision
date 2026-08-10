from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import websockets


PROTOCOL = "vem.vision.v2"
CONTRACT_ROOT = Path(__file__).resolve().parents[1] / "contracts" / "vem_vision_v2"


def v2_handshake_identity():
    manifest = json.loads((CONTRACT_ROOT / "manifest.json").read_text("utf-8"))
    return {
        "schemaVersion": manifest["schemaVersion"],
        "bundleVersion": manifest["bundleVersion"],
        "contractDigest": manifest["bundleDigest"],
    }


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def message(message_type, payload):
    return {
        "protocol": PROTOCOL,
        "type": message_type,
        "messageId": f"acceptance-{uuid4()}",
        "timestamp": now_iso(),
        "payload": payload,
    }


def read_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


async def run(args):
    health = read_json(f"{args.http_base}/health")
    if health.get("mockScenario") != "off":
        raise AssertionError("real-camera acceptance forbids mock mode")
    if not health.get("modelReady") or not health.get("cameraReady"):
        raise AssertionError(f"model/camera not ready: {health}")

    evidence = {
        "schemaVersion": "vending-vision-capability-acceptance/v1",
        "kind": "vision-capability-acceptance",
        "observedAt": now_iso(),
        "mockScenario": health.get("mockScenario"),
        "modelReady": health.get("modelReady"),
        "cameraReady": health.get("cameraReady"),
        "presenceStatus": False,
        "singleUsableProfile": False,
        "personDeparted": False,
        "tryOnStarted": False,
        "tryOnStopped": False,
    }
    async with websockets.connect(args.ws_url, origin="http://tauri.localhost") as socket:
        await socket.send(json.dumps(message("vision.hello", {
            "clientRole": "machine",
            "machineCode": args.machine_code,
            **v2_handshake_identity(),
            "capabilities": ["profile_push", "presence_status", "person_departed", "try_on_fast"],
        })))
        ready = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
        if ready.get("type") != "vision.ready" or not ready.get("payload", {}).get("cameraReady"):
            raise AssertionError(f"machine handshake failed: {ready}")

        deadline = asyncio.get_running_loop().time() + args.observation_timeout
        while asyncio.get_running_loop().time() < deadline:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=15))
            event_type = event.get("type")
            payload = event.get("payload", {})
            if event_type == "vision.presence_status" and payload.get("personPresent"):
                evidence["presenceStatus"] = True
            elif event_type == "vision.profile_result":
                evidence["singleUsableProfile"] = (
                    payload.get("occupancy", {}).get("state") == "single"
                    and payload.get("quality", {}).get("profileUsable") is True
                )
                if evidence["singleUsableProfile"] and not evidence["tryOnStarted"]:
                    await socket.send(json.dumps(message("vision.try_on.start", {
                        "sessionId": f"field-{uuid4()}",
                        "catalogKey": "field-acceptance",
                        "variantId": "field-acceptance",
                    })))
            elif event_type == "vision.try_on.started":
                evidence["tryOnStarted"] = True
                session_id = payload["sessionId"]
                with urllib.request.urlopen(payload["previewUrl"], timeout=10) as response:
                    if b"multipart" not in response.headers.get("Content-Type", "").encode():
                        raise AssertionError("try-on preview is not MJPEG multipart")
                    if not response.read(256):
                        raise AssertionError("try-on preview produced no frame bytes")
                await socket.send(json.dumps(message("vision.try_on.stop", {
                    "sessionId": session_id,
                    "reason": "user_exit",
                })))
            elif event_type == "vision.try_on.stopped":
                evidence["tryOnStopped"] = payload.get("reason") == "client_stop"
            elif event_type == "vision.person_departed":
                evidence["personDeparted"] = True
            if all(evidence[key] for key in (
                "presenceStatus", "singleUsableProfile", "personDeparted", "tryOnStarted", "tryOnStopped"
            )):
                break

    if not all(evidence[key] for key in (
        "presenceStatus", "singleUsableProfile", "personDeparted", "tryOnStarted", "tryOnStopped"
    )):
        raise AssertionError(f"capability acceptance incomplete: {evidence}")
    Path(args.evidence).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="VEM 视觉真实摄像头能力验收")
    parser.add_argument("--http-base", default="http://127.0.0.1:7892")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:7892/ws")
    parser.add_argument("--machine-code", default="VEM-WIN10-REAL-01")
    parser.add_argument("--observation-timeout", type=float, default=180.0)
    parser.add_argument("--evidence", default="vision-capability-acceptance.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
