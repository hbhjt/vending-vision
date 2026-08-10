"""Bounded physical Vision core capability probe (no customer generation path)."""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websockets


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


async def run(args: argparse.Namespace) -> None:
    health = read_json(f"{args.http_base}/health")
    if health.get("mockScenario") != "off":
        raise AssertionError("real-camera acceptance forbids mock mode")
    if not health.get("modelReady") or not health.get("cameraReady"):
        raise AssertionError(f"model/camera not ready: {health}")

    evidence = {
        "schemaVersion": "vending-vision-capability-acceptance/v2",
        "kind": "vision-core-capability-acceptance",
        "observedAt": now_iso(),
        "mockScenario": health.get("mockScenario"),
        "modelReady": health.get("modelReady"),
        "cameraReady": health.get("cameraReady"),
        "presenceStatus": False,
        "singleUsableProfile": False,
        "personDeparted": False,
    }
    async with websockets.connect(args.ws_url, origin="http://tauri.localhost") as socket:
        manifest = read_json(f"{args.http_base}/debug/contract-bundle")
        await socket.send(json.dumps({
            "protocol": "vem.vision.v2",
            "type": "vision.hello",
            "messageId": "physical-capability-probe",
            "timestamp": now_iso(),
            "payload": {
                "clientRole": "machine",
                "machineCode": args.machine_code,
                "schemaVersion": manifest["schemaVersion"],
                "bundleVersion": manifest["bundleVersion"],
                "contractDigest": manifest["contractDigest"],
                "capabilities": ["profile_push", "presence_status", "person_departed"],
            },
        }))
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
            elif event_type == "vision.person_departed":
                evidence["personDeparted"] = True
            if all(evidence[key] for key in ("presenceStatus", "singleUsableProfile", "personDeparted")):
                break
    if not all(evidence[key] for key in ("presenceStatus", "singleUsableProfile", "personDeparted")):
        raise AssertionError(f"capability acceptance incomplete: {evidence}")
    Path(args.evidence).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http-base", required=True)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--machine-code", default="physical-capability-probe")
    parser.add_argument("--observation-timeout", type=float, default=120)
    parser.add_argument("--evidence", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
