import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import websockets


PROTOCOL = "vem.vision.v1"
DEFAULT_URL = "ws://127.0.0.1:7892/ws"


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def envelope(message_type, message_id, payload):
    return {
        "protocol": PROTOCOL,
        "type": message_type,
        "messageId": message_id,
        "timestamp": now_iso(),
        "payload": payload,
    }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def require_envelope(message, expected_type=None):
    require(isinstance(message, dict), "message must be object")
    require(message.get("protocol") == PROTOCOL, "protocol mismatch")
    require(isinstance(message.get("type"), str), "type missing")
    require(isinstance(message.get("messageId"), str), "messageId missing")
    require(isinstance(message.get("timestamp"), str), "timestamp missing")
    require(isinstance(message.get("payload"), dict), "payload missing")

    if expected_type:
        require(
            message["type"] == expected_type,
            f"expected {expected_type}, got {message['type']}",
        )


def require_profile(profile):
    required = [
        "personPresent",
        "heightCm",
        "shoulderWidthCm",
        "ageRange",
        "gender",
        "bodyType",
        "upperColor",
        "confidence",
    ]

    for key in required:
        require(key in profile, f"profile.{key} missing")

    require(isinstance(profile["personPresent"], bool), "personPresent must be bool")
    require(
        profile["ageRange"] in {"child", "teen", "adult", "senior", "unknown"},
        "invalid ageRange",
    )
    require(profile["gender"] in {"male", "female", "unknown"}, "invalid gender")
    require(
        profile["bodyType"] in {"slim", "regular", "strong", "unknown"},
        "invalid bodyType",
    )
    require(isinstance(profile["confidence"], (int, float)), "confidence must be number")
    require(0 <= profile["confidence"] <= 1, "confidence out of range")


async def recv_json(websocket, timeout=None, transcript=None):
    raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
    message = json.loads(raw)
    if transcript is not None:
        transcript.append(
            {
                "direction": "recv",
                "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "message": message,
            }
        )
    print(json.dumps(message, ensure_ascii=False, indent=2))
    return message


async def send_json(websocket, message, transcript):
    await websocket.send(json.dumps(message, ensure_ascii=False))
    transcript.append(
        {
            "direction": "send",
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "message": message,
        }
    )


async def run(url, wait_seconds, transcript):
    status = "unknown"

    async with websockets.connect(url) as websocket:
        hello = envelope(
            "vision.hello",
            "hello-001",
            {
                "clientRole": "machine",
                "machineCode": "M001",
                "protocolVersion": 1,
                "capabilities": ["profile_push"],
            },
        )

        await send_json(websocket, hello, transcript)
        ready = await recv_json(websocket, timeout=5, transcript=transcript)
        require_envelope(ready, "vision.ready")
        payload = ready["payload"]
        require("cameraReady" in payload, "ready.cameraReady missing")
        require("modelReady" in payload, "ready.modelReady missing")
        require("profile_push" in payload.get("capabilities", []), "profile_push missing")

        ping = envelope("vision.ping", "ping-001", {})
        await send_json(websocket, ping, transcript)
        pong = await recv_json(websocket, timeout=5, transcript=transcript)
        require_envelope(pong, "vision.pong")

        print(f"waiting for active profile push, timeout={wait_seconds}s")

        while True:
            message = await recv_json(
                websocket,
                timeout=wait_seconds,
                transcript=transcript,
            )
            require_envelope(message)

            if message["type"] == "vision.profile_result":
                payload = message["payload"]
                require(isinstance(payload.get("eventId"), str), "eventId missing")
                require(isinstance(payload.get("detectedAt"), str), "detectedAt missing")
                require_profile(payload.get("profile", {}))
                require(isinstance(payload.get("quality"), dict), "quality missing")
                print("PUSH WS TEST PASSED")
                status = "profile_result"
                return status

            if message["type"] == "vision.error":
                payload = message["payload"]
                require(isinstance(payload.get("code"), str), "error code missing")
                require(isinstance(payload.get("retryable"), bool), "retryable missing")
                print("PUSH WS TEST PASSED WITH ERROR")
                status = "vision_error"
                return status

            raise AssertionError(f"unexpected message type: {message['type']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Push WebSocket protocol client")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default="test_reports/ws")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ws_{timestamp}.json"

    try:
        transcript = []
        status = asyncio.run(run(args.url, args.wait_seconds, transcript))
    except Exception as e:
        transcript = locals().get("transcript", [])
        status = "failed"
        error = str(e)
        raise
    finally:
        result = {
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "url": args.url,
            "waitSeconds": args.wait_seconds,
            "status": locals().get("status", "failed"),
            "error": locals().get("error", ""),
            "messages": locals().get("transcript", []),
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved: {output_path}")
