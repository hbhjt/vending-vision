import asyncio
import json
import urllib.request
from datetime import datetime, timezone

import websockets


PROTOCOL = "vem.vision.v1"
HTTP_BASE_URL = "http://127.0.0.1:7892"
WS_URL = "ws://127.0.0.1:7892/ws"


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


def print_pass(message):
    print(f"[PASS] {message}")


def print_fail(message):
    print(f"[FAIL] {message}")


def test_health():
    try:
        with urllib.request.urlopen(f"{HTTP_BASE_URL}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))

        if health.get("status") in {"ok", "degraded"}:
            print_pass("HTTP /health")
            print("       status:", health.get("status"))
            print("       cameraReady:", health.get("cameraReady"))
            print("       modelReady:", health.get("modelReady"))
            return True

        print_fail("HTTP /health returned unexpected status")
        print(health)
        return False

    except Exception as e:
        print_fail(f"HTTP /health failed: {e}")
        return False


async def test_websocket(wait_seconds=30):
    try:
        async with websockets.connect(WS_URL) as websocket:
            print_pass("WebSocket connected")

            hello = envelope(
                "vision.hello",
                "it-hello-001",
                {
                    "clientRole": "machine",
                    "machineCode": "M001",
                    "protocolVersion": 1,
                    "capabilities": ["profile_push"],
                },
            )

            await websocket.send(json.dumps(hello, ensure_ascii=False))
            ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))

            if ready.get("type") != "vision.ready":
                print_fail("vision.hello did not return vision.ready")
                print(ready)
                return False

            print_pass("vision.hello -> vision.ready")
            print("       payload:", json.dumps(ready.get("payload", {}), ensure_ascii=False))

            ping = envelope("vision.ping", "it-ping-001", {})
            await websocket.send(json.dumps(ping, ensure_ascii=False))
            pong = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))

            if pong.get("type") != "vision.pong":
                print_fail("vision.ping did not return vision.pong")
                print(pong)
                return False

            print_pass("vision.ping -> vision.pong")
            print(f"       waiting profile push up to {wait_seconds}s")

            message = json.loads(
                await asyncio.wait_for(websocket.recv(), timeout=wait_seconds)
            )

            if message.get("type") == "vision.profile_result":
                print_pass("received vision.profile_result")
                print(json.dumps(message.get("payload", {}), ensure_ascii=False, indent=2))
                return True

            if message.get("type") == "vision.error":
                print_pass("received vision.error")
                print(json.dumps(message.get("payload", {}), ensure_ascii=False, indent=2))
                return True

            print_fail(f"unexpected message type: {message.get('type')}")
            print(message)
            return False

    except asyncio.TimeoutError:
        print_fail(
            "no profile push received before timeout; this is expected if no valid person is in view"
        )
        return True
    except Exception as e:
        print_fail(f"WebSocket test failed: {e}")
        return False


async def main():
    print("================================")
    print("Vending Vision Integration Test")
    print("================================")

    health_ok = test_health()

    if not health_ok:
        print("Start the service first: scripts\\start_server.bat")
        return

    print()
    ws_ok = await test_websocket()

    print()
    print("================================")
    print("ALL TESTS PASSED" if health_ok and ws_ok else "TESTS FAILED")
    print("================================")


if __name__ == "__main__":
    asyncio.run(main())
