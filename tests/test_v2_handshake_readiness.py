import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as vision_app


BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"


def test_v2_digest_mismatch_withholds_fast_readiness_without_rejecting_core_connection(
    monkeypatch,
):
    """A contract mismatch is an enhancement diagnostic, not a core Vision outage."""
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text("utf-8"))
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))[0]
    hello["payload"]["bundleVersion"] = manifest["bundleVersion"]
    hello["payload"]["schemaVersion"] = manifest["schemaVersion"]
    hello["payload"]["contractDigest"] = "b" * 64

    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )
    client = TestClient(vision_app.app)
    with client.websocket_connect("/ws") as socket:
        socket.send_json(hello)
        ready = socket.receive_json()

    assert ready["protocol"] == "vem.vision.v2"
    assert ready["type"] == "vision.ready"
    assert ready["payload"]["contractDigest"] == manifest["bundleDigest"]
    assert ready["payload"]["bundleVersion"] == manifest["bundleVersion"]
    assert ready["payload"]["schemaVersion"] == manifest["schemaVersion"]
    assert ready["payload"]["visionBusinessReady"] is False
    assert ready["payload"]["fastReady"] is False
    assert ready["payload"]["cameraReady"] is True


def test_v2_handshake_rejects_unknown_envelope_properties_through_generated_boundary(
    monkeypatch,
):
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))[0]
    hello["unexpected"] = True
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )

    client = TestClient(vision_app.app)
    with client.websocket_connect("/ws") as socket:
        socket.send_json(hello)
        rejected = socket.receive_json()

    assert rejected["protocol"] == "vem.vision.v2"
    assert rejected["type"] == "vision.error"
    assert rejected["payload"]["code"] == "invalid_message"
    assert rejected["payload"]["message"] == "invalid_v2_boundary_message"
