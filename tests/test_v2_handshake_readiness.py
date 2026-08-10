import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as vision_app
from vision import v2_contract_bundle
from vision.v2_contract_bundle import V2ContractBundleUnavailable


BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"


def test_debug_contract_bundle_exposes_the_generated_manifest_identity_only():
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text("utf-8"))

    response = TestClient(vision_app.app).get("/debug/contract-bundle")

    assert response.status_code == 200
    assert response.json() == {
        "protocol": manifest["protocol"],
        "schemaVersion": manifest["schemaVersion"],
        "bundleVersion": manifest["bundleVersion"],
        "contractDigest": manifest["bundleDigest"],
    }


def test_debug_websocket_accepts_the_same_strict_manifest_bound_v2_hello(monkeypatch):
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text("utf-8"))
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "client-valid.json").read_text("utf-8"))[0]
    hello["payload"]["schemaVersion"] = manifest["schemaVersion"]
    hello["payload"]["bundleVersion"] = manifest["bundleVersion"]
    hello["payload"]["contractDigest"] = manifest["bundleDigest"]
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )

    with TestClient(vision_app.app).websocket_connect("/debug/ws") as socket:
        socket.send_json(hello)
        ready = socket.receive_json()

    assert ready["type"] == "vision.ready"
    assert ready["protocol"] == manifest["protocol"]
    assert ready["payload"]["schemaVersion"] == manifest["schemaVersion"]
    assert ready["payload"]["bundleVersion"] == manifest["bundleVersion"]
    assert ready["payload"]["contractDigest"] == manifest["bundleDigest"]


@pytest.mark.parametrize("mutation", ["pretty", "duplicate", "digest"])
def test_v2_identity_rejects_noncanonical_duplicate_and_digest_tampering(
    tmp_path: Path, monkeypatch, mutation: str
):
    bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, bundle)
    manifest_path = bundle / "manifest.json"
    raw = manifest_path.read_text("utf-8")
    if mutation == "pretty":
        manifest_path.write_text(
            json.dumps(json.loads(raw), indent=2) + "\n", encoding="utf-8"
        )
    elif mutation == "duplicate":
        manifest_path.write_text(
            raw.replace(
                '"protocol":"vem.vision.v2",',
                '"protocol":"vem.vision.v2","protocol":"vem.vision.v2",',
            ),
            encoding="utf-8",
        )
    else:
        manifest = json.loads(raw)
        manifest["files"]["vision-v2.client.schema.json"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(v2_contract_bundle, "_BUNDLE_ROOT", bundle)

    with pytest.raises(V2ContractBundleUnavailable):
        v2_contract_bundle.load_v2_contract_identity()


def test_v2_digest_mismatch_withholds_fast_readiness_without_rejecting_core_connection(
    monkeypatch,
):
    """A contract mismatch is an enhancement diagnostic, not a core Vision outage."""
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text("utf-8"))
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "client-valid.json").read_text("utf-8"))[0]
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
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "client-valid.json").read_text("utf-8"))[0]
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


def test_missing_identity_returns_a_strict_degraded_ready_after_valid_hello(
    monkeypatch,
):
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "client-valid.json").read_text("utf-8"))[0]
    monkeypatch.setattr(
        vision_app,
        "load_v2_contract_identity",
        lambda: (_ for _ in ()).throw(V2ContractBundleUnavailable("missing")),
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )

    with TestClient(vision_app.app).websocket_connect("/ws") as socket:
        socket.send_json(hello)
        ready = socket.receive_json()

    assert ready["type"] == "vision.ready"
    assert ready["payload"]["businessReadinessDiagnostic"] == "contract_bundle_unavailable"
    assert ready["payload"]["fastReady"] is False
    assert ready["payload"]["visionBusinessReady"] is False


def test_missing_generated_parser_never_falls_back_to_hello_capabilities(monkeypatch):
    hello = json.loads((BUNDLE_ROOT / "fixtures" / "client-valid.json").read_text("utf-8"))[0]
    monkeypatch.setattr(
        vision_app,
        "parse_v2_client_message",
        lambda _value: (_ for _ in ()).throw(V2ContractBundleUnavailable("missing")),
    )
    monkeypatch.setattr(
        vision_app,
        "get_runtime_status",
        lambda: {"cameraReady": True, "modelReady": True},
    )

    with TestClient(vision_app.app).websocket_connect("/ws") as socket:
        socket.send_json(hello)
        error = socket.receive_json()

    assert error["type"] == "vision.error"
    assert error["payload"]["code"] == "invalid_message"
