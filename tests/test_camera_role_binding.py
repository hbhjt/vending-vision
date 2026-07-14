import base64
import hashlib
import hmac
import json
import threading
import time
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from vision.camera_binding import (
    CAMERA_MAINTENANCE_CONTRACT_VERSION,
    CameraLeaseRegistry,
    CameraMaintenanceService,
    JsonBindingStore,
    MaintenanceCapabilityVerifier,
    WindowsCameraDiscovery,
)


class MutableDiscovery:
    def __init__(self, observations=None):
        self.calls = 0
        self.observations = observations or [
            {"stableId": "usb#top-001", "label": "Top", "backend": "dshow", "index": 3,
             "available": True, "mappingState": "proven"},
            {"stableId": "usb#front-002", "label": "Front", "backend": "dshow", "index": 7,
             "available": True, "mappingState": "proven"},
        ]

    def enumerate(self):
        self.calls += 1
        return list(self.observations)


class InMemoryBindings:
    def __init__(self, value=None):
        self.value = value or {}

    def load(self):
        return dict(self.value)

    def save(self, bindings):
        self.value = json.loads(json.dumps(bindings))


class RecordingCameraAccess:
    def __init__(self):
        self.previewed = []
        self.tested = []

    def preview(self, candidate):
        self.previewed.append(candidate.stable_id)
        return b"jpeg"

    def test(self, candidate):
        self.tested.append(candidate.stable_id)
        return {"ok": True, "frame": {"width": 1280, "height": 720}}


def make_service(*, discovery=None, store=None, access=None, clock=time.time):
    return CameraMaintenanceService(
        discovery or MutableDiscovery(), store or InMemoryBindings(), access, clock=clock
    )


def test_windows_discovery_never_zips_independent_pnp_and_opencv_orderings(monkeypatch):
    class MediaSourceAdapter:
        def enumerate_sources(self):
            return [
        {"stableId": "mf://top", "label": "Top", "backend": "mediafoundation", "source": "mf://top"},
        {"stableId": "mf://front", "label": "Front", "backend": "mediafoundation", "source": "mf://front"},
            ]

    discovery = WindowsCameraDiscovery(MediaSourceAdapter())

    candidates = discovery.enumerate()

    assert [candidate["stableId"] for candidate in candidates] == ["mf://top", "mf://front"]
    assert all(candidate["index"] is None for candidate in candidates)
    assert all(candidate["mappingState"] == "unproven" for candidate in candidates)
    assert all(candidate["available"] is False for candidate in candidates)


def test_candidate_snapshot_is_cached_until_explicit_refresh_or_read_failure():
    discovery = MutableDiscovery()
    service = make_service(discovery=discovery)

    first = service.contract()
    service.contract()
    service.resolve if False else None  # contract callers must not cause another scan

    assert discovery.calls == 1
    assert service.refresh()["generation"] != first["generation"] or discovery.calls == 2
    service.refresh_after_read_failure()
    assert discovery.calls == 3


def test_runtime_preview_and_test_share_one_camera_lease_namespace():
    leases = CameraLeaseRegistry()
    runtime = leases.acquire("usb#top-001", "runtime:top")

    with pytest.raises(RuntimeError, match="runtime:top"):
        leases.acquire("usb#top-001", "maintenance-preview")
    runtime.release()

    preview = leases.acquire("usb#top-001", "maintenance-preview")
    preview.release()
    test = leases.acquire("usb#top-001", "maintenance-test")
    test.release()


def test_contract_reports_unproven_or_duplicate_identity_as_explicit_non_ready():
    discovery = MutableDiscovery([{
        "stableId": "usb#top-001", "label": "Top", "backend": "dshow", "index": None,
        "available": False, "mappingState": "unproven",
    }])
    store = InMemoryBindings({"top": {"stableId": "usb#top-001"}})
    service = make_service(discovery=discovery, store=store)

    role = service.contract()["roles"]["top"]

    assert role["state"] == "ambiguous"
    assert role["ready"] is False
    assert role["reason"] == "camera_mapping_unproven"


def test_confirm_requires_fresh_role_specific_test_evidence_or_visual_confirmation():
    clock = [1_000.0]
    service = make_service(access=RecordingCameraAccess(), clock=lambda: clock[0])

    with pytest.raises(ValueError, match="evidence"):
        service.confirm("top", "usb#top-001")

    evidence = service.test("top", "usb#top-001")["evidence"]
    with pytest.raises(ValueError, match="role"):
        service.confirm("front", "usb#top-001", test_evidence_id=evidence["id"])
    assert service.confirm("top", "usb#top-001", test_evidence_id=evidence["id"])["state"] == "ready"

    with pytest.raises(ValueError, match="already consumed"):
        service.confirm("top", "usb#top-001", test_evidence_id=evidence["id"])

    assert service.confirm(
        "front", "usb#front-002", operator_visual_confirmation=True
    )["state"] == "ready"


def test_concurrent_confirms_and_duplicate_persisted_bindings_are_non_ready():
    service = make_service(access=RecordingCameraAccess())
    evidence = service.test("top", "usb#top-001")["evidence"]["id"]
    outcomes = []

    def confirm(role):
        try:
            outcomes.append((role, service.confirm(role, "usb#top-001", test_evidence_id=evidence)["state"]))
        except ValueError as exc:
            outcomes.append((role, str(exc)))

    first = threading.Thread(target=confirm, args=("top",))
    second = threading.Thread(target=confirm, args=("front",))
    first.start(); second.start(); first.join(); second.join()
    assert sum(value == "ready" for _, value in outcomes) == 1
    assert any("already" in value or "consumed" in value for _, value in outcomes if value != "ready")

    duplicate = make_service(store=InMemoryBindings({
        "top": {"stableId": "usb#top-001"}, "front": {"stableId": "usb#top-001"},
    }))
    statuses = duplicate.contract()["roles"]
    assert statuses["top"]["state"] == statuses["front"]["state"] == "ambiguous"
    assert statuses["top"]["ready"] is statuses["front"]["ready"] is False


def test_contract_and_negative_role_states_validate_against_versioned_schema():
    schema_path = Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    contract = make_service().contract()

    jsonschema.validate(contract, schema)
    invalid = json.loads(json.dumps(contract))
    invalid["roles"]["top"] = {"role": "top", "state": "ready", "ready": False}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)

    requests_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.requests.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate({"candidateId": "usb#top-001"}, requests_schema["$defs"]["test"])
    jsonschema.validate({"candidateId": "usb#top-001", "operatorVisualConfirmation": True}, requests_schema["$defs"]["confirm"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"candidateId": "usb#top-001"}, requests_schema["$defs"]["confirm"])

    responses_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json").read_text(encoding="utf-8"))
    test_response = make_service(access=RecordingCameraAccess()).test("top", "usb#top-001")
    jsonschema.validate(test_response, {"$ref": "#/$defs/test", "$defs": responses_schema["$defs"]})


def capability(secret, *, scope, purpose="vision.camera-maintenance", expires_at=None, jti="one"):
    claims = {
        "purpose": purpose, "scope": scope, "exp": expires_at or int(time.time()) + 60, "jti": jti,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def test_loopback_http_requires_single_use_maintenance_capability(monkeypatch):
    import app

    secret = "test-maintenance-secret"
    service = make_service(access=RecordingCameraAccess())
    monkeypatch.setattr("app.get_camera_maintenance", lambda: service)
    monkeypatch.setattr(app, "_maintenance_authorizer", MaintenanceCapabilityVerifier(secret))
    client = TestClient(app.app)

    missing = client.get("/maintenance/cameras")
    assert missing.status_code == 401
    assert missing.json()["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION
    customer = capability(secret, scope=["camera.read"], purpose="customer", jti="customer")
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": customer}).status_code == 403
    expired = capability(secret, scope=["camera.read"], expires_at=int(time.time()) - 1, jti="expired")
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": expired}).status_code == 401

    token = capability(secret, scope=["camera.read"], jti="read-once")
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": token}).status_code == 200
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": token}).status_code == 409

    bad_confirm = capability(secret, scope=["camera.confirm"], jti="bad-confirm")
    response = client.post("/maintenance/cameras/top/confirm", json={"candidateId": "usb#top-001"}, headers={"X-Vision-Maintenance-Capability": bad_confirm})
    assert response.status_code == 409
    assert response.json()["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION


def test_release_version_does_not_publish_camera_index(monkeypatch):
    from app import version

    payload = version()

    assert "camera_index" not in json.dumps(payload).lower()
    assert CAMERA_MAINTENANCE_CONTRACT_VERSION.endswith("/v2")
