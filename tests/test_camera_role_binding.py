import base64
import json
import threading
import time
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from vision.camera_binding import (
    CAMERA_MAINTENANCE_CONTRACT_VERSION,
    Cv2EnumerateCamerasDirectShowAdapter,
    CameraLeaseRegistry,
    CameraMaintenanceService,
    DurableReplayStore,
    JsonBindingStore,
    MaintenanceCapabilityError,
    MaintenanceCapabilityVerifier,
    OpenCvCameraAccess,
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

    def test(self, role, candidate):
        self.tested.append((role, candidate.stable_id))
        return {"ok": True, "frame": {"width": 1280, "height": 720}}


def make_service(*, discovery=None, store=None, access=None, clock=time.time):
    return CameraMaintenanceService(
        discovery or MutableDiscovery(), store or InMemoryBindings(), access, clock=clock
    )


def test_windows_discovery_never_zips_independent_pnp_and_opencv_orderings(monkeypatch):
    class MediaSourceAdapter:
        def enumerate_sources(self):
            return [
        {"stableId": "unmapped://top", "label": "Top", "backend": "unmapped", "source": "unmapped://top"},
        {"stableId": "unmapped://front", "label": "Front", "backend": "unmapped", "source": "unmapped://front"},
            ]

    discovery = WindowsCameraDiscovery(MediaSourceAdapter())

    candidates = discovery.enumerate()

    assert [candidate["stableId"] for candidate in candidates] == ["unmapped://top", "unmapped://front"]
    assert all(candidate["index"] is None for candidate in candidates)
    assert all(candidate["mappingState"] == "unproven" for candidate in candidates)
    assert all(candidate["available"] is False for candidate in candidates)


def test_production_windows_adapter_enumerates_stable_directshow_path_and_openable_index():
    """One pinned DirectShow boundary supplies both binding identity and capture index."""
    calls = []

    class CameraInfo:
        index = 4
        name = "Top USB Camera"
        path = r"@device:pnp:\\?\usb#vid_1111&pid_0001#top"
        backend = 700

    def enumerate_cameras(backend):
        calls.append(backend)
        return [CameraInfo()]

    adapter = Cv2EnumerateCamerasDirectShowAdapter(
        enumerate_cameras=enumerate_cameras,
        dshow_backend=700,
    )

    assert adapter.enumerate_sources() == [{
        "stableId": r"@device:pnp:\\?\usb#vid_1111&pid_0001#top",
        "label": "Top USB Camera",
        "backend": "dshow",
        "index": 4,
        "available": True,
        "mappingState": "proven",
    }]
    assert calls == [700]


def test_two_directshow_cameras_keep_confirmed_roles_after_replug_changes_indexes():
    discovery = MutableDiscovery([
        {"stableId": "@device:pnp:top", "label": "Top", "backend": "dshow", "index": 1,
         "available": True, "mappingState": "proven"},
        {"stableId": "@device:pnp:front", "label": "Front", "backend": "dshow", "index": 4,
         "available": True, "mappingState": "proven"},
    ])
    service = make_service(discovery=discovery, access=RecordingCameraAccess())
    top = service.test("top", "@device:pnp:top")["evidence"]
    front = service.test("front", "@device:pnp:front")["evidence"]
    service.confirm("top", "@device:pnp:top", test_evidence_id=top["id"],
                    operator_visual_confirmation=True, expected_generation=top["generation"])
    service.confirm("front", "@device:pnp:front", test_evidence_id=front["id"],
                    operator_visual_confirmation=True, expected_generation=front["generation"])

    discovery.observations = [
        {"stableId": "@device:pnp:front", "label": "Front", "backend": "dshow", "index": 0,
         "available": True, "mappingState": "proven"},
        {"stableId": "@device:pnp:top", "label": "Top", "backend": "dshow", "index": 8,
         "available": True, "mappingState": "proven"},
    ]
    service.refresh()

    assert service.resolve("top").stable_id == "@device:pnp:top"
    assert service.resolve("top").index == 8
    assert service.resolve("front").stable_id == "@device:pnp:front"
    assert service.resolve("front").index == 0


def test_daemon_issued_ed25519_capability_binds_machine_session_and_survives_verifier_restart(tmp_path):
    """Vision verifies the daemon's public key; it never has an issuer secret."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    keyring = tmp_path / "daemon-maintenance-keys.json"
    session = tmp_path / "daemon-maintenance-session.json"
    replay = tmp_path / "camera-maintenance-replay.json"
    keyring.write_text(json.dumps({
        "version": 1,
        "issuer": "vem.vending-daemon",
        "keys": [{
            "id": "daemon-ed25519-202607",
            "publicKey": base64.urlsafe_b64encode(public_key).decode().rstrip("="),
            "notBefore": 900,
            "notAfter": 10_000,
        }],
    }), encoding="utf-8")
    session.write_text(json.dumps({
        "version": 1,
        "machineCode": "VEM-TESTBED-01",
        "sessionId": "maintenance-session-01",
        "keyId": "daemon-ed25519-202607",
        "expiresAt": 2_000,
    }), encoding="utf-8")

    claims = {
        "iss": "vem.vending-daemon",
        "aud": "vem.vision.camera-maintenance",
        "machine": "VEM-TESTBED-01",
        "session": "maintenance-session-01",
        "purpose": "vision.camera-maintenance",
        "scope": ["camera.read"],
        "iat": 1_000,
        "exp": 1_120,
        "jti": "maintenance-read-once",
    }
    header = {"alg": "EdDSA", "kid": "daemon-ed25519-202607", "typ": "JWT"}
    encoded_header = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode()).decode().rstrip("=")
    encoded_claims = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).decode().rstrip("=")
    signed = f"{encoded_header}.{encoded_claims}".encode()
    token = f"{signed.decode()}.{base64.urlsafe_b64encode(private_key.sign(signed)).decode().rstrip('=')}"

    verifier = MaintenanceCapabilityVerifier(
        keyring, session, DurableReplayStore(replay), clock=lambda: 1_010
    )
    assert verifier.verify(token, "camera.read") == claims
    with pytest.raises(MaintenanceCapabilityError) as replayed:
        verifier.verify(token, "camera.read")
    assert replayed.value.status_code == 409

    restarted = MaintenanceCapabilityVerifier(
        keyring, session, DurableReplayStore(replay), clock=lambda: 1_010
    )
    with pytest.raises(MaintenanceCapabilityError) as persisted_replay:
        restarted.verify(token, "camera.read")
    assert persisted_replay.value.status_code == 409


def test_missing_daemon_public_key_and_session_material_is_an_explicit_maintenance_blocker(tmp_path):
    verifier = MaintenanceCapabilityVerifier(None, None, DurableReplayStore(tmp_path / "replay.json"))

    with pytest.raises(MaintenanceCapabilityError) as blocked:
        verifier.verify(None, "camera.read")

    assert blocked.value.status_code == 503
    assert "blocked" in str(blocked.value)


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


def test_runtime_camera_handoff_allows_maintenance_test_confirmation_then_resumes(monkeypatch):
    """A persistent runtime owner yields the camera, then is allowed to reopen."""
    leases = CameraLeaseRegistry()
    runtime = leases.acquire("usb#top-001", "runtime:top")
    events = []

    class Handoff:
        def release(self):
            events.append("resume")

    def handoff(candidate_id):
        assert candidate_id == "usb#top-001"
        events.append("quiesce")
        runtime.release()
        return Handoff()

    class Capture:
        def release(self):
            events.append("capture-release")

    access = OpenCvCameraAccess(leases, runtime_handoff=handoff)
    monkeypatch.setattr(access, "_open", lambda candidate: Capture())
    monkeypatch.setattr("vision.camera.read_warmup_frame", lambda capture, _: np.zeros((720, 1280, 3), dtype=np.uint8))
    service = make_service(access=access)

    evidence = service.test("top", "usb#top-001")["evidence"]
    confirmed = service.confirm(
        "top", "usb#top-001", test_evidence_id=evidence["id"],
        operator_visual_confirmation=True, expected_generation=evidence["generation"],
    )

    assert confirmed["state"] == "ready"
    assert events == ["quiesce", "capture-release", "resume"]
    resumed_runtime = leases.acquire("usb#top-001", "runtime:top")
    resumed_runtime.release()


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


def test_confirm_requires_matching_role_test_visual_confirmation_and_atomic_generation():
    clock = [1_000.0]
    access = RecordingCameraAccess()
    service = make_service(access=access, clock=lambda: clock[0])
    generation = service.contract()["generation"]

    with pytest.raises(ValueError, match="evidence"):
        service.confirm("top", "usb#top-001", operator_visual_confirmation=True,
                        expected_generation=generation)

    evidence = service.test("top", "usb#top-001")["evidence"]
    assert access.tested == [("top", "usb#top-001")]
    with pytest.raises(ValueError, match="role"):
        service.confirm("front", "usb#top-001", test_evidence_id=evidence["id"],
                        operator_visual_confirmation=True, expected_generation=evidence["generation"])
    with pytest.raises(ValueError, match="visual"):
        service.confirm("top", "usb#top-001", test_evidence_id=evidence["id"],
                        expected_generation=evidence["generation"])
    with pytest.raises(ValueError, match="generation"):
        service.confirm("top", "usb#top-001", test_evidence_id=evidence["id"],
                        operator_visual_confirmation=True, expected_generation="stale-generation")
    assert service.confirm(
        "top", "usb#top-001", test_evidence_id=evidence["id"],
        operator_visual_confirmation=True, expected_generation=evidence["generation"],
    )["state"] == "ready"

    with pytest.raises(ValueError, match="already consumed"):
        service.confirm("top", "usb#top-001", test_evidence_id=evidence["id"],
                        operator_visual_confirmation=True, expected_generation=evidence["generation"])

    front_evidence = service.test("front", "usb#front-002")["evidence"]
    assert service.confirm("front", "usb#front-002", test_evidence_id=front_evidence["id"],
                           operator_visual_confirmation=True,
                           expected_generation=front_evidence["generation"])["state"] == "ready"


def test_concurrent_confirms_and_duplicate_persisted_bindings_are_non_ready():
    service = make_service(access=RecordingCameraAccess())
    top_evidence = service.test("top", "usb#top-001")["evidence"]
    front_evidence = service.test("front", "usb#top-001")["evidence"]
    outcomes = []

    def confirm(role, evidence):
        try:
            outcomes.append((role, service.confirm(
                role, "usb#top-001", test_evidence_id=evidence["id"],
                operator_visual_confirmation=True, expected_generation=evidence["generation"],
            )["state"]))
        except ValueError as exc:
            outcomes.append((role, str(exc)))

    first = threading.Thread(target=confirm, args=("top", top_evidence))
    second = threading.Thread(target=confirm, args=("front", front_evidence))
    first.start(); second.start(); first.join(); second.join()
    assert sum(value == "ready" for _, value in outcomes) == 1
    assert any("already" in value or "confirmed" in value for _, value in outcomes if value != "ready")

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
    role_mismatch = json.loads(json.dumps(contract))
    role_mismatch["roles"]["top"] = {"role": "front", "state": "unbound", "ready": False,
                                         "reason": "camera_not_confirmed"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(role_mismatch, schema)

    requests_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.requests.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate({"candidateId": "usb#top-001"}, requests_schema["$defs"]["test"])
    jsonschema.validate({
        "candidateId": "usb#top-001", "testEvidenceId": "evidence-1",
        "operatorVisualConfirmation": True, "expectedGeneration": contract["generation"],
    }, requests_schema["$defs"]["confirm"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"candidateId": "usb#top-001", "testEvidenceId": "evidence-1"}, requests_schema["$defs"]["confirm"])

    responses_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json").read_text(encoding="utf-8"))
    contract_response = make_service().contract()
    jsonschema.validate(contract_response, {"$ref": "#/$defs/contract", "$defs": responses_schema["$defs"]})
    jsonschema.validate(contract_response, {"$ref": "#/$defs/refresh", "$defs": responses_schema["$defs"]})
    response_service = make_service(access=RecordingCameraAccess())
    test_response = response_service.test("top", "usb#top-001")
    jsonschema.validate(test_response, {"$ref": "#/$defs/test", "$defs": responses_schema["$defs"]})
    confirmed = response_service.confirm(
        "top", "usb#top-001", test_evidence_id=test_response["evidence"]["id"],
        operator_visual_confirmation=True, expected_generation=test_response["generation"],
    )
    jsonschema.validate(confirmed, {"$ref": "#/$defs/confirm", "$defs": responses_schema["$defs"]})


def capability(private_key, *, scope, purpose="vision.camera-maintenance", expires_at=None, jti="one", **overrides):
    claims = {
        "iss": "vem.vending-daemon",
        "aud": "vem.vision.camera-maintenance",
        "machine": "VEM-TESTBED-01",
        "session": "maintenance-session-01",
        "purpose": purpose,
        "scope": scope,
        "iat": 1_000,
        "exp": expires_at or 1_120,
        "jti": jti,
    }
    claims.update(overrides)
    header = {"alg": "EdDSA", "kid": "daemon-ed25519-202607", "typ": "JWT"}
    encoded_header = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode()).decode().rstrip("=")
    encoded_claims = base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode()).decode().rstrip("=")
    signed = f"{encoded_header}.{encoded_claims}".encode()
    return f"{signed.decode()}.{base64.urlsafe_b64encode(private_key.sign(signed)).decode().rstrip('=')}"


def daemon_authorizer(tmp_path, *, clock=lambda: 1_010):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    keyring = tmp_path / "daemon-maintenance-keys.json"
    session = tmp_path / "daemon-maintenance-session.json"
    keyring.write_text(json.dumps({
        "version": 1,
        "issuer": "vem.vending-daemon",
        "keys": [{
            "id": "daemon-ed25519-202607",
            "publicKey": base64.urlsafe_b64encode(public_key).decode().rstrip("="),
            "notBefore": 900,
            "notAfter": 10_000,
        }],
    }), encoding="utf-8")
    session.write_text(json.dumps({
        "version": 1,
        "machineCode": "VEM-TESTBED-01",
        "sessionId": "maintenance-session-01",
        "keyId": "daemon-ed25519-202607",
        "expiresAt": 2_000,
    }), encoding="utf-8")
    return private_key, MaintenanceCapabilityVerifier(
        keyring, session, DurableReplayStore(tmp_path / "camera-maintenance-replay.json"), clock=clock
    )


@pytest.mark.parametrize(
    ("overrides", "status_code"),
    [
        ({"aud": "another-audience"}, 403),
        ({"machine": "VEM-OTHER"}, 403),
        ({"session": "another-session"}, 403),
        ({"iat": 1_041, "exp": 1_120}, 401),
        ({"exp": 1_301}, 401),
    ],
)
def test_capability_rejects_wrong_audience_machine_session_and_lifetime(tmp_path, overrides, status_code):
    private_key, verifier = daemon_authorizer(tmp_path)
    token = capability(private_key, scope=["camera.read"], jti=f"invalid-{status_code}-{overrides}", **overrides)

    with pytest.raises(MaintenanceCapabilityError) as rejected:
        verifier.verify(token, "camera.read")

    assert rejected.value.status_code == status_code


def test_loopback_http_requires_single_use_maintenance_capability(monkeypatch, tmp_path):
    import app

    private_key, authorizer = daemon_authorizer(tmp_path)
    service = make_service(access=RecordingCameraAccess())
    monkeypatch.setattr("app.get_camera_maintenance", lambda: service)
    monkeypatch.setattr(app, "_maintenance_authorizer", authorizer)
    client = TestClient(app.app)
    responses_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json").read_text(encoding="utf-8"))

    missing = client.get("/maintenance/cameras")
    assert missing.status_code == 401
    assert missing.json()["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION
    customer = capability(private_key, scope=["camera.read"], purpose="customer", jti="customer")
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": customer}).status_code == 403
    expired = capability(private_key, scope=["camera.read"], expires_at=1_009, jti="expired")
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": expired}).status_code == 401

    token = capability(private_key, scope=["camera.read"], jti="read-once")
    listed = client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": token})
    assert listed.status_code == 200
    jsonschema.validate(listed.json(), {"$ref": "#/$defs/contract", "$defs": responses_schema["$defs"]})
    assert client.get("/maintenance/cameras", headers={"X-Vision-Maintenance-Capability": token}).status_code == 409

    refresh_token = capability(private_key, scope=["camera.refresh"], jti="refresh-once")
    refreshed = client.post("/maintenance/cameras/refresh", headers={"X-Vision-Maintenance-Capability": refresh_token})
    assert refreshed.status_code == 200
    jsonschema.validate(refreshed.json(), {"$ref": "#/$defs/refresh", "$defs": responses_schema["$defs"]})

    bad_confirm = capability(private_key, scope=["camera.confirm"], jti="bad-confirm")
    response = client.post("/maintenance/cameras/top/confirm", json={"candidateId": "usb#top-001"}, headers={"X-Vision-Maintenance-Capability": bad_confirm})
    assert response.status_code == 409
    assert response.json()["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION

    test_token = capability(private_key, scope=["camera.test"], jti="role-test")
    tested = client.post(
        "/maintenance/cameras/top/test", json={"candidateId": "usb#top-001"},
        headers={"X-Vision-Maintenance-Capability": test_token},
    )
    assert tested.status_code == 200
    jsonschema.validate(tested.json(), {"$ref": "#/$defs/test", "$defs": responses_schema["$defs"]})
    evidence = tested.json()["evidence"]
    confirm_token = capability(private_key, scope=["camera.confirm"], jti="role-confirm")
    confirmed = client.post(
        "/maintenance/cameras/top/confirm",
        json={
            "candidateId": "usb#top-001", "testEvidenceId": evidence["id"],
            "operatorVisualConfirmation": True, "expectedGeneration": evidence["generation"],
        },
        headers={"X-Vision-Maintenance-Capability": confirm_token},
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["state"] == "ready"
    jsonschema.validate(confirmed.json(), {"$ref": "#/$defs/confirm", "$defs": responses_schema["$defs"]})


def test_production_removes_legacy_snapshot_bypass_and_hides_development_dashboard(monkeypatch):
    import app

    monkeypatch.setattr(app.settings, "DEVELOPMENT_DASHBOARD_ENABLED", False)
    client = TestClient(app.app)

    assert client.get("/camera/top/snapshot.jpg").status_code == 404
    assert client.get("/dashboard").status_code == 404
    blocked = client.get("/maintenance/cameras")
    assert blocked.status_code == 503
    assert "blocked" in blocked.json()["error"]["message"]


def test_release_version_does_not_publish_camera_index(monkeypatch):
    from app import version

    payload = version()

    assert "camera_index" not in json.dumps(payload).lower()
    assert CAMERA_MAINTENANCE_CONTRACT_VERSION.endswith("/v2")
