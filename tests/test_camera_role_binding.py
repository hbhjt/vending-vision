import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision.camera_binding import (
    CAMERA_MAINTENANCE_CONTRACT_VERSION,
    Cv2EnumerateCamerasDirectShowAdapter,
    CameraCandidate,
    CameraLeaseRegistry,
    CameraMaintenanceService,
    JsonBindingStore,
    OpenCvCameraAccess,
    WindowsCameraDiscovery,
)
from vision.directshow_broker import DirectShowCameraBroker


def _maintenance_handoff_broker_target(connection, config):
    active = config["activeCounter"]
    starts = config["startCounter"]
    with starts.get_lock():
        starts.value += 1
    with active.get_lock():
        active.value += 1
    try:
        if config.get("_brokerReadyHandshake", False):
            connection.send(("ready", {"pid": os.getpid()}))
        while True:
            command, _payload = connection.recv()
            if command == "shutdown":
                with active.get_lock():
                    active.value -= 1
                connection.send(("ok", None))
                return
            if command == "read":
                connection.send(("ok", {"pid": os.getpid(), "image": np.zeros((8, 10, 3), dtype=np.uint8)}))
    finally:
        with active.get_lock():
            active.value = 0
        connection.close()


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


def test_startup_camera_check_refreshes_unobserved_generation_and_reports_roles(monkeypatch):
    from vision import self_check

    discovery = MutableDiscovery()
    service = make_service(
        discovery=discovery,
        store=InMemoryBindings({
            "top": {"stableId": "usb#top-001"},
            "front": {"stableId": "usb#front-002"},
        }),
    )

    monkeypatch.setattr(self_check.settings, "MOCK_SCENARIO", "off")
    monkeypatch.setattr(self_check, "get_camera_maintenance", lambda: service)

    result = self_check.check_camera()

    assert discovery.calls == 1
    assert result["ok"] is True
    assert result["detail"]["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION
    assert result["detail"]["generation"] != "unobserved"
    assert result["detail"]["roles"]["top"]["ready"] is True
    assert result["detail"]["roles"]["front"]["ready"] is True


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


def test_active_directshow_broker_is_dead_before_parent_maintenance_open_and_restarts_lazily(monkeypatch):
    from vision import camera_manager

    context = multiprocessing.get_context("spawn")
    active = context.Value("i", 0)
    starts = context.Value("i", 0)
    config = {
        "role": "profile_try_on",
        "index": 7,
        "backend": "dshow",
        "stableId": "usb#front-002",
        "keep_open": True,
        "_brokerReadyHandshake": True,
        "activeCounter": active,
        "startCounter": starts,
    }
    broker = DirectShowCameraBroker(
        "front", config, context=context, target=_maintenance_handoff_broker_target
    )
    source = camera_manager.DirectShowFrameSource("front", config)
    with camera_manager._streams_lock:
        camera_manager._dshow_brokers["front"] = broker

    class Capture:
        def isOpened(self):
            return True

        def release(self):
            assert active.value == 0

    access = OpenCvCameraAccess(runtime_handoff=camera_manager.quiesce_runtime_camera)
    monkeypatch.setattr(access, "_open", lambda _candidate: Capture() if active.value == 0 else (_ for _ in ()).throw(AssertionError("double open")))
    monkeypatch.setattr("vision.camera.read_warmup_frame", lambda _capture, _frames: np.zeros((8, 10, 3), dtype=np.uint8))
    candidate = MutableDiscovery().observations[1]
    try:
        assert source.read(timeout=1.0).shape == (8, 10, 3)
        assert active.value == 1
        assert access.test("front", CameraCandidate.from_observation(candidate))["ok"] is True
        assert active.value == 0
        assert broker.assert_dead()
        assert source.read(timeout=1.0).shape == (8, 10, 3)
        assert starts.value == 2
    finally:
        broker.release()
        with camera_manager._streams_lock:
            camera_manager._dshow_brokers.pop("front", None)


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


def test_refresh_during_role_test_invalidates_old_capture_before_confirmation():
    """A role test may only mint evidence for the exact candidate generation it captured."""
    class BlockingCameraAccess:
        def __init__(self):
            self.capture_started = threading.Event()
            self.allow_capture_to_finish = threading.Event()
            self.captured_indexes = []

        def test(self, role, candidate):
            self.captured_indexes.append(candidate.index)
            self.capture_started.set()
            assert self.allow_capture_to_finish.wait(timeout=5)
            return {"ok": True, "role": role, "frame": {"width": 1280, "height": 720}}

    discovery = MutableDiscovery()
    access = BlockingCameraAccess()
    service = make_service(discovery=discovery, access=access)
    original_generation = service.contract()["generation"]
    outcome = {}

    def run_role_test():
        try:
            outcome["result"] = service.test("top", "usb#top-001")
        except Exception as exc:  # test records the public failure result
            outcome["error"] = exc

    worker = threading.Thread(target=run_role_test)
    worker.start()
    assert access.capture_started.wait(timeout=5)

    discovery.observations[0] = {
        **discovery.observations[0],
        "index": 8,
    }
    refreshed_generation = service.refresh()["generation"]
    assert refreshed_generation != original_generation
    access.allow_capture_to_finish.set()
    worker.join(timeout=5)

    assert access.captured_indexes == [3]
    assert "result" not in outcome
    assert isinstance(outcome.get("error"), ValueError)
    assert "generation changed" in str(outcome["error"])

    fresh = service.test("top", "usb#top-001")["evidence"]
    assert fresh["generation"] == refreshed_generation
    assert service.confirm(
        "top", "usb#top-001", test_evidence_id=fresh["id"],
        operator_visual_confirmation=True, expected_generation=refreshed_generation,
    )["state"] == "ready"


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


def test_loopback_http_exposes_plain_maintenance_contract(monkeypatch):
    import app

    discovery = MutableDiscovery()
    service = make_service(discovery=discovery, access=RecordingCameraAccess())
    monkeypatch.setattr("app.get_camera_maintenance", lambda: service)
    client = TestClient(app.app)
    responses_schema = json.loads((Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json").read_text(encoding="utf-8"))

    listed = client.get("/maintenance/cameras")
    assert listed.status_code == 200
    assert discovery.calls == 1
    assert len(listed.json()["candidates"]) == 2
    jsonschema.validate(listed.json(), {"$ref": "#/$defs/contract", "$defs": responses_schema["$defs"]})

    refreshed = client.post("/maintenance/cameras/refresh")
    assert refreshed.status_code == 200
    assert discovery.calls == 2
    assert len(refreshed.json()["candidates"]) == 2
    jsonschema.validate(refreshed.json(), {"$ref": "#/$defs/refresh", "$defs": responses_schema["$defs"]})

    preview = client.get("/maintenance/cameras/usb%23top-001/preview.jpg")
    assert preview.status_code == 200
    assert preview.content == b"jpeg"
    assert preview.headers["cache-control"] == "no-store"

    response = client.post("/maintenance/cameras/top/confirm", json={"candidateId": "usb#top-001"})
    assert response.status_code == 409
    assert response.json()["contractVersion"] == CAMERA_MAINTENANCE_CONTRACT_VERSION

    tested = client.post(
        "/maintenance/cameras/top/test", json={"candidateId": "usb#top-001"},
    )
    assert tested.status_code == 200
    jsonschema.validate(tested.json(), {"$ref": "#/$defs/test", "$defs": responses_schema["$defs"]})
    evidence = tested.json()["evidence"]
    confirmed = client.post(
        "/maintenance/cameras/top/confirm",
        json={
            "candidateId": "usb#top-001", "testEvidenceId": evidence["id"],
            "operatorVisualConfirmation": True, "expectedGeneration": evidence["generation"],
        },
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
    assert client.get("/maintenance/cameras").status_code == 200


def test_health_status_does_not_reopen_real_cameras(monkeypatch):
    import app

    monkeypatch.setattr(
        app,
        "startup_check",
        {
            "ok": False,
            "checks": {
                "camera": {"ok": False, "message": "front camera unbound"},
                "modelManifest": {"ok": True},
                "pose": {"ok": True},
                "face": {"ok": True},
                "person": {"modelReady": True},
                "ageGender": {"modelReady": True, "mode": "model"},
            },
        },
    )
    monkeypatch.setattr(
        app,
        "get_all_camera_statuses",
        lambda: (_ for _ in ()).throw(AssertionError("health reopened a camera")),
    )

    status = app.get_runtime_status()

    assert status["cameraReady"] is False
    assert status["modelReady"] is True


def test_packaged_verifier_prints_captured_stdout_when_startup_wait_fails(
    monkeypatch, tmp_path, capsys
):
    from scripts import verify_packaged_exe

    class PackagedProcess:
        def __init__(self):
            self.returncode = None

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = PackagedProcess()
    def popen(*args, **kwargs):
        kwargs["stdout"].write("packaged startup entered camera discovery\n")
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(verify_packaged_exe.subprocess, "Popen", popen)
    monkeypatch.setattr(
        verify_packaged_exe,
        "wait_for_http",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("startup timed out")),
    )

    with pytest.raises(TimeoutError, match="startup timed out"):
        verify_packaged_exe.verify_managed_production_surface(
            tmp_path / "vending-vision.exe",
            port=17893,
            startup_timeout=0.01,
            temp_dir=tmp_path / "managed-production",
        )

    assert "packaged startup entered camera discovery" in capsys.readouterr().err


def test_packaged_verifier_reaps_an_already_exited_windows_server_without_taskkill(
    monkeypatch, tmp_path, capsys
):
    from scripts import verify_packaged_exe

    calls = []

    class PackagedProcess:
        pid = 4242
        returncode = 23

        def poll(self):
            calls.append("poll")
            return self.returncode

        def terminate(self):
            raise AssertionError("exited Windows leader must not be terminated by PID")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return self.returncode

    def taskkill(command, **kwargs):
        raise AssertionError("exited Windows leader must not be taskkilled by PID")

    monkeypatch.setattr(verify_packaged_exe.sys, "platform", "win32")
    monkeypatch.setattr(verify_packaged_exe.subprocess, "run", taskkill)
    process_log = (tmp_path / "packaged.log").open("w+", encoding="utf-8")
    try:
        process_log.write("actual packaged startup failure\n")
        process_log.flush()
        with pytest.raises(AssertionError, match="original verification failure"):
            try:
                raise AssertionError("original verification failure")
            finally:
                verify_packaged_exe.terminate_packaged_process(
                    PackagedProcess(), process_log, verification_failed=True
                )
    finally:
        process_log.close()

    assert calls == ["poll", ("wait", 10)]
    assert "actual packaged startup failure" in capsys.readouterr().err


@pytest.mark.parametrize("outcome", ["nonzero", "taskkill_timeout", "wait_timeout"])
def test_packaged_verifier_fails_closed_when_live_windows_tree_cannot_be_proven_dead(
    monkeypatch, tmp_path, outcome
):
    from scripts import verify_packaged_exe

    class PackagedProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            if outcome == "wait_timeout":
                raise verify_packaged_exe.subprocess.TimeoutExpired(
                    "vending-vision.exe", timeout
                )
            self.returncode = 1
            return self.returncode

    def taskkill(command, **kwargs):
        if outcome == "taskkill_timeout":
            raise verify_packaged_exe.subprocess.TimeoutExpired(
                command, kwargs["timeout"]
            )

        class Completed:
            returncode = 1

        return Completed()

    monkeypatch.setattr(verify_packaged_exe.sys, "platform", "win32")
    monkeypatch.setattr(verify_packaged_exe.subprocess, "run", taskkill)
    process_log = (tmp_path / "packaged.log").open("w+", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="Windows server tree"):
            verify_packaged_exe.terminate_packaged_process(PackagedProcess(), process_log)
    finally:
        process_log.close()


def test_packaged_verifier_keeps_verification_failure_primary_when_windows_cleanup_fails(
    monkeypatch, tmp_path, capsys
):
    from scripts import verify_packaged_exe

    class PackagedProcess:
        pid = 4242
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            raise AssertionError("cleanup must stop before leader wait")

    class Completed:
        returncode = 1

    monkeypatch.setattr(verify_packaged_exe.sys, "platform", "win32")
    monkeypatch.setattr(verify_packaged_exe.subprocess, "run", lambda *args, **kwargs: Completed())
    process_log = (tmp_path / "packaged.log").open("w+", encoding="utf-8")
    try:
        process_log.write("actual packaged startup failure\n")
        process_log.flush()
        with pytest.raises(AssertionError, match="original verification failure"):
            try:
                raise AssertionError("original verification failure")
            finally:
                verify_packaged_exe.terminate_packaged_process(
                    PackagedProcess(), process_log, verification_failed=True
                )
    finally:
        process_log.close()

    diagnostics = capsys.readouterr().err
    assert "actual packaged startup failure" in diagnostics
    assert "cleanup failed" in diagnostics


def test_packaged_verifier_kills_and_joins_a_live_windows_server_tree(monkeypatch, tmp_path):
    from scripts import verify_packaged_exe

    calls = []

    class PackagedProcess:
        pid = 4242
        returncode = None

        def poll(self):
            calls.append("poll")
            return None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 1
            return self.returncode

    def taskkill(command, **kwargs):
        calls.append(("taskkill", command, kwargs))

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(verify_packaged_exe.sys, "platform", "win32")
    monkeypatch.setattr(verify_packaged_exe.subprocess, "run", taskkill)
    process_log = (tmp_path / "packaged.log").open("w+", encoding="utf-8")
    try:
        verify_packaged_exe.terminate_packaged_process(PackagedProcess(), process_log)
    finally:
        process_log.close()

    assert calls == [
        "poll",
        (
            "taskkill",
            ["taskkill", "/PID", "4242", "/T", "/F"],
            {
                "stdin": verify_packaged_exe.subprocess.DEVNULL,
                "stdout": verify_packaged_exe.subprocess.DEVNULL,
                "stderr": verify_packaged_exe.subprocess.DEVNULL,
                "timeout": 10,
                "check": False,
            },
        ),
        ("wait", 10),
    ]


def test_release_version_does_not_publish_camera_index(monkeypatch):
    from app import version

    payload = version()

    assert "camera_index" not in json.dumps(payload).lower()
    assert CAMERA_MAINTENANCE_CONTRACT_VERSION.endswith("/v2")
