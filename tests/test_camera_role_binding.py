import json
from pathlib import Path

import jsonschema

from vision.camera_binding import CameraMaintenanceService, JsonBindingStore


class StaticCameraDiscovery:
    def __init__(self, observations=None):
        self.observations = observations or [
            {
                "stableId": "usb#top-001",
                "label": "Top camera",
                "backend": "dshow",
                "index": 3,
                "available": True,
            },
            {
                "stableId": "usb#front-002",
                "label": "Front camera",
                "backend": "dshow",
                "index": 7,
                "available": True,
            },
        ]

    def enumerate(self):
        return self.observations


class InMemoryBindings:
    def load(self):
        return {}

    def save(self, bindings):
        self.saved = bindings


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


def test_maintenance_contract_lists_stable_candidates_and_backend_observations():
    service = CameraMaintenanceService(StaticCameraDiscovery(), InMemoryBindings())

    contract = service.contract()

    assert contract["contractVersion"] == "vem.vision.camera-maintenance/v1"
    assert contract["candidates"] == [
        {
            "id": "usb#front-002",
            "label": "Front camera",
            "backendObservation": {
                "backend": "dshow",
                "index": 7,
                "available": True,
            },
        },
        {
            "id": "usb#top-001",
            "label": "Top camera",
            "backendObservation": {
                "backend": "dshow",
                "index": 3,
                "available": True,
            },
        },
    ]


def test_maintenance_contract_remains_compatible_with_its_versioned_schema():
    schema_path = Path(__file__).parents[1] / "config" / "vending-vision-camera-maintenance-v1.schema.json"
    service = CameraMaintenanceService(StaticCameraDiscovery(), InMemoryBindings())

    jsonschema.validate(service.contract(), json.loads(schema_path.read_text(encoding="utf-8")))


def test_confirmed_role_uses_stable_identity_after_backend_index_changes():
    discovery = StaticCameraDiscovery()
    store = InMemoryBindings()
    service = CameraMaintenanceService(discovery, store)

    service.confirm("top", "usb#top-001")
    discovery.observations = [
        {
            "stableId": "usb#top-001",
            "label": "Top camera",
            "backend": "dshow",
            "index": 11,
            "available": True,
        },
        {
            "stableId": "usb#front-002",
            "label": "Front camera",
            "backend": "dshow",
            "index": 2,
            "available": True,
        },
    ]

    role = service.contract()["roles"]["top"]

    assert store.saved["top"]["stableId"] == "usb#top-001"
    assert role["state"] == "ready"
    assert role["candidateId"] == "usb#top-001"
    assert role["backendObservation"]["index"] == 11


def test_binding_survives_a_new_service_instance(tmp_path):
    binding_file = tmp_path / "vision" / "camera-bindings.json"
    first = CameraMaintenanceService(StaticCameraDiscovery(), JsonBindingStore(binding_file))
    first.confirm("front", "usb#front-002")

    restarted = CameraMaintenanceService(StaticCameraDiscovery(), JsonBindingStore(binding_file))

    role = restarted.contract()["roles"]["front"]
    assert role["state"] == "ready"
    assert role["candidateId"] == "usb#front-002"


def test_missing_and_duplicate_stable_identities_are_explicitly_not_ready():
    store = InMemoryBindings()
    discovery = StaticCameraDiscovery()
    service = CameraMaintenanceService(discovery, store)
    service.confirm("top", "usb#top-001")
    discovery.observations = []

    assert service.contract()["roles"]["top"] == {
        "role": "top",
        "state": "missing",
        "ready": False,
        "candidateId": "usb#top-001",
        "reason": "bound_camera_missing",
    }

    discovery.observations = [
        {"stableId": "usb#top-001", "label": "one", "backend": "dshow", "index": 0, "available": True},
        {"stableId": "usb#top-001", "label": "two", "backend": "dshow", "index": 1, "available": True},
    ]

    assert service.contract()["roles"]["top"]["state"] == "ambiguous"
    assert service.contract()["roles"]["top"]["reason"] == "stable_identity_is_not_unique"


def test_preview_and_role_test_use_only_the_selected_candidate():
    access = RecordingCameraAccess()
    service = CameraMaintenanceService(StaticCameraDiscovery(), InMemoryBindings(), access)

    assert service.preview("usb#front-002") == b"jpeg"
    result = service.test("front", "usb#front-002")

    assert access.previewed == ["usb#front-002"]
    assert access.tested == ["usb#front-002"]
    assert result == {
        "role": "front",
        "candidateId": "usb#front-002",
        "ok": True,
        "frame": {"width": 1280, "height": 720},
    }


def test_loopback_maintenance_endpoints_expose_contract_preview_test_and_confirm(monkeypatch):
    from app import (
        camera_maintenance_confirm,
        camera_maintenance_contract,
        camera_maintenance_preview,
        camera_maintenance_test,
    )

    access = RecordingCameraAccess()
    service = CameraMaintenanceService(StaticCameraDiscovery(), InMemoryBindings(), access)
    monkeypatch.setattr("app.get_camera_maintenance", lambda: service)

    contract = camera_maintenance_contract()
    preview = camera_maintenance_preview("usb#front-002")
    tested = camera_maintenance_test("front", {"candidateId": "usb#front-002"})
    confirmed = camera_maintenance_confirm("front", {"candidateId": "usb#front-002"})

    assert contract["contractVersion"] == "vem.vision.camera-maintenance/v1"
    assert preview.media_type == "image/jpeg"
    assert preview.headers["cache-control"] == "no-store"
    assert tested["ok"] is True
    assert confirmed["state"] == "ready"
