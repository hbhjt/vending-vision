import pytest
from fastapi.testclient import TestClient

import app as vision_app


class ObserverStub:
    def __init__(self):
        self.pid = 1001
        self.ready = True
        self.fatal_error = None
        self.stopped = False

    async def kill_child(self):
        self.stopped = True
        return True


class BrokerStub:
    def __init__(self):
        self.pid = 1002
        self.ready = True
        self.fatal_error = None

    async def shutdown(self):
        self.pid = None
        self.ready = False


def test_runtime_roles_reports_declared_process_roles(monkeypatch):
    observer = ObserverStub()
    broker = BrokerStub()
    monkeypatch.setattr(vision_app, "_get_acquisition_observer", lambda: observer)
    monkeypatch.setattr(vision_app, "_try_on_render_broker", broker)

    response = TestClient(vision_app.app).get("/v2/runtime/roles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == "vem-vision-runtime-roles/v1"
    by_name = {role["name"]: role for role in payload["roles"]}
    assert by_name["observer"]["pid"] == 1001
    assert by_name["observer"]["ready"] is True
    assert by_name["broker"]["pid"] == 1002
    assert by_name["broker"]["ready"] is True


def test_stop_observer_uses_the_declared_role_boundary(monkeypatch):
    observer = ObserverStub()
    monkeypatch.setattr(vision_app, "_get_acquisition_observer", lambda: observer)
    monkeypatch.setattr(vision_app, "_try_on_render_broker", BrokerStub())

    response = TestClient(vision_app.app).post("/v2/runtime/roles/observer/stop")

    assert response.status_code == 200
    assert response.json() == {"role": "observer", "stopped": True}
    assert observer.stopped is True


def test_stop_broker_shuts_down_the_render_worker(monkeypatch):
    broker = BrokerStub()
    monkeypatch.setattr(vision_app, "_get_acquisition_observer", lambda: ObserverStub())
    monkeypatch.setattr(vision_app, "_try_on_render_broker", broker)

    response = TestClient(vision_app.app).post("/v2/runtime/roles/broker/stop")

    assert response.status_code == 200
    assert broker.pid is None


def test_stop_unknown_role_is_rejected(monkeypatch):
    monkeypatch.setattr(vision_app, "_get_acquisition_observer", lambda: ObserverStub())
    monkeypatch.setattr(vision_app, "_try_on_render_broker", BrokerStub())

    response = TestClient(vision_app.app).post("/v2/runtime/roles/mystery/stop")

    assert response.status_code == 404
