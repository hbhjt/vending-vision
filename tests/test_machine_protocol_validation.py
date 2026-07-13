from datetime import datetime, timezone

from app import validate_envelope, validate_message_payload


def envelope(message_type="vision.ping", message_id="message-1", timestamp=None, payload=None):
    return {
        "protocol": "vem.vision.v1",
        "type": message_type,
        "messageId": message_id,
        "timestamp": timestamp
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": payload or {},
    }


def test_envelope_rejects_non_iso_or_timezone_less_timestamp():
    assert validate_envelope(envelope(timestamp="yesterday")) is not None
    assert validate_envelope(envelope(timestamp="2026-07-13T10:00:00")) is not None


def test_envelope_rejects_overlong_message_id():
    assert validate_envelope(envelope(message_id="m" * 129)) is not None
    assert validate_envelope(envelope(message_id="m" * 128)) is None


def test_hello_requires_bounded_machine_contract_fields():
    valid = {
        "clientRole": "machine",
        "machineCode": "M" * 64,
        "protocolVersion": 1,
        "capabilities": ["c" * 64],
    }
    assert validate_message_payload("vision.hello", valid) is None
    assert validate_message_payload("vision.hello", {**valid, "machineCode": "M" * 65}) is not None
    assert validate_message_payload("vision.hello", {**valid, "capabilities": ["c" * 65]}) is not None


def test_try_on_payload_matches_vem_bounds_and_stop_enum():
    assert validate_message_payload(
        "vision.try_on.start",
        {"sessionId": "s" * 128, "catalogKey": "c" * 128, "variantId": "v" * 128},
    ) is None
    assert validate_message_payload("vision.try_on.start", {"sessionId": "s" * 129}) is not None
    assert validate_message_payload(
        "vision.try_on.stop", {"sessionId": "session", "reason": "route_leave"}
    ) is None
    assert validate_message_payload(
        "vision.try_on.stop", {"sessionId": "session", "reason": "websocket_disconnected"}
    ) is not None


def test_unknown_client_message_type_is_rejected():
    assert validate_message_payload("vision.start_profile", {}) is not None
