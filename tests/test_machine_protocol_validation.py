from datetime import datetime, timezone

from app import validate_envelope, validate_message_payload
from vision.v2_contract_bundle import parse_v2_boundary_message


def envelope(message_type="vision.ping", message_id="message-1", timestamp=None, payload=None):
    return {
        "protocol": "vem.vision.v2",
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


def test_hello_uses_the_generated_v2_boundary_without_a_v1_payload_adapter():
    valid = {
        "protocol": "vem.vision.v2",
        "type": "vision.hello",
        "messageId": "hello-1",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {
            "clientRole": "machine",
            "machineCode": "M" * 64,
            "schemaVersion": "vem-vision-v2-contract-bundle/v1",
            "bundleVersion": "1",
            "contractDigest": "a" * 64,
            "capabilities": ["c" * 64],
        },
    }
    assert parse_v2_boundary_message(valid).type == "vision.hello"
    assert validate_message_payload("vision.hello", valid["payload"]) is not None

    invalid = {**valid, "payload": {**valid["payload"], "machineCode": "M" * 65}}
    try:
        parse_v2_boundary_message(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("generated V2 boundary accepted an overlong machineCode")


def test_unknown_client_message_types_are_hard_rejected():
    assert validate_message_payload("unsupported.client.action", {}) is not None


def test_unknown_client_message_type_is_rejected():
    assert validate_message_payload("vision.start_profile", {}) is not None
