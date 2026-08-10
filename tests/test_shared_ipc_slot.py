import json
import multiprocessing

import pytest

from vision.shared_ipc_slot import (
    _HEADER,
    SharedIpcChildConnection,
    SharedIpcError,
    SharedIpcSlot,
)


def _write_raw_response(slot: SharedIpcSlot, message: dict, response_blob: bytes = b"") -> None:
    encoded = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    _request_json, response_json, _request_bytes, response_bytes = slot._offsets()
    _HEADER.pack_into(slot._shm.buf, 0, 0, len(encoded), 0, len(response_blob))
    slot._shm.buf[response_json : response_json + len(encoded)] = encoded
    if response_blob:
        slot._shm.buf[response_bytes : response_bytes + len(response_blob)] = response_blob
    slot._response_event.set()


def _write_raw_request(slot: SharedIpcSlot, message: dict, request_blob: bytes = b"") -> None:
    encoded = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request_json, _response_json, request_bytes, _response_bytes = slot._offsets()
    _HEADER.pack_into(slot._shm.buf, 0, len(encoded), 0, len(request_blob), 0)
    slot._shm.buf[request_json : request_json + len(encoded)] = encoded
    if request_blob:
        slot._shm.buf[request_bytes : request_bytes + len(request_blob)] = request_blob
    slot._request_event.set()


@pytest.mark.parametrize(
    "bad_value",
    [True, "1", -1, 9],
)
def test_shared_ipc_response_rejects_non_exact_or_out_of_cap_response_bytes_before_slice(
    bad_value,
):
    slot = SharedIpcSlot(
        context=multiprocessing.get_context("spawn"),
        name_prefix="vem_test",
        response_bytes=1,
    )
    try:
        _write_raw_response(
            slot,
            {
                "kind": "ok",
                "payload": {"payloadType": "bytes", "byteSize": 1},
                "processGeneration": 3,
                "requestGeneration": 7,
                "responseBytes": bad_value,
            },
            b"x",
        )

        with pytest.raises(SharedIpcError):
            slot.recv_response(
                expected_process_generation=3,
                expected_request_generation=7,
            )
    finally:
        slot.close(unlink=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda message: {**message, "kind": 200},
        lambda message: {**message, "processGeneration": True},
        lambda message: {**message, "requestGeneration": "7"},
        lambda message: {**message, "processGeneration": 4},
        lambda message: {**message, "requestGeneration": 8},
        lambda message: {**message, "extra": "rejected"},
        lambda message: {
            key: value for key, value in message.items() if key != "requestGeneration"
        },
    ],
)
def test_shared_ipc_response_rejects_status_generation_and_key_drift(mutation):
    slot = SharedIpcSlot(
        context=multiprocessing.get_context("spawn"),
        name_prefix="vem_test",
        response_bytes=0,
    )
    try:
        message = mutation(
            {
                "kind": "ok",
                "payload": None,
                "processGeneration": 3,
                "requestGeneration": 7,
                "responseBytes": 0,
            }
        )
        _write_raw_response(slot, message)

        with pytest.raises(SharedIpcError):
            slot.recv_response(
                expected_process_generation=3,
                expected_request_generation=7,
            )
    finally:
        slot.close(unlink=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda message: {**message, "requestBytes": True},
        lambda message: {**message, "requestBytes": 1.0},
        lambda message: {**message, "requestBytes": "1"},
        lambda message: {**message, "requestBytes": -1},
        lambda message: {**message, "requestBytes": 2},
        lambda message: {**message, "payload": {**message["payload"], "garmentBytes": True}},
        lambda message: {**message, "payload": {**message["payload"], "garmentBytes": 1.0}},
        lambda message: {**message, "payload": {**message["payload"], "garmentBytes": "1"}},
        lambda message: {**message, "payload": {**message["payload"], "garmentBytes": -1}},
        lambda message: {**message, "payload": {**message["payload"], "garmentBytes": 2}},
    ],
)
def test_shared_ipc_child_request_rejects_non_exact_byte_metadata_before_payload_decode(
    mutation,
):
    slot = SharedIpcSlot(
        context=multiprocessing.get_context("spawn"),
        name_prefix="vem_test",
        request_bytes=1,
    )
    child = SharedIpcChildConnection(slot.config)
    try:
        message = mutation(
            {
                "command": "render",
                "payload": {"garmentBytes": 1},
                "processGeneration": 3,
                "requestGeneration": 7,
                "requestBytes": 1,
            }
        )
        _write_raw_request(slot, message, b"x")

        with pytest.raises(SharedIpcError):
            child.recv()
    finally:
        child.close()
        slot.close(unlink=True)
