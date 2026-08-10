"""Spawn-safe single-slot IPC over fixed shared memory and events."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import is_dataclass
from multiprocessing import shared_memory
from typing import Any, Callable
from uuid import uuid4


_HEADER = struct.Struct("<IIII")
_JSON_BYTES = 64 * 1024


class SharedIpcError(RuntimeError):
    pass


class SharedIpcSlot:
    """Parent-owned fixed control slot with one in-flight request."""

    def __init__(
        self,
        *,
        context,
        name_prefix: str,
        request_bytes: int = 0,
        response_bytes: int = 0,
    ):
        self._context = context
        self._request_bytes_cap = int(request_bytes)
        self._response_bytes_cap = int(response_bytes)
        self._request_event = context.Event()
        self._response_event = context.Event()
        self._lock = context.Lock()
        self._closed = False
        self._shm = shared_memory.SharedMemory(
            create=True,
            size=_HEADER.size + (_JSON_BYTES * 2) + self._request_bytes_cap + self._response_bytes_cap,
            name=f"{name_prefix}_ctl_{uuid4().hex}",
        )
        self._clear_lengths()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "name": self._shm.name,
            "requestJsonBytes": _JSON_BYTES,
            "responseJsonBytes": _JSON_BYTES,
            "requestBytes": self._request_bytes_cap,
            "responseBytes": self._response_bytes_cap,
            "requestEvent": self._request_event,
            "responseEvent": self._response_event,
            "lock": self._lock,
        }

    def _offsets(self) -> tuple[int, int, int, int]:
        request_json = _HEADER.size
        response_json = request_json + _JSON_BYTES
        request_bytes = response_json + _JSON_BYTES
        response_bytes = request_bytes + self._request_bytes_cap
        return request_json, response_json, request_bytes, response_bytes

    def _clear_lengths(self) -> None:
        with self._lock:
            _HEADER.pack_into(self._shm.buf, 0, 0, 0, 0, 0)
            self._request_event.clear()
            self._response_event.clear()

    def close(self, *, unlink: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
        finally:
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def submit(
        self,
        command: str,
        payload: dict[str, Any] | None,
        *,
        process_generation: int,
        request_generation: int,
    ) -> None:
        if self._closed:
            raise SharedIpcError("shared IPC slot is closed")
        payload = dict(payload or {})
        request_blob = b""
        if "garmentPng" in payload:
            garment = payload.pop("garmentPng")
            if not isinstance(garment, bytes):
                raise SharedIpcError("garmentPng must be bytes")
            request_blob = garment
            payload["garmentBytes"] = len(garment)
        message = {
            "command": command,
            "payload": payload,
            "processGeneration": process_generation,
            "requestGeneration": request_generation,
            "requestBytes": len(request_blob),
        }
        encoded = _encode_json_object(message, max_bytes=_JSON_BYTES, label="request")
        if len(request_blob) > self._request_bytes_cap:
            raise SharedIpcError("request bytes exceed IPC cap")
        request_json_offset, _response_json_offset, request_bytes_offset, _response_bytes_offset = self._offsets()
        with self._lock:
            self._request_event.clear()
            self._response_event.clear()
            _HEADER.pack_into(self._shm.buf, 0, len(encoded), 0, len(request_blob), 0)
            self._shm.buf[request_json_offset: request_json_offset + len(encoded)] = encoded
            if request_blob:
                self._shm.buf[request_bytes_offset: request_bytes_offset + len(request_blob)] = request_blob
            self._request_event.set()

    def poll_response(self) -> bool:
        return self._response_event.is_set()

    def recv_response(
        self,
        *,
        expected_process_generation: int | None = None,
        expected_request_generation: int | None = None,
    ) -> tuple[str, Any, int, int]:
        if not self._response_event.is_set():
            raise SharedIpcError("response is not ready")
        _request_json_len, response_json_len, _request_bytes_len, response_bytes_len = _HEADER.unpack_from(self._shm.buf, 0)
        if response_json_len <= 0 or response_json_len > _JSON_BYTES:
            raise SharedIpcError("invalid response metadata length")
        if response_bytes_len > self._response_bytes_cap:
            raise SharedIpcError("invalid response bytes length")
        _request_json_offset, response_json_offset, _request_bytes_offset, response_bytes_offset = self._offsets()
        raw = bytes(self._shm.buf[response_json_offset: response_json_offset + response_json_len])
        message = _decode_json_object(raw, label="response")
        expected_keys = {"kind", "payload", "processGeneration", "requestGeneration", "responseBytes"}
        if set(message) != expected_keys:
            raise SharedIpcError("invalid response metadata keys")
        kind = message["kind"]
        process_generation = _strict_int(message["processGeneration"], "processGeneration")
        request_generation = _strict_int(message["requestGeneration"], "requestGeneration")
        response_bytes = _strict_int(message["responseBytes"], "responseBytes")
        if response_bytes < 0 or response_bytes > self._response_bytes_cap:
            raise SharedIpcError("invalid response bytes length")
        if response_bytes != response_bytes_len:
            raise SharedIpcError("response byte length mismatch")
        if not isinstance(kind, str):
            raise SharedIpcError("response kind must be string")
        if (
            expected_process_generation is not None
            and process_generation != expected_process_generation
        ):
            raise SharedIpcError("response process generation mismatch")
        if (
            expected_request_generation is not None
            and request_generation != expected_request_generation
        ):
            raise SharedIpcError("response request generation mismatch")
        response_blob = bytes(self._shm.buf[response_bytes_offset: response_bytes_offset + response_bytes_len])
        payload = _decode_payload_from_json(message["payload"], response_blob)
        self._response_event.clear()
        return (
            kind,
            payload,
            process_generation,
            request_generation,
        )


class SharedIpcChildConnection:
    """Connection-like adapter used inside the spawned child."""

    def __init__(self, config: dict[str, Any]):
        self._request_json_cap = int(config["requestJsonBytes"])
        self._response_json_cap = int(config["responseJsonBytes"])
        self._request_bytes_cap = int(config["requestBytes"])
        self._response_bytes_cap = int(config["responseBytes"])
        self._request_event = config["requestEvent"]
        self._response_event = config["responseEvent"]
        self._lock = config["lock"]
        self._shm = shared_memory.SharedMemory(name=str(config["name"]))
        self.expected_process_generation = _optional_strict_int(
            config.get("expectedProcessGeneration"),
            "expectedProcessGeneration",
        )
        self._last_process_generation = 0
        self._last_request_generation = 0
        self._closed = False

    def _offsets(self) -> tuple[int, int, int, int]:
        request_json = _HEADER.size
        response_json = request_json + self._request_json_cap
        request_bytes = response_json + self._response_json_cap
        response_bytes = request_bytes + self._request_bytes_cap
        return request_json, response_json, request_bytes, response_bytes

    def recv(self) -> tuple[str, Any]:
        self._request_event.wait()
        request_json_len, _response_json_len, request_bytes_len, _response_bytes_len = _HEADER.unpack_from(self._shm.buf, 0)
        if request_json_len <= 0 or request_json_len > self._request_json_cap:
            raise SharedIpcError("invalid request metadata length")
        if request_bytes_len < 0 or request_bytes_len > self._request_bytes_cap:
            raise SharedIpcError("invalid request bytes length")
        request_json_offset, _response_json_offset, request_bytes_offset, _response_bytes_offset = self._offsets()
        raw = bytes(self._shm.buf[request_json_offset: request_json_offset + request_json_len])
        message = _decode_json_object(raw, label="request")
        expected_keys = {"command", "payload", "processGeneration", "requestGeneration", "requestBytes"}
        if set(message) != expected_keys:
            raise SharedIpcError("invalid request metadata keys")
        if message["requestBytes"] != request_bytes_len:
            raise SharedIpcError("request byte length mismatch")
        command = message["command"]
        payload = message["payload"]
        if not isinstance(command, str) or not isinstance(payload, dict):
            raise SharedIpcError("invalid request command payload")
        self._last_process_generation = _strict_int(message["processGeneration"], "processGeneration")
        self._last_request_generation = _strict_int(message["requestGeneration"], "requestGeneration")
        if (
            self.expected_process_generation is not None
            and self._last_process_generation != self.expected_process_generation
        ):
            raise SharedIpcError("request process generation mismatch")
        request_blob = bytes(self._shm.buf[request_bytes_offset: request_bytes_offset + request_bytes_len])
        if "garmentBytes" in payload:
            if payload["garmentBytes"] != request_bytes_len:
                raise SharedIpcError("garment byte length mismatch")
            payload = dict(payload)
            payload["garmentPng"] = request_blob
            del payload["garmentBytes"]
        self._request_event.clear()
        return command, payload

    def send(self, message: tuple[str, Any]) -> None:
        if not isinstance(message, tuple) or len(message) != 2:
            raise SharedIpcError("child response must be a pair")
        kind, payload = message
        payload_json, response_blob = _encode_payload_for_json(payload)
        if len(response_blob) > self._response_bytes_cap:
            raise SharedIpcError("response bytes exceed IPC cap")
        response = {
            "kind": kind,
            "payload": payload_json,
            "processGeneration": self._last_process_generation,
            "requestGeneration": self._last_request_generation,
            "responseBytes": len(response_blob),
        }
        encoded = _encode_json_object(response, max_bytes=self._response_json_cap, label="response")
        _request_json_offset, response_json_offset, _request_bytes_offset, response_bytes_offset = self._offsets()
        with self._lock:
            _HEADER.pack_into(
                self._shm.buf,
                0,
                _HEADER.unpack_from(self._shm.buf, 0)[0],
                len(encoded),
                _HEADER.unpack_from(self._shm.buf, 0)[2],
                len(response_blob),
            )
            self._shm.buf[response_json_offset: response_json_offset + len(encoded)] = encoded
            if response_blob:
                self._shm.buf[response_bytes_offset: response_bytes_offset + len(response_blob)] = response_blob
            self._response_event.set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()


def run_shared_ipc_child(
    target: Callable[..., Any],
    slot_config: dict[str, Any],
    target_args: tuple[Any, ...],
) -> None:
    connection = SharedIpcChildConnection(slot_config)
    try:
        target(connection, *target_args)
    finally:
        connection.close()


def wait_for_event(event, timeout: float, *, process=None) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if event.is_set():
            return True
        if process is not None and not process.is_alive():
            return False
        time.sleep(0.002)
    return event.is_set()


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise SharedIpcError(f"{label} must be int")
    return value


def _optional_strict_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _strict_int(value, label)


def _encode_json_object(value: dict[str, Any], *, max_bytes: int, label: str) -> bytes:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > max_bytes:
        raise SharedIpcError(f"{label} JSON exceeds IPC cap")
    return encoded


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise SharedIpcError(f"{label} JSON is invalid") from exc
    if not isinstance(value, dict):
        raise SharedIpcError(f"{label} JSON must be object")
    return value


def _encode_payload_for_json(payload: Any) -> tuple[Any, bytes]:
    if payload is None or isinstance(payload, (str, int, float, bool, list, dict)):
        return payload, b""
    if isinstance(payload, bytes):
        return {"payloadType": "bytes", "byteSize": len(payload)}, payload
    if is_dataclass(payload) and payload.__class__.__name__ == "AcquisitionObservation":
        return {
            "payloadType": "acquisition_observation",
            "byteSize": len(payload.jpeg),
            "occupancy": payload.occupancy,
            "aligned": payload.aligned,
        }, payload.jpeg
    raise SharedIpcError(f"unsupported response payload type: {type(payload).__name__}")


def _decode_payload_from_json(payload: Any, response_blob: bytes) -> Any:
    if isinstance(payload, dict) and payload.get("payloadType") == "bytes":
        if (
            set(payload) != {"payloadType", "byteSize"}
            or _strict_int(payload["byteSize"], "byteSize") != len(response_blob)
        ):
            raise SharedIpcError("invalid byte response metadata")
        return response_blob
    if isinstance(payload, dict) and payload.get("payloadType") == "acquisition_observation":
        if set(payload) != {"payloadType", "byteSize", "occupancy", "aligned"}:
            raise SharedIpcError("invalid acquisition response metadata")
        if _strict_int(payload["byteSize"], "byteSize") != len(response_blob):
            raise SharedIpcError("acquisition response byte length mismatch")
        from vision.acquisition_observer import AcquisitionObservation

        if not isinstance(payload["occupancy"], str) or type(payload["aligned"]) is not bool:
            raise SharedIpcError("invalid acquisition response fields")
        return AcquisitionObservation(response_blob, payload["occupancy"], payload["aligned"])
    if response_blob:
        raise SharedIpcError("unexpected response bytes")
    return payload
