"""Vision-owned, fail-closed camera role binding.

The Windows discovery boundary deliberately does *not* correlate independent
PnP and OpenCV enumerations.  A candidate is usable only when one adapter has
proved the stable identity and the capture source belong to the same device.
Backend indexes are maintenance observations, never persisted identities.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


CAMERA_MAINTENANCE_CONTRACT_VERSION = "vem.vision.camera-maintenance/v2"
CAMERA_ROLES = ("top", "front")
TEST_EVIDENCE_TTL_SECONDS = 120


class CameraDiscovery(Protocol):
    def enumerate(self) -> list[dict]: ...


class BindingStore(Protocol):
    def load(self) -> dict: ...

    def save(self, bindings: dict) -> None: ...


class CameraAccess(Protocol):
    def preview(self, candidate: "CameraCandidate") -> bytes: ...

    def test(self, candidate: "CameraCandidate") -> dict: ...


class WindowsMediaSourceAdapter(Protocol):
    """Public boundary for an adapter that can prove identity-to-source mapping."""

    def enumerate_sources(self) -> list[dict]: ...


class JsonBindingStore:
    """Atomically stores only Vision-owned stable identities."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"camera binding store is unreadable: {self.path}") from exc
        if not isinstance(value, dict) or value.get("version") != 1:
            raise RuntimeError("camera binding store has an unsupported format")
        bindings = value.get("bindings")
        if not isinstance(bindings, dict):
            raise RuntimeError("camera binding store is missing bindings")
        return bindings

    def save(self, bindings: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": 1, "bindings": bindings}, ensure_ascii=False, indent=2)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


class PowerShellWindowsMediaSourceAdapter:
    """Enumerates Media Foundation identities but never invents an OpenCV index."""

    def enumerate_sources(self) -> list[dict]:
        if os.name != "nt":
            return []
        command = r'''Add-Type -AssemblyName System.Runtime.WindowsRuntime
$class=[Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime]
$enum=[Windows.Devices.Enumeration.DeviceInformation,Windows.Devices.Enumeration,ContentType=WindowsRuntime]::FindAllAsync([Windows.Devices.Enumeration.DeviceClass]::VideoCapture)
$devices=[System.WindowsRuntimeSystemExtensions]::AsTask($enum).Result
@($devices | ForEach-Object { [PSCustomObject]@{ stableId=$_.Id; label=$_.Name; backend='mediafoundation'; source=$_.Id } }) | ConvertTo-Json -Compress'''
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        if isinstance(values, dict):
            values = [values]
        return values if isinstance(values, list) else []


class WindowsCameraDiscovery:
    """Windows Media Foundation source enumeration, with a fail-closed adapter.

    The Windows Runtime VideoCapture API returns a media-source identifier from
    the same enumeration that returned the device identity.  OpenCV's numeric
    DirectShow indexes are not that API, so this default adapter deliberately
    leaves ``index`` empty until a capture adapter can prove a source mapping.
    It is therefore safe on a machine with multiple or replugged cameras.
    """

    def __init__(self, adapter: WindowsMediaSourceAdapter | None = None):
        self._adapter = adapter or PowerShellWindowsMediaSourceAdapter()

    def enumerate(self) -> list[dict]:
        values = self._adapter.enumerate_sources()
        result = []
        for value in values:
            stable_id = value.get("stableId") if isinstance(value, dict) else None
            if not isinstance(stable_id, str) or not stable_id:
                continue
            index = value.get("index")
            proven = value.get("mappingState") == "proven" and isinstance(index, int) and index >= 0
            result.append({
                "stableId": stable_id,
                "label": str(value.get("label") or stable_id),
                "backend": str(value.get("backend") or "mediafoundation"),
                "index": index if proven else None,
                "available": proven,
                "mappingState": "proven" if proven else "unproven",
                "source": value.get("source") or stable_id,
            })
        return result

class CameraLeaseRegistry:
    """One local owner across runtime, preview and role test capture."""

    def __init__(self):
        self._lock = threading.RLock()
        self._owners: dict[str, str] = {}

    def acquire(self, candidate_id: str, owner: str):
        with self._lock:
            current = self._owners.get(candidate_id)
            if current is not None:
                raise RuntimeError(f"camera is leased by {current}")
            self._owners[candidate_id] = owner
        return _CameraLease(self, candidate_id, owner)

    def _release(self, candidate_id: str, owner: str) -> None:
        with self._lock:
            if self._owners.get(candidate_id) == owner:
                self._owners.pop(candidate_id, None)


class _CameraLease:
    def __init__(self, registry: CameraLeaseRegistry, candidate_id: str, owner: str):
        self._registry, self._candidate_id, self._owner = registry, candidate_id, owner

    def release(self) -> None:
        self._registry._release(self._candidate_id, self._owner)


class OpenCvCameraAccess:
    """Short-lived maintenance access sharing the process-wide lease registry."""

    def __init__(self, leases: CameraLeaseRegistry | None = None):
        self.leases = leases or _camera_leases

    def _open(self, candidate: "CameraCandidate"):
        from vision.camera import open_camera
        if candidate.index is None:
            raise RuntimeError("camera mapping is unproven; no capture source is available")
        return open_camera(camera_index=candidate.index, backend_name=candidate.backend)

    def preview(self, candidate: "CameraCandidate") -> bytes:
        import cv2
        from vision.camera import read_warmup_frame
        lease = self.leases.acquire(candidate.stable_id, "maintenance-preview")
        try:
            capture = self._open(candidate)
            try:
                frame = read_warmup_frame(capture, 1)
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    raise RuntimeError("failed to encode camera preview")
                return encoded.tobytes()
            finally:
                capture.release()
        finally:
            lease.release()

    def test(self, candidate: "CameraCandidate") -> dict:
        from vision.camera import describe_capture, read_warmup_frame
        lease = self.leases.acquire(candidate.stable_id, "maintenance-test")
        try:
            capture = self._open(candidate)
            try:
                frame = read_warmup_frame(capture, 1)
                height, width = frame.shape[:2]
                return {"ok": True, "frame": {"width": width, "height": height},
                        "backendObservation": {"backend": candidate.backend, "index": candidate.index,
                                               "available": True, "mappingState": "proven"}}
            finally:
                capture.release()
        finally:
            lease.release()


@dataclass(frozen=True)
class CameraCandidate:
    stable_id: str
    label: str
    backend: str
    index: int | None
    available: bool
    mapping_state: str

    @classmethod
    def from_observation(cls, value: dict) -> "CameraCandidate":
        stable_id = value.get("stableId")
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError("camera candidate requires a stableId")
        index = value.get("index")
        if isinstance(index, bool) or (index is not None and not isinstance(index, int)):
            raise ValueError("camera candidate index must be an integer or null")
        mapping_state = value.get("mappingState", "proven" if value.get("available") else "unproven")
        if mapping_state not in {"proven", "unproven"}:
            raise ValueError("camera candidate mappingState is invalid")
        available = bool(value.get("available", False)) and mapping_state == "proven" and index is not None
        return cls(stable_id, str(value.get("label") or stable_id), str(value.get("backend") or "unknown"),
                   index, available, mapping_state)

    def contract_value(self) -> dict:
        return {"id": self.stable_id, "label": self.label, "backendObservation": {
            "backend": self.backend, "index": self.index, "available": self.available,
            "mappingState": self.mapping_state,
        }}


class MaintenanceCapabilityError(ValueError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class MaintenanceCapabilityVerifier:
    """Verifies a single-use, scoped maintenance capability without JWT deps."""

    def __init__(self, secret: str | None, clock=time.time):
        self._secret = secret.encode() if secret else None
        self._clock = clock
        self._used: dict[str, int] = {}
        self._lock = threading.RLock()

    def verify(self, token: str | None, required_scope: str) -> dict:
        if not self._secret or not token:
            raise MaintenanceCapabilityError("maintenance capability is required", 401)
        try:
            encoded, signature = token.split(".", 1)
            expected = hmac.new(self._secret, encoded.encode(), hashlib.sha256).hexdigest()
            padded = encoded + "=" * (-len(encoded) % 4)
            claims = json.loads(base64.urlsafe_b64decode(padded))
        except Exception as exc:
            raise MaintenanceCapabilityError("maintenance capability is invalid", 401) from exc
        if not hmac.compare_digest(signature, expected):
            raise MaintenanceCapabilityError("maintenance capability is invalid", 401)
        now = int(self._clock())
        if not isinstance(claims, dict) or claims.get("purpose") != "vision.camera-maintenance":
            raise MaintenanceCapabilityError("maintenance capability has the wrong purpose", 403)
        scopes, expires_at, jti = claims.get("scope"), claims.get("exp"), claims.get("jti")
        if not isinstance(scopes, list) or required_scope not in scopes:
            raise MaintenanceCapabilityError("maintenance capability lacks required scope", 403)
        if isinstance(expires_at, bool) or not isinstance(expires_at, int) or expires_at <= now:
            raise MaintenanceCapabilityError("maintenance capability has expired", 401)
        if not isinstance(jti, str) or not jti:
            raise MaintenanceCapabilityError("maintenance capability lacks replay id", 401)
        with self._lock:
            self._used = {key: expiry for key, expiry in self._used.items() if expiry > now}
            if jti in self._used:
                raise MaintenanceCapabilityError("maintenance capability was already used", 409)
            self._used[jti] = expires_at
        return claims


class CameraMaintenanceService:
    """A cached candidate generation and atomic role/evidence state machine."""

    _mutation_lock = threading.RLock()

    def __init__(self, discovery: CameraDiscovery, store: BindingStore, access: CameraAccess | None = None, *, clock=time.time):
        self._discovery, self._store, self._access, self._clock = discovery, store, access, clock
        self._bindings = store.load()
        if not isinstance(self._bindings, dict):
            raise RuntimeError("camera binding store returned invalid bindings")
        self._snapshot: list[CameraCandidate] | None = None
        self._generation_number = 0
        self._generation = ""
        self._evidence: dict[str, dict] = {}
        self._lock = threading.RLock()

    def refresh(self) -> dict:
        with self._lock:
            observations = self._discovery.enumerate()
            self._snapshot = sorted((CameraCandidate.from_observation(value) for value in observations), key=lambda c: c.stable_id)
            self._generation_number += 1
            digest = hashlib.sha256(json.dumps([c.contract_value() for c in self._snapshot], sort_keys=True).encode()).hexdigest()[:16]
            self._generation = f"{self._generation_number}-{digest}"
            return {"generation": self._generation, "candidates": [c.contract_value() for c in self._snapshot]}

    def refresh_after_read_failure(self) -> dict:
        return self.refresh()

    def _candidates(self) -> list[CameraCandidate]:
        if self._snapshot is None:
            self.refresh()
        return list(self._snapshot or [])

    def contract(self) -> dict:
        candidates = self._candidates()
        return {"contractVersion": CAMERA_MAINTENANCE_CONTRACT_VERSION, "generation": self._generation,
                "candidates": [candidate.contract_value() for candidate in candidates],
                "roles": {role: self._role_status(role, candidates) for role in CAMERA_ROLES}}

    def resolve(self, role: str) -> CameraCandidate:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        candidates = self._candidates()
        status = self._role_status(role, candidates)
        if status["state"] != "ready":
            raise RuntimeError(f"{role} camera is not ready: {status.get('reason')}")
        return next(candidate for candidate in candidates if candidate.stable_id == status["candidateId"])

    def preview(self, candidate_id: str) -> bytes:
        if self._access is None:
            raise RuntimeError("camera preview is unavailable")
        return self._access.preview(self._candidate(candidate_id))

    def test(self, role: str, candidate_id: str) -> dict:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        if self._access is None:
            raise RuntimeError("camera test is unavailable")
        candidate = self._candidate(candidate_id)
        result = self._access.test(candidate)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise RuntimeError("camera test did not produce successful evidence")
        evidence_id = secrets.token_urlsafe(18)
        expires_at = int(self._clock()) + TEST_EVIDENCE_TTL_SECONDS
        self._evidence[evidence_id] = {"role": role, "candidateId": candidate_id,
                                       "generation": self._generation, "expiresAt": expires_at, "used": False}
        return {"role": role, "candidateId": candidate_id, "generation": self._generation, **result,
                "evidence": {"id": evidence_id, "role": role, "candidateId": candidate_id,
                             "generation": self._generation, "expiresAt": expires_at}}

    def confirm(self, role: str, candidate_id: str, *, test_evidence_id: str | None = None,
                operator_visual_confirmation: bool = False) -> dict:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        with self._mutation_lock, self._lock:
            candidate = self._candidate(candidate_id)
            if not operator_visual_confirmation:
                self._consume_evidence(test_evidence_id, role, candidate_id)
            # Reload under the mutation lock so a second service cannot confirm
            # against stale persisted state within this process.
            current = self._store.load()
            if not isinstance(current, dict):
                raise RuntimeError("camera binding store returned invalid bindings")
            for other_role, binding in current.items():
                if other_role != role and isinstance(binding, dict) and binding.get("stableId") == candidate.stable_id:
                    raise ValueError("camera candidate is already confirmed for another role")
            current[role] = {
                "stableId": candidate.stable_id,
                "confirmation": {
                    "method": "operator_visual" if operator_visual_confirmation else "role_test",
                    "generation": self._generation,
                    "confirmedAt": int(self._clock()),
                },
            }
            self._store.save(current)
            self._bindings = current
            return self._role_status(role, self._candidates())

    def _consume_evidence(self, evidence_id: str | None, role: str, candidate_id: str) -> None:
        if not isinstance(evidence_id, str):
            raise ValueError("fresh successful role test evidence is required")
        evidence = self._evidence.get(evidence_id)
        if evidence is None:
            raise ValueError("test evidence is unknown")
        if evidence["used"]:
            raise ValueError("test evidence is already consumed")
        if evidence["expiresAt"] <= int(self._clock()):
            raise ValueError("test evidence has expired")
        if evidence["role"] != role:
            raise ValueError("test evidence is for another role")
        if evidence["candidateId"] != candidate_id or evidence["generation"] != self._generation:
            raise ValueError("test evidence does not match the current candidate generation")
        evidence["used"] = True

    def _candidate(self, candidate_id: str) -> CameraCandidate:
        matches = [candidate for candidate in self._candidates() if candidate.stable_id == candidate_id]
        if len(matches) != 1:
            raise ValueError("camera candidate is missing or ambiguous")
        candidate = matches[0]
        if not candidate.available:
            raise ValueError("camera candidate mapping is unproven or unavailable")
        return candidate

    def _role_status(self, role: str, candidates: list[CameraCandidate]) -> dict:
        binding = self._bindings.get(role)
        candidate_id = binding.get("stableId") if isinstance(binding, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            return {"role": role, "state": "unbound", "ready": False, "reason": "camera_not_confirmed"}
        duplicate_roles = [name for name, value in self._bindings.items()
                           if isinstance(value, dict) and value.get("stableId") == candidate_id]
        if len(duplicate_roles) > 1:
            return {"role": role, "state": "ambiguous", "ready": False, "candidateId": candidate_id,
                    "reason": "stable_identity_bound_to_multiple_roles"}
        matches = [candidate for candidate in candidates if candidate.stable_id == candidate_id]
        if not matches:
            return {"role": role, "state": "missing", "ready": False, "candidateId": candidate_id,
                    "reason": "bound_camera_missing"}
        if len(matches) > 1:
            return {"role": role, "state": "ambiguous", "ready": False, "candidateId": candidate_id,
                    "reason": "stable_identity_is_not_unique"}
        candidate = matches[0]
        if candidate.mapping_state != "proven":
            return {"role": role, "state": "ambiguous", "ready": False, "candidateId": candidate_id,
                    "reason": "camera_mapping_unproven", "backendObservation": candidate.contract_value()["backendObservation"]}
        if not candidate.available:
            return {"role": role, "state": "missing", "ready": False, "candidateId": candidate_id,
                    "reason": "bound_camera_unavailable", "backendObservation": candidate.contract_value()["backendObservation"]}
        return {"role": role, "state": "ready", "ready": True, "candidateId": candidate_id,
                "backendObservation": candidate.contract_value()["backendObservation"]}


def default_camera_binding_path() -> Path:
    root = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) if os.name == "nt" else Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "VendingVision" / "camera-bindings.json"


_camera_leases = CameraLeaseRegistry()
_maintenance_service: CameraMaintenanceService | None = None


def acquire_runtime_camera_lease(candidate_id: str, role: str):
    """Claim the same local lease namespace used by preview and role tests."""
    return _camera_leases.acquire(candidate_id, f"runtime:{role}")


def get_camera_maintenance() -> CameraMaintenanceService:
    global _maintenance_service
    if _maintenance_service is None:
        _maintenance_service = CameraMaintenanceService(WindowsCameraDiscovery(), JsonBindingStore(default_camera_binding_path()), OpenCvCameraAccess())
    return _maintenance_service
