"""Vision-owned stable camera-role binding and maintenance contract.

The camera backend index is deliberately an observation, never the persisted
identity.  The small public service below is the only API the HTTP layer and
camera runtime need: it hides discovery, persistence, ambiguity handling and
role resolution behind a versioned local contract.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


CAMERA_MAINTENANCE_CONTRACT_VERSION = "vem.vision.camera-maintenance/v1"
CAMERA_ROLES = ("top", "front")


class CameraDiscovery(Protocol):
    def enumerate(self) -> list[dict]: ...


class BindingStore(Protocol):
    def load(self) -> dict: ...

    def save(self, bindings: dict) -> None: ...


class CameraAccess(Protocol):
    def preview(self, candidate: "CameraCandidate") -> bytes: ...

    def test(self, candidate: "CameraCandidate") -> dict: ...


class JsonBindingStore:
    """Atomically stores Vision-owned identities outside the site-config schema."""

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
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


class WindowsCameraDiscovery:
    """Enumerates Windows PnP identities and observes transient OpenCV indexes.

    DirectShow exposes numeric indexes while Windows PnP exposes persistent
    instance IDs.  We only pair them when the observations are one-to-one;
    otherwise the contract deliberately leaves the index unknown rather than
    guessing a binding.
    """

    def __init__(self, max_indexes: int = 8, backend: str = "dshow"):
        self.max_indexes = max_indexes
        self.backend = backend

    def enumerate(self) -> list[dict]:
        identities = self._windows_identities()
        indexes = self._backend_indexes()
        matched_indexes = indexes if len(identities) == len(indexes) else [None] * len(identities)
        return [
            {
                "stableId": identity["stableId"],
                "label": identity["label"],
                "backend": self.backend,
                "index": index,
                "available": index is not None,
            }
            for identity, index in zip(identities, matched_indexes)
        ]

    def _windows_identities(self) -> list[dict]:
        if os.name != "nt":
            return []
        command = (
            "Get-PnpDevice -PresentOnly | Where-Object {$_.Class -eq 'Camera'} | "
            "Select-Object -Property InstanceId,FriendlyName,Status | ConvertTo-Json -Compress"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        try:
            values = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            return []
        identities = []
        for value in values:
            instance_id = value.get("InstanceId") if isinstance(value, dict) else None
            if not isinstance(instance_id, str) or not instance_id:
                continue
            identities.append(
                {
                    "stableId": instance_id,
                    "label": value.get("FriendlyName") or instance_id,
                }
            )
        return sorted(identities, key=lambda item: item["stableId"])

    def _backend_indexes(self) -> list[int]:
        try:
            import cv2
        except ImportError:
            return []
        backend = getattr(cv2, "CAP_DSHOW", 0) if self.backend.lower() == "dshow" else 0
        indexes = []
        for index in range(self.max_indexes):
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened():
                    indexes.append(index)
            finally:
                capture.release()
        return indexes


class OpenCvCameraAccess:
    """Short-lived local-only camera access used by maintenance preview/test."""

    def _open(self, candidate: CameraCandidate):
        from vision.camera import open_camera

        if candidate.index is None:
            raise RuntimeError("camera has no current backend index")
        return open_camera(camera_index=candidate.index, backend_name=candidate.backend)

    def preview(self, candidate: CameraCandidate) -> bytes:
        import cv2

        from vision.camera import read_warmup_frame

        capture = self._open(candidate)
        try:
            frame = read_warmup_frame(capture, 1)
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise RuntimeError("failed to encode camera preview")
            return encoded.tobytes()
        finally:
            capture.release()

    def test(self, candidate: CameraCandidate) -> dict:
        from vision.camera import describe_capture, read_warmup_frame

        capture = self._open(candidate)
        try:
            frame = read_warmup_frame(capture, 1)
            height, width = frame.shape[:2]
            return {
                "ok": True,
                "frame": {"width": width, "height": height},
                "backendObservation": {
                    "backend": candidate.backend,
                    "index": candidate.index,
                    "available": True,
                    "actual": describe_capture(capture),
                },
            }
        finally:
            capture.release()


def default_camera_binding_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "VendingVision" / "camera-bindings.json"


_maintenance_service: CameraMaintenanceService | None = None


def get_camera_maintenance() -> CameraMaintenanceService:
    global _maintenance_service
    if _maintenance_service is None:
        _maintenance_service = CameraMaintenanceService(
            WindowsCameraDiscovery(),
            JsonBindingStore(default_camera_binding_path()),
            OpenCvCameraAccess(),
        )
    return _maintenance_service


@dataclass(frozen=True)
class CameraCandidate:
    stable_id: str
    label: str
    backend: str
    index: int | None
    available: bool

    @classmethod
    def from_observation(cls, value: dict) -> "CameraCandidate":
        stable_id = value.get("stableId")
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError("camera candidate requires a stableId")
        index = value.get("index")
        if isinstance(index, bool) or (index is not None and not isinstance(index, int)):
            raise ValueError("camera candidate index must be an integer or null")
        return cls(
            stable_id=stable_id,
            label=str(value.get("label") or stable_id),
            backend=str(value.get("backend") or "unknown"),
            index=index,
            available=bool(value.get("available", False)),
        )

    def contract_value(self) -> dict:
        return {
            "id": self.stable_id,
            "label": self.label,
            "backendObservation": {
                "backend": self.backend,
                "index": self.index,
                "available": self.available,
            },
        }


class CameraMaintenanceService:
    """Owns local camera candidates and reports the versioned maintenance view."""

    def __init__(
        self,
        discovery: CameraDiscovery,
        store: BindingStore,
        access: CameraAccess | None = None,
    ):
        self._discovery = discovery
        self._store = store
        self._access = access
        loaded = store.load()
        self._bindings = loaded if isinstance(loaded, dict) else {}

    def candidates(self) -> list[CameraCandidate]:
        return sorted(
            (CameraCandidate.from_observation(value) for value in self._discovery.enumerate()),
            key=lambda candidate: candidate.stable_id,
        )

    def contract(self) -> dict:
        candidates = self.candidates()
        return {
            "contractVersion": CAMERA_MAINTENANCE_CONTRACT_VERSION,
            "candidates": [candidate.contract_value() for candidate in candidates],
            "roles": {
                role: self._role_status(role, candidates)
                for role in CAMERA_ROLES
            },
        }

    def confirm(self, role: str, candidate_id: str) -> dict:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        candidates = self.candidates()
        matches = [candidate for candidate in candidates if candidate.stable_id == candidate_id]
        if len(matches) != 1 or not matches[0].available:
            raise ValueError("camera candidate is missing, ambiguous, or unavailable")
        for other_role, binding in self._bindings.items():
            if (
                other_role != role
                and isinstance(binding, dict)
                and binding.get("stableId") == candidate_id
            ):
                raise ValueError("camera candidate is already confirmed for another role")
        self._bindings[role] = {"stableId": candidate_id}
        self._store.save(self._bindings)
        return self._role_status(role, candidates)

    def preview(self, candidate_id: str) -> bytes:
        if self._access is None:
            raise RuntimeError("camera preview is unavailable")
        return self._access.preview(self._candidate(candidate_id))

    def test(self, role: str, candidate_id: str) -> dict:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        if self._access is None:
            raise RuntimeError("camera test is unavailable")
        result = self._access.test(self._candidate(candidate_id))
        if not isinstance(result, dict):
            raise RuntimeError("camera test returned invalid evidence")
        return {"role": role, "candidateId": candidate_id, **result}

    def resolve(self, role: str) -> CameraCandidate:
        if role not in CAMERA_ROLES:
            raise ValueError(f"unknown camera role: {role}")
        binding = self._bindings.get(role)
        candidate_id = binding.get("stableId") if isinstance(binding, dict) else None
        if not isinstance(candidate_id, str):
            raise RuntimeError(f"{role} camera is not confirmed")
        candidates = [candidate for candidate in self.candidates() if candidate.stable_id == candidate_id]
        if len(candidates) != 1 or not candidates[0].available:
            raise RuntimeError(f"{role} camera is not ready")
        return candidates[0]

    def _candidate(self, candidate_id: str) -> CameraCandidate:
        candidates = [candidate for candidate in self.candidates() if candidate.stable_id == candidate_id]
        if len(candidates) != 1:
            raise ValueError("camera candidate is missing or ambiguous")
        candidate = candidates[0]
        if not candidate.available:
            raise ValueError("camera candidate is unavailable")
        return candidate

    def _role_status(self, role: str, candidates: list[CameraCandidate]) -> dict:
        binding = self._bindings.get(role)
        if not isinstance(binding, dict) or not binding.get("stableId"):
            return {
                "role": role,
                "state": "unbound",
                "ready": False,
                "reason": "camera_not_confirmed",
            }
        candidate_id = binding["stableId"]
        matches = [candidate for candidate in candidates if candidate.stable_id == candidate_id]
        if len(matches) == 0:
            return {
                "role": role,
                "state": "missing",
                "ready": False,
                "candidateId": candidate_id,
                "reason": "bound_camera_missing",
            }
        if len(matches) > 1:
            return {
                "role": role,
                "state": "ambiguous",
                "ready": False,
                "candidateId": candidate_id,
                "reason": "stable_identity_is_not_unique",
            }
        candidate = matches[0]
        if not candidate.available:
            return {
                "role": role,
                "state": "missing",
                "ready": False,
                "candidateId": candidate_id,
                "reason": "bound_camera_unavailable",
                "backendObservation": candidate.contract_value()["backendObservation"],
            }
        return {
            "role": role,
            "state": "ready",
            "ready": True,
            "candidateId": candidate_id,
            "backendObservation": candidate.contract_value()["backendObservation"],
        }
