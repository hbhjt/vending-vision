import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from vision.config import settings


ALLOWED_FRONT_CAMERA_OWNERS = {"vision", "tryon_frontend"}
FRONT_CAMERA_PRIORITY = {
    "idle": 0,
    "vision": 1,
    "tryon_frontend": 2,
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class FrontCameraOwner:
    def __init__(self):
        self.lock = threading.RLock()
        self.owner = "idle"
        self.reason = None
        self.updated_at = _now_iso()
        self.updated_monotonic = time.time()

    def _expire_locked(self):
        if self.owner == "idle":
            return

        timeout_ms = max(int(settings.FRONT_CAMERA_OWNER_TIMEOUT_MS), 0)
        if timeout_ms <= 0:
            return

        elapsed_ms = int((time.time() - self.updated_monotonic) * 1000)
        if elapsed_ms < timeout_ms:
            return

        self.owner = "idle"
        self.reason = "owner_timeout"
        self.updated_at = _now_iso()
        self.updated_monotonic = time.time()

    def status(self):
        with self.lock:
            self._expire_locked()
            return {
                "owner": self.owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
                "timeoutMs": settings.FRONT_CAMERA_OWNER_TIMEOUT_MS,
            }

    def acquire(self, owner: str, reason: str | None = None):
        if owner not in ALLOWED_FRONT_CAMERA_OWNERS:
            return {
                "ok": False,
                "owner": self.owner,
                "requestedOwner": owner,
                "error": "invalid_owner",
            }

        with self.lock:
            self._expire_locked()
            current_owner = self.owner
            current_priority = FRONT_CAMERA_PRIORITY.get(current_owner, 0)
            requested_priority = FRONT_CAMERA_PRIORITY[owner]

            if current_owner not in {"idle", owner} and requested_priority < current_priority:
                return {
                    "ok": False,
                    "owner": current_owner,
                    "requestedOwner": owner,
                    "reason": self.reason,
                    "error": "front_camera_busy",
                }

            self.owner = owner
            self.reason = reason
            self.updated_at = _now_iso()
            self.updated_monotonic = time.time()

            return {
                "ok": True,
                "owner": self.owner,
                "previousOwner": current_owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
            }

    def release(self, owner: str, reason: str | None = None):
        with self.lock:
            self._expire_locked()

            if owner not in ALLOWED_FRONT_CAMERA_OWNERS:
                return {
                    "ok": False,
                    "owner": self.owner,
                    "requestedOwner": owner,
                    "error": "invalid_owner",
                }

            if self.owner != owner:
                return {
                    "ok": False,
                    "owner": self.owner,
                    "requestedOwner": owner,
                    "error": "owner_mismatch",
                }

            previous_owner = self.owner
            self.owner = "idle"
            self.reason = reason
            self.updated_at = _now_iso()
            self.updated_monotonic = time.time()

            return {
                "ok": True,
                "owner": self.owner,
                "previousOwner": previous_owner,
                "reason": self.reason,
                "updatedAt": self.updated_at,
            }


_front_camera_owner = FrontCameraOwner()
_front_camera_io_lock = threading.RLock()


def get_front_camera_owner():
    return _front_camera_owner.status()


def acquire_front_camera(owner: str, reason: str | None = None):
    return _front_camera_owner.acquire(owner, reason=reason)


def release_front_camera(owner: str, reason: str | None = None):
    return _front_camera_owner.release(owner, reason=reason)


@contextmanager
def front_camera_io_lock():
    with _front_camera_io_lock:
        yield
