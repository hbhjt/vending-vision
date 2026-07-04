import re
import threading
import time
from datetime import datetime, timezone

import cv2

from vision.camera_manager import read_camera
from vision.camera_owner import (
    acquire_front_camera,
    front_camera_io_lock,
    release_front_camera,
)
from vision.config import settings


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _preview_url(session_id: str) -> str:
    return f"http://{settings.HOST}:{settings.PORT}/try-on/{session_id}.mjpeg"


def _validate_session_id(session_id: str | None):
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("payload.sessionId must be a non-empty string")

    session_id = session_id.strip()
    if not SESSION_ID_PATTERN.match(session_id):
        raise ValueError("payload.sessionId contains unsupported characters")

    return session_id


class TryOnSessionManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active_session_id = None
        self.sessions = {}

    def start(self, session_id: str, catalog_key: str | None = None, variant_id: str | None = None):
        session_id = _validate_session_id(session_id)
        acquired = acquire_front_camera(
            "tryon_frontend",
            reason=f"try_on_start:{session_id}",
        )

        if not acquired.get("ok"):
            raise RuntimeError(acquired.get("error") or "try_on_unavailable")

        with self.lock:
            if self.active_session_id and self.active_session_id != session_id:
                previous = self.sessions.get(self.active_session_id)
                if previous is not None:
                    previous["active"] = False
                    previous["stoppedAt"] = _now_iso()
                    previous["stopReason"] = "session_replaced"

            session = {
                "sessionId": session_id,
                "catalogKey": catalog_key,
                "variantId": variant_id,
                "previewUrl": _preview_url(session_id),
                "streamType": "mjpeg",
                "active": True,
                "departedDuringTryon": False,
                "departureEvent": None,
                "startedAt": _now_iso(),
                "updatedAt": _now_iso(),
            }
            self.sessions[session_id] = session
            self.active_session_id = session_id
            return dict(session)

    def mark_departed(self, departure_event: dict | None = None):
        with self.lock:
            if not self.active_session_id:
                return None

            session = self.sessions.get(self.active_session_id)
            if session is None or not session.get("active"):
                return None

            session["departedDuringTryon"] = True
            session["departureEvent"] = departure_event
            session["updatedAt"] = _now_iso()
            return dict(session)

    def stop(self, session_id: str, reason: str | None = None):
        session_id = _validate_session_id(session_id)
        reason = reason or "client_stop"
        release_owner = False
        departed_during_tryon = False

        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = {
                    "sessionId": session_id,
                    "active": False,
                    "startedAt": None,
                }
                self.sessions[session_id] = session

            release_owner = self.active_session_id == session_id and bool(
                session.get("active")
            )
            departed_during_tryon = bool(session.get("departedDuringTryon"))
            session["active"] = False
            session["stoppedAt"] = _now_iso()
            session["updatedAt"] = session["stoppedAt"]
            session["stopReason"] = reason

            if self.active_session_id == session_id:
                self.active_session_id = None

        if release_owner:
            release_front_camera(
                "tryon_frontend",
                reason=f"try_on_stop:{session_id}:{reason}",
            )

        return {
            "sessionId": session_id,
            "reason": reason,
            "departedDuringTryon": departed_during_tryon,
            "shouldRefreshProfile": departed_during_tryon,
        }

    def is_active(self, session_id: str):
        with self.lock:
            session = self.sessions.get(session_id)
            return bool(session and session.get("active"))

    def status(self):
        with self.lock:
            active = self.sessions.get(self.active_session_id)
            return {
                "activeSessionId": self.active_session_id,
                "activeSession": dict(active) if active else None,
                "sessionCount": len(self.sessions),
            }


_try_on_sessions = TryOnSessionManager()


def start_try_on_session(session_id: str, catalog_key: str | None = None, variant_id: str | None = None):
    return _try_on_sessions.start(session_id, catalog_key=catalog_key, variant_id=variant_id)


def stop_try_on_session(session_id: str, reason: str | None = None):
    return _try_on_sessions.stop(session_id, reason=reason)


def mark_active_try_on_departed(departure_event: dict | None = None):
    return _try_on_sessions.mark_departed(departure_event=departure_event)


def is_try_on_session_active(session_id: str):
    try:
        session_id = _validate_session_id(session_id)
    except ValueError:
        return False

    return _try_on_sessions.is_active(session_id)


def get_try_on_status():
    return _try_on_sessions.status()


def iter_try_on_mjpeg(session_id: str, fps: float = 10.0, jpeg_quality: int = 80):
    session_id = _validate_session_id(session_id)
    delay = 1.0 / max(float(fps), 1.0)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    while is_try_on_session_active(session_id):
        with front_camera_io_lock():
            frame = read_camera("front", warmup_frames=1)

        ok, encoded = cv2.imencode(".jpg", frame, encode_params)

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

        time.sleep(delay)
