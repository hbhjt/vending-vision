import copy
import threading
import time
from uuid import uuid4

from vision.protocol import now_iso


ACTIVE_STATES = {
    "approach_detected",
    "waiting_front_camera",
    "profiling",
    "profile_pushed",
    "browsing",
    "tryon_active",
    "multiple",
    "unusable",
}


def _age_ms(started_monotonic):
    if started_monotonic is None:
        return None

    return int(max(time.time() - started_monotonic, 0.0) * 1000)


class VisionSessionState:
    def __init__(self):
        self.lock = threading.RLock()
        self.active_session = None
        self.last_session = None
        self.session_count = 0

    def _new_session_locked(self, reason=None):
        now = now_iso()
        self.session_count += 1
        self.active_session = {
            "sessionId": f"vision-session-{uuid4()}",
            "state": "approach_detected",
            "reason": reason or "person_present",
            "profilePushed": False,
            "profileEventId": None,
            "profilePayload": None,
            "tryOnSessionId": None,
            "departedDuringTryon": False,
            "departureEvent": None,
            "startedAt": now,
            "updatedAt": now,
            "lastPresenceAt": now,
            "startedMonotonic": time.time(),
        }
        return self.active_session

    def _ensure_active_locked(self, reason=None):
        if self.active_session is None:
            return self._new_session_locked(reason=reason)

        return self.active_session

    def _summary_locked(self, session):
        if not session:
            return None

        profile_payload = session.get("profilePayload") or {}
        quality = profile_payload.get("quality") or {}
        profile_cache = None

        if profile_payload:
            profile_cache = {
                "eventId": profile_payload.get("eventId"),
                "detectedAt": profile_payload.get("detectedAt"),
                "profile": copy.deepcopy(profile_payload.get("profile")),
                "quality": {
                    "overall": quality.get("overall"),
                    "profileUsable": quality.get("profileUsable"),
                    "warnings": copy.deepcopy(quality.get("warnings", [])),
                    "validFrameCount": quality.get("validFrameCount"),
                    "sampleCount": quality.get("sampleCount"),
                },
            }

        return {
            "sessionId": session.get("sessionId"),
            "state": session.get("state"),
            "reason": session.get("reason"),
            "profilePushed": bool(session.get("profilePushed")),
            "profileEventId": session.get("profileEventId"),
            "tryOnSessionId": session.get("tryOnSessionId"),
            "departedDuringTryon": bool(session.get("departedDuringTryon")),
            "startedAt": session.get("startedAt"),
            "updatedAt": session.get("updatedAt"),
            "lastPresenceAt": session.get("lastPresenceAt"),
            "ageMs": _age_ms(session.get("startedMonotonic")),
            "profileCache": profile_cache,
        }

    def status(self):
        with self.lock:
            active = self._summary_locked(self.active_session)
            return {
                "state": active["state"] if active else "empty",
                "activeSessionId": active["sessionId"] if active else None,
                "activeSession": active,
                "lastSession": self._summary_locked(self.last_session),
                "sessionCount": self.session_count,
            }

    def active_summary(self):
        with self.lock:
            return self._summary_locked(self.active_session)

    def payload_summary(self):
        with self.lock:
            active = self._summary_locked(self.active_session)

            if active:
                return active

            last = self._summary_locked(self.last_session)
            return {
                "sessionId": None,
                "state": "empty",
                "profilePushed": False,
                "departedDuringTryon": False,
                "lastSessionId": last.get("sessionId") if last else None,
            }

    def mark_presence(self, state, reason=None, proximity=None, occupancy=None):
        with self.lock:
            session = self._ensure_active_locked(reason=reason)
            session["state"] = state if state in ACTIVE_STATES else "approach_detected"
            session["reason"] = reason
            session["lastPresenceAt"] = now_iso()
            session["updatedAt"] = session["lastPresenceAt"]
            session["proximity"] = copy.deepcopy(proximity)
            session["occupancy"] = copy.deepcopy(occupancy)
            return self._summary_locked(session)

    def mark_waiting_front_camera(self, reason=None, owner_status=None):
        with self.lock:
            session = self._ensure_active_locked(reason=reason)
            session["state"] = "waiting_front_camera"
            session["reason"] = reason or "front_camera_busy"
            session["frontCamera"] = copy.deepcopy(owner_status)
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_profiling(self, reason=None):
        with self.lock:
            session = self._ensure_active_locked(reason=reason)
            session["state"] = "profiling"
            session["reason"] = reason or "front_profile_sampling"
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_unusable(self, reason=None):
        with self.lock:
            session = self._ensure_active_locked(reason=reason)
            session["state"] = "unusable"
            session["reason"] = reason or "front_profile_unusable"
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_profile_pushed(self, payload):
        with self.lock:
            session = self._ensure_active_locked(reason="profile_pushed")
            clean_payload = copy.deepcopy(payload)
            clean_payload.pop("session", None)
            session["state"] = "profile_pushed"
            session["reason"] = "profile_pushed"
            session["profilePushed"] = True
            session["profileEventId"] = clean_payload.get("eventId")
            session["profilePayload"] = clean_payload
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_tryon_started(self, tryon_session):
        with self.lock:
            session = self._ensure_active_locked(reason="tryon_started")
            session["state"] = "tryon_active"
            session["reason"] = "tryon_started"
            session["tryOnSessionId"] = tryon_session.get("sessionId")
            session["departedDuringTryon"] = False
            session["departureEvent"] = None
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_tryon_departed(self, departure_event=None):
        with self.lock:
            session = self._ensure_active_locked(reason="departed_during_tryon")
            session["state"] = "tryon_active"
            session["reason"] = "departed_during_tryon"
            session["departedDuringTryon"] = True
            session["departureEvent"] = copy.deepcopy(departure_event)
            session["updatedAt"] = now_iso()
            return self._summary_locked(session)

    def mark_tryon_stopped(self, stopped):
        with self.lock:
            if self.active_session is None:
                return None

            session = self.active_session
            session["tryOnSessionId"] = None
            session["updatedAt"] = now_iso()

            if stopped.get("shouldRefreshProfile"):
                session["state"] = "departed"
                session["reason"] = "departed_during_tryon"
                self.last_session = copy.deepcopy(session)
                self.active_session = None
                return self._summary_locked(self.last_session)

            if session.get("profilePushed"):
                session["state"] = "profile_pushed"
                session["reason"] = "tryon_stopped_keep_profile"
            else:
                session["state"] = "approach_detected"
                session["reason"] = "tryon_stopped_wait_profile"

            return self._summary_locked(session)

    def mark_departed(self, departure_event=None):
        with self.lock:
            if self.active_session is None:
                return None

            session = self.active_session
            session["state"] = "departed"
            session["reason"] = (
                departure_event.get("reason")
                if isinstance(departure_event, dict)
                else "departed"
            )
            session["departureEvent"] = copy.deepcopy(departure_event)
            session["updatedAt"] = now_iso()
            self.last_session = copy.deepcopy(session)
            self.active_session = None
            return self._summary_locked(self.last_session)

    def reset(self, reason=None):
        with self.lock:
            if self.active_session is not None:
                self.active_session["state"] = "empty"
                self.active_session["reason"] = reason or "reset"
                self.active_session["updatedAt"] = now_iso()
                self.last_session = copy.deepcopy(self.active_session)

            self.active_session = None
            return self.status()


_vision_session_state = VisionSessionState()


def get_vision_session_status():
    return _vision_session_state.status()


def get_vision_session_payload():
    return _vision_session_state.payload_summary()


def mark_vision_session_presence(state, reason=None, proximity=None, occupancy=None):
    return _vision_session_state.mark_presence(
        state,
        reason=reason,
        proximity=proximity,
        occupancy=occupancy,
    )


def mark_vision_session_waiting_front_camera(reason=None, owner_status=None):
    return _vision_session_state.mark_waiting_front_camera(
        reason=reason,
        owner_status=owner_status,
    )


def mark_vision_session_profiling(reason=None):
    return _vision_session_state.mark_profiling(reason=reason)


def mark_vision_session_unusable(reason=None):
    return _vision_session_state.mark_unusable(reason=reason)


def mark_vision_session_profile_pushed(payload):
    return _vision_session_state.mark_profile_pushed(payload)


def mark_vision_session_tryon_started(tryon_session):
    return _vision_session_state.mark_tryon_started(tryon_session)


def mark_vision_session_tryon_departed(departure_event=None):
    return _vision_session_state.mark_tryon_departed(
        departure_event=departure_event,
    )


def mark_vision_session_tryon_stopped(stopped):
    return _vision_session_state.mark_tryon_stopped(stopped)


def mark_vision_session_departed(departure_event=None):
    return _vision_session_state.mark_departed(departure_event=departure_event)


def reset_vision_session(reason=None):
    return _vision_session_state.reset(reason=reason)
