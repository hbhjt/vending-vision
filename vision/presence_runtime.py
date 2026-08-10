"""Lightweight top-camera presence runtime.

This module deliberately owns the mutable top-camera detector state.  Profile
sampling is a separate, slower activity; it consumes a candidate emitted here
but never blocks presence or departure events.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from uuid import uuid4

from vision.config import settings
from vision.profile_messages import build_presence_status, profile_update
from vision.profile_sampling import estimate_ambient_light
from vision.profile_state import (
    ensure_active_track,
    get_departure_tracker,
    get_occupancy_gate,
    mark_active_track_missing,
    protocol_occupancy_snapshot,
    ResponsiveOccupancyFilter,
    signature_match_score,
    target_signature_from_proximity,
)
from vision.proximity import ProximityMonitor
from vision.session_state import (
    mark_vision_session_departed,
    mark_vision_session_presence,
    reset_vision_session,
)


@dataclass
class ProfileCandidate:
    generation: int
    event_id: str
    proximity: dict
    occupancy: dict
    ambient_light: dict | None
    track: object


@dataclass
class PresencePollResult:
    update: dict | None
    candidate: ProfileCandidate | None
    snapshot: dict


class PresenceRuntime:
    """Owns top-camera polling, state edges and profile-candidate validity."""

    def __init__(self, monitor: ProximityMonitor | None = None):
        self.monitor = monitor or ProximityMonitor()
        self.lock = threading.RLock()
        self.latest_snapshot: dict | None = None
        self.collection_generation: int | None = None
        self.next_generation = 0
        self.retry_after = 0.0
        self.profiled_signature: dict | None = None
        self.target_change_streak = 0
        self.occupancy_filter = ResponsiveOccupancyFilter()

    def latest(self):
        with self.lock:
            return copy.deepcopy(self.latest_snapshot)

    def _invalidate_collection_locked(self):
        self.collection_generation = None

    def is_candidate_valid(self, generation: int) -> bool:
        with self.lock:
            snapshot = self.latest_snapshot or {}
            return bool(
                self.collection_generation == generation
                and snapshot.get("occupancy", {}).get("state") == "single"
                and get_occupancy_gate().can_trigger()
            )

    def finish_collection(self, generation: int, pushed: bool = False):
        with self.lock:
            if self.collection_generation == generation:
                self.collection_generation = None
            if pushed:
                self.profiled_signature = copy.deepcopy(
                    (self.latest_snapshot or {}).get("targetSignature")
                )
                self.target_change_streak = 0
            else:
                # Avoid immediately re-running a costly collection after a
                # quality failure while preserving the protocol's approach state.
                self.retry_after = time.monotonic() + 0.5

    def poll(self, include_status: bool, include_ambient_light: bool, include_departure: bool):
        """Read exactly one top frame and produce a lightweight protocol update."""
        proximity, image, source_frame = self.monitor.check_once(
            return_image=True,
            camera_role="top",
            return_source=True,
        )
        ambient_light = estimate_ambient_light(image) if include_ambient_light else None
        occupancy = self.occupancy_filter.update(
            proximity,
            protocol_occupancy_snapshot(proximity),
        )
        event_id = f"vision-event-{uuid4()}"
        now = time.monotonic()

        with self.lock:
            snapshot = {
                "eventId": event_id,
                "proximity": copy.deepcopy(proximity),
                "occupancy": copy.deepcopy(occupancy),
                "ambientLight": copy.deepcopy(ambient_light),
                "polledAt": time.time(),
            }
            self.latest_snapshot = snapshot

            gate = get_occupancy_gate()
            departure_tracker = get_departure_tracker()
            state = occupancy["state"]
            candidate = None

            if state == "none":
                gate.mark_absent()
                mark_active_track_missing()
                self._invalidate_collection_locked()
                departure = departure_tracker.mark_absent(
                    reason="no_person", ambient_light=ambient_light,
                )
                if departure is not None:
                    self.profiled_signature = None
                    self.target_change_streak = 0
                    mark_vision_session_departed(departure)
                    if include_departure:
                        departure = dict(departure)
                        departure["source"] = "top"
                        if source_frame is not None:
                            departure["sourceFrame"] = source_frame
                        return PresencePollResult(
                            profile_update("vision.person_departed", departure),
                            None,
                            copy.deepcopy(snapshot),
                        )

                if include_status:
                    return PresencePollResult(
                        profile_update(
                            "vision.presence_status",
                            build_presence_status(
                                event_id=event_id,
                                state="empty",
                                reason="no_person",
                                proximity=proximity,
                                occupancy=occupancy,
                                ambient_light=ambient_light,
                                source="top",
                                source_frame=source_frame,
                            ),
                        ),
                        None,
                        copy.deepcopy(snapshot),
                    )
                return PresencePollResult(None, None, copy.deepcopy(snapshot))

            gate.mark_present()
            departure_tracker.mark_present()
            signature = target_signature_from_proximity(proximity)
            snapshot["targetSignature"] = copy.deepcopy(signature)
            track = ensure_active_track(signature, "single" if state == "single" else "approach")

            if (
                state == "single"
                and gate.public_state().get("state") == "occupied"
                and self.profiled_signature
                and signature
            ):
                score = signature_match_score(self.profiled_signature, signature)
                if score < settings.PROFILE_TRACK_MIN_MATCH_SCORE:
                    self.target_change_streak += 1
                else:
                    self.target_change_streak = 0
                if self.target_change_streak >= 2:
                    gate.mark_target_changed()
                    reset_vision_session(reason="stable_target_changed")
                    self.profiled_signature = None
                    self.target_change_streak = 0
            elif state != "single":
                self.target_change_streak = 0

            if state == "multiple":
                self._invalidate_collection_locked()
                mark_vision_session_presence(
                    "multiple", reason="multiple_people_detected",
                    proximity=proximity, occupancy=occupancy,
                )
                status_state, reason = "occupied", "multiple_people_detected"
            elif state == "unknown":
                self._invalidate_collection_locked()
                mark_vision_session_presence(
                    "waiting_front_camera", reason="top_occupancy_unknown",
                    proximity=proximity, occupancy=occupancy,
                )
                status_state, reason = "waiting", "top_occupancy_unknown"
            elif not gate.can_trigger():
                mark_vision_session_presence(
                    "profile_pushed", reason="occupancy_gate_locked",
                    proximity=proximity, occupancy=occupancy,
                )
                status_state, reason = "occupied", "occupancy_gate_locked"
            else:
                close_enough = bool(proximity.get("close"))
                reason = (
                    "single_person_profile_pending"
                    if close_enough
                    else "person_present_but_not_close"
                )
                mark_vision_session_presence(
                    "approach_detected", reason=reason,
                    proximity=proximity, occupancy=occupancy,
                )
                status_state = "approach"
                # Stable single-person occupancy starts front-camera pre-sampling
                # immediately.  The close flag only controls whether sampling may
                # finish before the one-second pre-sampling floor.
                if self.collection_generation is None and now >= self.retry_after:
                    self.next_generation += 1
                    self.collection_generation = self.next_generation
                    candidate = ProfileCandidate(
                        generation=self.collection_generation,
                        event_id=event_id,
                        proximity=copy.deepcopy(proximity),
                        occupancy=copy.deepcopy(occupancy),
                        ambient_light=copy.deepcopy(ambient_light),
                        track=track,
                    )

            update = None
            if include_status:
                update = profile_update(
                    "vision.presence_status",
                    build_presence_status(
                        event_id=event_id,
                        state=status_state,
                        reason=reason,
                        proximity=proximity,
                        tracking=track.public_state(),
                        occupancy=occupancy,
                        ambient_light=ambient_light,
                        source="top",
                        source_frame=source_frame,
                    ),
                )

            return PresencePollResult(update, candidate, copy.deepcopy(snapshot))


_runtime = None
_runtime_lock = threading.RLock()


def get_presence_runtime() -> PresenceRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = PresenceRuntime()
        return _runtime
