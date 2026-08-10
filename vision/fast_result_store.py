"""Bounded, attempt-scoped storage for generated Fast result bytes.

The store deliberately has a very small interface.  Entries are immutable
snapshots and reads never turn into writes: in particular a read cannot extend
the retention window or change eviction order.  The registry owns the async
state gate; ``*_unlocked`` methods are used while that gate is held so result
admission and terminal publication can be one commit.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping


class ResultAdmissionError(RuntimeError):
    """A result cannot be admitted without changing the existing store."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ResultEntry:
    attempt_id: str
    token: str
    bytes: bytes
    reference: str
    digest: str
    content_type: str
    byte_size: int
    width: int
    height: int
    inserted_at: float
    expires_at: float

    def public(self) -> dict:
        return {
            "reference": self.reference,
            "digest": self.digest,
            "contentType": self.content_type,
            "byteSize": self.byte_size,
            "width": self.width,
            "height": self.height,
        }

    def stored(self) -> dict:
        value = self.public()
        value.update({"token": self.token, "bytes": self.bytes})
        return value


@dataclass(frozen=True)
class ResultAdmission:
    """The immutable entry committed by one admission and every displaced ID."""

    entry: ResultEntry
    evicted_attempt_ids: tuple[str, ...]


class FastResultStore:
    """An in-memory bounded result store with atomic admission planning."""

    def __init__(
        self,
        *,
        max_count: int = 1000,
        max_bytes: int = 256 * 1024 * 1024,
        single_max_bytes: int | None = None,
        ttl_seconds: float = 5 * 60,
        clock: Callable[[], float] = time.monotonic,
    ):
        single_limit = max_bytes if single_max_bytes is None else int(single_max_bytes)
        if max_count < 1 or max_bytes < 1 or single_limit < 1 or ttl_seconds <= 0:
            raise ValueError("invalid Fast result store limits")
        self.max_count = int(max_count)
        self.max_bytes = int(max_bytes)
        self.single_max_bytes = single_limit
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[str, ResultEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def aggregate_bytes(self) -> int:
        """Compatibility name for the aggregate encoded-byte counter."""
        return self._total_bytes

    def _assert_invariants(self) -> None:
        assert self._total_bytes == sum(entry.byte_size for entry in self._entries.values())
        assert self._total_bytes >= 0
        assert len(self._entries) >= 0

    def _remove_unlocked(self, attempt_id: str) -> ResultEntry | None:
        entry = self._entries.pop(attempt_id, None)
        if entry is not None:
            self._total_bytes -= entry.byte_size
        return entry

    def _prune_unlocked(self, now: float | None = None) -> tuple[str, ...]:
        now = self._clock() if now is None else now
        evicted: list[str] = []
        # Iterate over every entry.  Expiry is not an LRU property and an old
        # entry may be behind a recently inserted one.
        for attempt_id, entry in list(self._entries.items()):
            if entry.expires_at <= now:
                self._remove_unlocked(attempt_id)
                evicted.append(attempt_id)
        self._assert_invariants()
        return tuple(evicted)

    def _entry_from_result(
        self, attempt_id: str, result: Mapping[str, object], now: float
    ) -> ResultEntry:
        image = result.get("bytes")
        token = result.get("token")
        if not isinstance(image, bytes) or not image:
            raise ResultAdmissionError("result_store_invalid_bytes")
        if not isinstance(token, str) or not token:
            raise ResultAdmissionError("result_store_invalid_token")
        values = {
            "reference": result.get("reference"),
            "digest": result.get("digest"),
            "content_type": result.get("contentType"),
            "width": result.get("width"),
            "height": result.get("height"),
        }
        if not isinstance(values["reference"], str) or not isinstance(values["digest"], str):
            raise ResultAdmissionError("result_store_invalid_metadata")
        if values["content_type"] != "image/png":
            raise ResultAdmissionError("result_store_invalid_metadata")
        if not all(isinstance(values[key], int) and values[key] > 0 for key in ("width", "height")):
            raise ResultAdmissionError("result_store_invalid_metadata")
        immutable_bytes = bytes(image)
        return ResultEntry(
            attempt_id=attempt_id,
            token=token,
            bytes=immutable_bytes,
            reference=values["reference"],
            digest=values["digest"],
            content_type="image/png",
            byte_size=len(immutable_bytes),
            width=values["width"],
            height=values["height"],
            inserted_at=now,
            expires_at=now + self.ttl_seconds,
        )

    def _admit_unlocked(
        self,
        attempt_id: str,
        result: Mapping[str, object],
        *,
        now: float | None = None,
    ) -> ResultAdmission:
        """Plan evictions, then commit one result atomically.

        The single-result limit is checked before expiry cleanup or any store
        mutation.  A failed overwrite therefore leaves the previous entry
        byte-for-byte intact.
        """
        image = result.get("bytes")
        if not isinstance(image, bytes) or not image:
            raise ResultAdmissionError("result_store_invalid_bytes")
        if len(image) > self.single_max_bytes:
            raise ResultAdmissionError("result_store_too_large")

        now = self._clock() if now is None else now
        candidate = self._entry_from_result(attempt_id, result, now)
        # Validate and calculate the full transaction before changing the
        # store.  A rejected candidate cannot silently prune or evict a live
        # grant, which is essential when the registry has a matching terminal
        # replay record.
        expired_ids = tuple(
            stored_id
            for stored_id, entry in self._entries.items()
            if entry.expires_at <= now
        )
        expired = set(expired_ids)
        existing = (
            self._entries.get(attempt_id)
            if attempt_id not in expired
            else None
        )
        # Build the candidate totals without touching the live map.
        retained_count = len(self._entries) - len(expired_ids)
        retained_bytes = self._total_bytes - sum(
            self._entries[stored_id].byte_size for stored_id in expired_ids
        )
        candidate_count = retained_count - (1 if existing is not None else 0) + 1
        candidate_bytes = (
            retained_bytes - (existing.byte_size if existing else 0) + candidate.byte_size
        )
        if candidate_count <= self.max_count and candidate_bytes <= self.max_bytes:
            evictions: list[str] = []
        else:
            evictions = []
            remaining_count = candidate_count
            remaining_bytes = candidate_bytes
            for old_id, old in self._entries.items():
                if old_id in expired or old_id == attempt_id:
                    continue
                if remaining_count <= self.max_count and remaining_bytes <= self.max_bytes:
                    break
                evictions.append(old_id)
                remaining_count -= 1
                remaining_bytes -= old.byte_size
            if remaining_count > self.max_count or remaining_bytes > self.max_bytes:
                raise ResultAdmissionError("result_store_capacity")

        # Commit exactly the plan.  No OLD entry is removed until NEW is known
        # to fit; this also makes overwrites all-or-nothing.  The caller needs
        # the precise eviction IDs to retire matching terminal replay records
        # in the same registry transition.
        for old_id in (*expired_ids, *evictions):
            self._remove_unlocked(old_id)
        self._remove_unlocked(attempt_id)
        self._entries[attempt_id] = candidate
        self._total_bytes += candidate.byte_size
        self._assert_invariants()
        return ResultAdmission(
            entry=candidate,
            evicted_attempt_ids=tuple((*expired_ids, *evictions)),
        )

    async def admit(
        self, attempt_id: str, result: Mapping[str, object]
    ) -> ResultAdmission:
        async with self._lock:
            return self._admit_unlocked(attempt_id, result)

    def _get_unlocked(self, attempt_id: str, token: str) -> ResultEntry | None:
        self._prune_unlocked()
        entry = self._entries.get(attempt_id)
        if entry is None or not secrets.compare_digest(entry.token, token):
            return None
        # Deliberately do not move the entry or refresh expires_at.
        return entry

    async def get(self, attempt_id: str, token: str) -> ResultEntry | None:
        async with self._lock:
            return self._get_unlocked(attempt_id, token)

    def snapshot(self) -> tuple[ResultEntry, ...]:
        self._prune_unlocked()
        return tuple(self._entries.values())
