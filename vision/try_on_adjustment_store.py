"""Bounded attempt-scoped retention of the Try-On source for locked-center re-scaling.

Re-rendering a completed Try-On result at another garment scale needs the
original source frame and garment bytes, which the stored result PNG cannot
provide. This store keeps exactly that snapshot, bounded in count and time,
and never exposes it to the Machine: reads happen only inside the Vision
process and produce a replacement result.

The snapshot is deliberately bound to the Try-On result TTL and is additionally
dropped when the Machine closes the owning socket after route leave or its
stable customer-departure edge, so one customer's source cannot outlive their
interaction.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TryOnAdjustmentSnapshot:
    attempt_id: str
    frame: Any
    garment_png: bytes
    garment_digest: str
    template: str
    inserted_at: float
    expires_at: float


class TryOnAdjustmentStore:
    def __init__(
        self,
        *,
        max_count: int = 8,
        ttl_seconds: float = 5 * 60,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_count < 1 or ttl_seconds <= 0:
            raise ValueError("invalid Try-On adjustment store limits")
        self.max_count = int(max_count)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._snapshots: OrderedDict[str, TryOnAdjustmentSnapshot] = OrderedDict()

    def _prune_expired(self) -> None:
        now = self._clock()
        for attempt_id, snapshot in list(self._snapshots.items()):
            if snapshot.expires_at <= now:
                self._snapshots.pop(attempt_id, None)

    def admit(
        self,
        attempt_id: str,
        frame: Any,
        garment_png: bytes,
        garment_digest: str,
        template: str,
    ) -> None:
        self._prune_expired()
        now = self._clock()
        self._snapshots[attempt_id] = TryOnAdjustmentSnapshot(
            attempt_id=attempt_id,
            frame=frame,
            garment_png=bytes(garment_png),
            garment_digest=garment_digest,
            template=template,
            inserted_at=now,
            expires_at=now + self.ttl_seconds,
        )
        self._snapshots.move_to_end(attempt_id)
        while len(self._snapshots) > self.max_count:
            self._snapshots.popitem(last=False)

    def get(self, attempt_id: str) -> TryOnAdjustmentSnapshot | None:
        self._prune_expired()
        return self._snapshots.get(attempt_id)

    def discard(self, attempt_id: str) -> None:
        self._snapshots.pop(attempt_id, None)

    def discard_all(self) -> None:
        self._snapshots.clear()

    @property
    def count(self) -> int:
        self._prune_expired()
        return len(self._snapshots)
