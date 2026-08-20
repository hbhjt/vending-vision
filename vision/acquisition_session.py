"""The bounded, single-owner acquisition module for one Try-On attempt.

The public seam is :meth:`AcquisitionSession.acquire`: it emits acquisition
facts and returns the exact checked frame selected for composition.  Camera
leases, preview fan-out, latest-frame scheduling and countdown bookkeeping are
deliberately implementation details of this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import cv2
import numpy as np


class AcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedFrame:
    frame: np.ndarray
    source: dict[str, Any]
    frame_id: str
    png: bytes
    digest: str
    width: int
    height: int

    def public(self, reference: str) -> dict[str, Any]:
        return {
            "reference": reference,
            "digest": self.digest,
            "contentType": "image/png",
            "byteSize": len(self.png),
            "width": self.width,
            "height": self.height,
            "frameId": self.frame_id,
        }


@dataclass(frozen=True)
class _Frame:
    value: np.ndarray
    source: dict[str, Any]
    sequence: int


class CapturedFrameStore:
    """Attempt-scoped immutable captured-frame capabilities.

    This is intentionally separate from preview and result storage: a preview
    token can never read a captured image, and successful result publication
    cannot overwrite the captured fact.
    """

    def __init__(
        self, *, max_count: int = 1000, ttl_seconds: float = 5 * 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_count = max_count
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[str, CapturedFrame, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    def _prune_unlocked(self) -> None:
        now = self._clock()
        for attempt_id, (_token, _captured, expires_at) in list(self._entries.items()):
            if expires_at <= now:
                self._entries.pop(attempt_id, None)
        while len(self._entries) > self._max_count:
            self._entries.popitem(last=False)

    async def admit(self, attempt_id: str, captured: CapturedFrame) -> str:
        token = secrets.token_urlsafe(32)
        async with self._lock:
            self._prune_unlocked()
            self._entries[attempt_id] = (token, captured, self._clock() + self._ttl_seconds)
            self._prune_unlocked()
        return token

    async def get(self, token: str) -> CapturedFrame | None:
        async with self._lock:
            self._prune_unlocked()
            for stored_token, captured, _expires_at in self._entries.values():
                if secrets.compare_digest(stored_token, token):
                    return captured
        return None

    async def discard(self, attempt_id: str) -> None:
        async with self._lock:
            self._entries.pop(attempt_id, None)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()


class AcquisitionSession:
    """Acquire one checked frame while independently keeping preview live."""

    def __init__(
        self,
        *,
        read_frame: Callable[[float], Awaitable[tuple[np.ndarray, dict[str, Any]]]],
        observe: Callable[[np.ndarray, float], Awaitable[Any]],
        preview_open: Callable[[str, bytes], Awaitable[str]],
        preview_update: Callable[[str, str, bytes], Awaitable[bool]],
        publish: Callable[[str, str, str, bool, int | None], Awaitable[None]],
        attempt_id: str,
        timeout_seconds: float = 30.0,
        stable_seconds: float = 3.0,
        preview_interval_seconds: float = 0.05,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._read_frame = read_frame
        self._observe = observe
        self._preview_open = preview_open
        self._preview_update = preview_update
        self._publish = publish
        self._attempt_id = attempt_id
        self._timeout_seconds = timeout_seconds
        self._stable_seconds = stable_seconds
        self._preview_interval_seconds = preview_interval_seconds
        self._clock = clock or asyncio.get_running_loop().time

    @staticmethod
    def _jpeg(frame: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise AcquisitionError("acquisition_preview_encode_failed")
        return encoded.tobytes()

    @staticmethod
    def _captured(frame: _Frame) -> CapturedFrame:
        ok, encoded = cv2.imencode(".png", frame.value)
        if not ok:
            raise AcquisitionError("acquisition_capture_encode_failed")
        png = encoded.tobytes()
        height, width = frame.value.shape[:2]
        # The identity is sourced from the Vision-owned decoded frame, never
        # an MJPEG snapshot or a caller-supplied DOM representation.
        frame_id = f"frame-{frame.sequence}-{hashlib.sha256(png).hexdigest()[:16]}"
        return CapturedFrame(
            frame=frame.value.copy(), source=dict(frame.source), frame_id=frame_id,
            png=png, digest=f"sha256:{hashlib.sha256(png).hexdigest()}",
            width=int(width), height=int(height),
        )

    async def acquire(
        self, *, manual_requested: Callable[[], Awaitable[bool]],
        consume_manual: Callable[[], Awaitable[None]],
    ) -> CapturedFrame:
        """Return only the final single-and-aligned inference frame.

        The producer is the sole camera reader.  Its one-element hand-off to
        inference is overwrite-only, so a blocked observer cannot accumulate
        stale work or stop tokenized preview updates.
        """
        started = self._clock()
        deadline = started + self._timeout_seconds
        latest: _Frame | None = None
        latest_changed = asyncio.Event()
        stop = asyncio.Event()
        preview_token: str | None = None
        preview_lock = asyncio.Lock()
        producer_error: BaseException | None = None

        async def producer() -> None:
            nonlocal latest, preview_token, producer_error
            sequence = 0
            try:
                while not stop.is_set() and self._clock() < deadline:
                    frame, source = await self._read_frame(max(0.001, deadline - self._clock()))
                    sequence += 1
                    candidate = _Frame(frame.copy(), dict(source or {}), sequence)
                    jpeg = self._jpeg(candidate.value)
                    async with preview_lock:
                        if preview_token is None:
                            preview_token = await self._preview_open(self._attempt_id, jpeg)
                        else:
                            await self._preview_update(self._attempt_id, preview_token, jpeg)
                    latest = candidate
                    latest_changed.set()
                    await asyncio.sleep(self._preview_interval_seconds)
            except BaseException as error:
                producer_error = error
                latest_changed.set()

        producer_task = asyncio.create_task(producer(), name=f"acquisition-preview:{self._attempt_id}")
        consumed = 0
        stable_started: float | None = None
        last_qualified: _Frame | None = None
        last_fact: tuple[str, bool, int | None] | None = None
        state: tuple[str, bool] = ("none", False)
        observed = False
        fact_lock = asyncio.Lock()

        async def publish_current() -> None:
            nonlocal last_fact
            if not observed:
                return
            occupancy, aligned = state
            now = self._clock()
            eligible = occupancy == "single" and aligned
            remaining = (
                None
                if stable_started is None
                else max(0, math.ceil((stable_started + self._stable_seconds - now) * 1000))
            )
            fact = (occupancy, aligned, remaining)
            async with fact_lock:
                if fact == last_fact:
                    return
                token = preview_token
                if token is None:
                    return
                await self._publish(
                    token,
                    occupancy,
                    "counting_down" if eligible else (
                        "no_person" if occupancy == "none" else "multiple_people" if occupancy == "multiple" else "align"
                    ),
                    aligned,
                    remaining,
                )
                last_fact = fact

        async def countdown_truth() -> None:
            # This independent publisher is the source of the live countdown
            # fact even when inference is temporarily slow.  It does not make
            # capture decisions: capture remains gated by a checked frame.
            while not stop.is_set() and self._clock() < deadline:
                await publish_current()
                await asyncio.sleep(min(0.1, self._preview_interval_seconds))

        ticker_task = asyncio.create_task(countdown_truth(), name=f"acquisition-countdown:{self._attempt_id}")
        try:
            while self._clock() < deadline:
                if producer_error is not None:
                    raise producer_error
                await asyncio.wait_for(latest_changed.wait(), timeout=max(0.001, deadline - self._clock()))
                latest_changed.clear()
                frame = latest
                if frame is None or frame.sequence == consumed:
                    continue
                consumed = frame.sequence
                observation = await self._observe(frame.value, max(0.001, deadline - self._clock()))
                occupancy, aligned = observation.occupancy, bool(observation.aligned)
                state = (occupancy, aligned)
                observed = True
                now = self._clock()
                eligible = occupancy == "single" and aligned
                if eligible:
                    last_qualified = frame
                    if stable_started is None:
                        stable_started = now
                else:
                    stable_started = None
                    last_qualified = None
                await publish_current()
                manual = await manual_requested()
                if manual and eligible:
                    await consume_manual()
                    return self._captured(frame)
                if stable_started is not None and now >= stable_started + self._stable_seconds:
                    if last_qualified is None:
                        continue
                    return self._captured(last_qualified)
            raise asyncio.TimeoutError()
        finally:
            stop.set()
            producer_task.cancel()
            ticker_task.cancel()
            await asyncio.gather(producer_task, ticker_task, return_exceptions=True)
