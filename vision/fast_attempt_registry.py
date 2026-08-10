"""Process-wide ownership and replay state for Fast try-on attempts.

The registry deliberately has one small async transition gate.  It makes
replacement wait for the old worker (and every resource it owns) before the
next attempt becomes visible, while still leaving network and image work
outside the gate.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AttemptReceipt:
    attempt_id: str
    owner_token: str
    generation: int


@dataclass
class AttemptSubscriber:
    key: int
    websocket: Any
    send_lock: asyncio.Lock


@dataclass
class ActiveAttempt:
    receipt: AttemptReceipt
    task: asyncio.Task
    cancel_event: asyncio.Event
    subscribers: dict[int, AttemptSubscriber] = field(default_factory=dict)
    accepted: dict | None = None
    latest_progress: dict | None = None


@dataclass
class TerminalAttempt:
    message: dict
    result: dict | None
    terminal_at: float


@dataclass
class AttemptAdmission:
    receipt: AttemptReceipt | None
    replay: list[dict]

    @property
    def is_owner(self) -> bool:
        return self.receipt is not None


class FastAttemptRegistry:
    """One active attempt plus bounded canonical terminal replay records."""

    def __init__(self, *, terminal_ttl_seconds: float, terminal_max_count: int = 32):
        self._transition_gate = asyncio.Lock()
        self._gate = asyncio.Lock()
        self._active: ActiveAttempt | None = None
        self._terminals: OrderedDict[str, TerminalAttempt] = OrderedDict()
        self._generation = 0
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._terminal_max_count = terminal_max_count

    def _prune_unlocked(self) -> None:
        cutoff = time.monotonic() - self._terminal_ttl_seconds
        while self._terminals:
            attempt_id, terminal = next(iter(self._terminals.items()))
            if terminal.terminal_at > cutoff:
                break
            self._terminals.pop(attempt_id)
        while len(self._terminals) > self._terminal_max_count:
            self._terminals.popitem(last=False)

    async def admit(
        self,
        *,
        attempt_id: str,
        websocket: Any,
        send_lock: asyncio.Lock,
        task: asyncio.Task,
        accepted: dict,
        progress: dict,
    ) -> AttemptAdmission:
        """Attach to a canonical attempt or admit one after replacement joins."""
        subscriber = AttemptSubscriber(id(websocket), websocket, send_lock)
        async with self._transition_gate:
            async with self._gate:
                self._prune_unlocked()
                terminal = self._terminals.get(attempt_id)
                if terminal is not None:
                    self._terminals.move_to_end(attempt_id)
                    return AttemptAdmission(None, [terminal.message])

                active = self._active
                if active is not None and active.receipt.attempt_id == attempt_id:
                    active.subscribers[subscriber.key] = subscriber
                    replay = [message for message in (active.accepted, active.latest_progress) if message]
                    return AttemptAdmission(None, replay)
                if active is not None:
                    active.cancel_event.set()

            if active is not None and active.task is not task:
                # Do not admit new work into the single worker lane until the
                # old task has closed its response and joined executor work.
                # The state lock is deliberately not held while joining.
                await asyncio.shield(active.task)

            async with self._gate:
                if active is not None and self._active is active:
                    # A task never intentionally reaches this fallback, but a
                    # cancellation injected before its terminal CAS must still
                    # not let a replacement inherit live resources.
                    self._active = None
                self._generation += 1
                receipt = AttemptReceipt(attempt_id, uuid4().hex, self._generation)
                self._active = ActiveAttempt(
                    receipt=receipt,
                    task=task,
                    cancel_event=asyncio.Event(),
                    subscribers={subscriber.key: subscriber},
                    accepted=accepted,
                    latest_progress=progress,
                )
                return AttemptAdmission(receipt, [accepted, progress])

    async def is_current(self, receipt: AttemptReceipt) -> bool:
        async with self._gate:
            active = self._active
            return active is not None and active.receipt == receipt and not active.cancel_event.is_set()

    async def cancel_event_for(self, receipt: AttemptReceipt) -> asyncio.Event:
        async with self._gate:
            active = self._active
            if active is not None and active.receipt == receipt:
                return active.cancel_event
        canceled = asyncio.Event()
        canceled.set()
        return canceled

    async def cancel_owner_and_join(self, receipt: AttemptReceipt) -> None:
        async with self._gate:
            active = self._active
            if active is None or active.receipt != receipt:
                return
            active.cancel_event.set()
            task = active.task
        if task is not asyncio.current_task():
            await asyncio.shield(task)

    async def detach_subscriber(self, websocket: Any) -> None:
        async with self._gate:
            if self._active is not None:
                self._active.subscribers.pop(id(websocket), None)

    async def commit_terminal(
        self,
        receipt: AttemptReceipt,
        message: dict,
        result: dict | None = None,
    ) -> list[AttemptSubscriber]:
        """Atomically publish the one canonical terminal and its result grant."""
        async with self._gate:
            active = self._active
            if active is None or active.receipt != receipt:
                return []
            if active.cancel_event.is_set() and message.get("type") == "vision.try_on.attempt.completed":
                return []
            self._terminals[receipt.attempt_id] = TerminalAttempt(
                message=message,
                result=result,
                terminal_at=time.monotonic(),
            )
            self._terminals.move_to_end(receipt.attempt_id)
            subscribers = list(active.subscribers.values())
            self._active = None
            self._prune_unlocked()
            return subscribers

    async def get_result(self, attempt_id: str, token: str) -> dict | None:
        async with self._gate:
            self._prune_unlocked()
            terminal = self._terminals.get(attempt_id)
            if terminal is None or terminal.result is None:
                return None
            if terminal.result.get("token") != token:
                return None
            self._terminals.move_to_end(attempt_id)
            return terminal.result

    async def shutdown(self) -> None:
        async with self._gate:
            active = self._active
            if active is None:
                return
            active.cancel_event.set()
            task = active.task
        if task is not asyncio.current_task():
            await asyncio.shield(task)
