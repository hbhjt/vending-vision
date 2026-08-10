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
    owner_receipts: set[AttemptReceipt] | None = None
    owner_subscriber_key: int | None = None
    subscribers: dict[int, AttemptSubscriber] = field(default_factory=dict)
    accepted: dict | None = None
    latest_progress: dict | None = None
    canceled_terminal: dict | None = None


@dataclass
class TerminalAttempt:
    message: dict
    result: dict | None
    terminal_at: float


@dataclass
class AttemptAdmission:
    receipt: AttemptReceipt | None
    replay: list[dict]
    transitions: list[TerminalTransition] = field(default_factory=list)

    @property
    def is_owner(self) -> bool:
        return self.receipt is not None


@dataclass
class TerminalTransition:
    """The sole winning terminal plus the live subscribers to notify."""

    message: dict
    subscribers: list[AttemptSubscriber]


class FastAttemptRegistry:
    """One active attempt plus bounded canonical terminal replay records.

    Replays retain the newest ``terminal_max_count`` terminal records for the
    configured TTL.  Once a record expires, the attempt ID is intentionally
    eligible for a new attempt; retained records always replay byte-for-byte.
    Live subscriber retention is also bounded.  An evicted idle subscriber can
    submit the same ID again and receives that same canonical terminal once it
    exists.
    """

    def __init__(
        self,
        *,
        terminal_ttl_seconds: float,
        terminal_max_count: int = 32,
        subscriber_max_count: int = 32,
    ):
        self._transition_gate = asyncio.Lock()
        self._gate = asyncio.Lock()
        self._active: ActiveAttempt | None = None
        self._terminals: OrderedDict[str, TerminalAttempt] = OrderedDict()
        self._generation = 0
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._terminal_max_count = terminal_max_count
        self._subscriber_max_count = subscriber_max_count

    def _attach_subscriber_unlocked(self, active: ActiveAttempt, subscriber: AttemptSubscriber) -> None:
        if subscriber.key in active.subscribers:
            active.subscribers[subscriber.key] = subscriber
            return
        while len(active.subscribers) >= self._subscriber_max_count:
            stale_key = next(
                (key for key in active.subscribers if key != active.owner_subscriber_key),
                None,
            )
            if stale_key is None:
                break
            active.subscribers.pop(stale_key, None)
        if len(active.subscribers) < self._subscriber_max_count:
            active.subscribers[subscriber.key] = subscriber

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
        accepted: dict | None,
        progress: dict | None,
        canceled_terminal: dict | None = None,
        owner_receipts: set[AttemptReceipt] | None = None,
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
                    self._attach_subscriber_unlocked(active, subscriber)
                    replay = [message for message in (active.accepted, active.latest_progress) if message]
                    return AttemptAdmission(None, replay)
                transitions: list[TerminalTransition] = []
                if active is not None:
                    active.cancel_event.set()
                    if active.canceled_terminal is not None:
                        transitions.append(
                            self._commit_terminal_unlocked(active, active.canceled_terminal)
                        )

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
                if owner_receipts is not None:
                    owner_receipts.add(receipt)
                self._active = ActiveAttempt(
                    receipt=receipt,
                    task=task,
                    cancel_event=asyncio.Event(),
                    owner_receipts=owner_receipts,
                    owner_subscriber_key=subscriber.key,
                    subscribers={subscriber.key: subscriber},
                    accepted=accepted,
                    latest_progress=progress,
                    canceled_terminal=canceled_terminal,
                )
                return AttemptAdmission(
                    receipt,
                    [message for message in (accepted, progress) if message],
                    transitions,
                )

    async def is_current(self, receipt: AttemptReceipt) -> bool:
        async with self._gate:
            active = self._active
            return active is not None and active.receipt == receipt and not active.cancel_event.is_set()

    async def reject_or_replay(
        self,
        *,
        attempt_id: str,
        websocket: Any,
        send_lock: asyncio.Lock,
        terminal: dict,
    ) -> AttemptAdmission:
        """Bounded-admission fallback with stable terminal replay semantics."""
        subscriber = AttemptSubscriber(id(websocket), websocket, send_lock)
        async with self._gate:
            self._prune_unlocked()
            stored = self._terminals.get(attempt_id)
            if stored is not None:
                self._terminals.move_to_end(attempt_id)
                return AttemptAdmission(None, [stored.message])
            active = self._active
            if active is not None and active.receipt.attempt_id == attempt_id:
                self._attach_subscriber_unlocked(active, subscriber)
                return AttemptAdmission(
                    None,
                    [message for message in (active.accepted, active.latest_progress) if message],
                )
            self._terminals[attempt_id] = TerminalAttempt(
                message=terminal,
                result=None,
                terminal_at=time.monotonic(),
            )
            self._terminals.move_to_end(attempt_id)
            self._prune_unlocked()
            return AttemptAdmission(None, [terminal])

    async def cancel_event_for(self, receipt: AttemptReceipt) -> asyncio.Event:
        async with self._gate:
            active = self._active
            if active is not None and active.receipt == receipt:
                return active.cancel_event
        canceled = asyncio.Event()
        canceled.set()
        return canceled

    def _commit_terminal_unlocked(
        self, active: ActiveAttempt, message: dict, result: dict | None = None
    ) -> TerminalTransition:
        self._terminals[active.receipt.attempt_id] = TerminalAttempt(
            message=message,
            result=result,
            terminal_at=time.monotonic(),
        )
        self._terminals.move_to_end(active.receipt.attempt_id)
        subscribers = list(active.subscribers.values())
        self._active = None
        if active.owner_receipts is not None:
            active.owner_receipts.discard(active.receipt)
        self._prune_unlocked()
        return TerminalTransition(message=message, subscribers=subscribers)

    async def cancel_owner_and_join(self, receipt: AttemptReceipt) -> TerminalTransition | None:
        async with self._gate:
            active = self._active
            if active is None or active.receipt != receipt:
                return None
            active.cancel_event.set()
            task = active.task
            transition = (
                self._commit_terminal_unlocked(active, active.canceled_terminal)
                if active.canceled_terminal is not None
                else None
            )
        if task is not asyncio.current_task():
            await asyncio.shield(task)
        return transition

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
            return self._commit_terminal_unlocked(active, message, result).subscribers

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

    async def shutdown(self) -> TerminalTransition | None:
        async with self._gate:
            active = self._active
            if active is None:
                return None
            active.cancel_event.set()
            task = active.task
            transition = (
                self._commit_terminal_unlocked(active, active.canceled_terminal)
                if active.canceled_terminal is not None
                else None
            )
        if task is not asyncio.current_task():
            await asyncio.shield(task)
        return transition
