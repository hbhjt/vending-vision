"""Process-wide ownership and replay state for Fast try-on attempts.

Replacement uses a pending admission reservation while the old owner joins.
No accepted/progress state becomes visible until readiness and active ownership
are committed together under the short state gate.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable
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
    expires_at: float


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


@dataclass
class PendingAdmission:
    attempt_id: str
    token: str
    task: asyncio.Task
    prior_task: asyncio.Task | None
    owner_receipts: set[AttemptReceipt] | None
    owner_subscriber_key: int
    subscribers: dict[int, AttemptSubscriber]
    canceled_terminal: dict | None
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class CleanupReservation:
    prior_task: asyncio.Task
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class AdmissionPreparation:
    attempt_id: str
    resolved: bool = False
    token: str | None = None
    join_task: asyncio.Task | None = None
    wait_event: asyncio.Event | None = None
    replay: list[dict] = field(default_factory=list)
    transitions: list[TerminalTransition] = field(default_factory=list)

    @property
    def is_pending_owner(self) -> bool:
        return self.token is not None


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
        self._gate = asyncio.Lock()
        self._active: ActiveAttempt | None = None
        self._pending: PendingAdmission | None = None
        self._cleanup: CleanupReservation | None = None
        self._terminals: OrderedDict[str, TerminalAttempt] = OrderedDict()
        self._generation = 0
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._terminal_max_count = terminal_max_count
        self._subscriber_max_count = subscriber_max_count

    def _attach_subscriber_unlocked(
        self, active: ActiveAttempt, subscriber: AttemptSubscriber
    ) -> None:
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

    def _attach_pending_subscriber_unlocked(
        self, pending: PendingAdmission, subscriber: AttemptSubscriber
    ) -> None:
        if subscriber.key in pending.subscribers:
            pending.subscribers[subscriber.key] = subscriber
            return
        while len(pending.subscribers) >= self._subscriber_max_count:
            stale_key = next(
                (
                    key
                    for key in pending.subscribers
                    if key != pending.owner_subscriber_key
                ),
                None,
            )
            if stale_key is None:
                break
            pending.subscribers.pop(stale_key, None)
        if len(pending.subscribers) < self._subscriber_max_count:
            pending.subscribers[subscriber.key] = subscriber

    def _prune_unlocked(self) -> None:
        now = time.monotonic()
        for attempt_id, terminal in list(self._terminals.items()):
            if terminal.expires_at <= now:
                self._terminals.pop(attempt_id, None)
        while len(self._terminals) > self._terminal_max_count:
            self._terminals.popitem(last=False)

    def _new_terminal(
        self, message: dict, result: dict | None = None
    ) -> TerminalAttempt:
        terminal_at = time.monotonic()
        return TerminalAttempt(
            message=message,
            result=result,
            terminal_at=terminal_at,
            expires_at=terminal_at + self._terminal_ttl_seconds,
        )

    async def prepare_admission(
        self,
        *,
        attempt_id: str,
        websocket: Any,
        send_lock: asyncio.Lock,
        task: asyncio.Task,
        canceled_terminal: dict | None = None,
        owner_receipts: set[AttemptReceipt] | None = None,
    ) -> AdmissionPreparation:
        """Reserve one admission without awaiting an old task under the gate."""
        subscriber = AttemptSubscriber(id(websocket), websocket, send_lock)
        while True:
            wait_for_other: asyncio.Event | None = None
            wait_for_cleanup: asyncio.Event | None = None
            transitions: list[TerminalTransition] = []
            async with self._gate:
                self._prune_unlocked()
                terminal = self._terminals.get(attempt_id)
                if terminal is not None:
                    self._terminals.move_to_end(attempt_id)
                    return AdmissionPreparation(
                        attempt_id=attempt_id,
                        resolved=True,
                        replay=[terminal.message],
                    )

                active = self._active
                if active is not None and active.receipt.attempt_id == attempt_id:
                    self._attach_subscriber_unlocked(active, subscriber)
                    replay = [
                        message
                        for message in (active.accepted, active.latest_progress)
                        if message
                    ]
                    return AdmissionPreparation(
                        attempt_id=attempt_id, resolved=True, replay=replay
                    )

                cleanup = self._cleanup
                pending = self._pending
                if cleanup is not None:
                    wait_for_cleanup = cleanup.done
                elif pending is not None:
                    if pending.attempt_id == attempt_id:
                        self._attach_pending_subscriber_unlocked(pending, subscriber)
                        return AdmissionPreparation(
                            attempt_id=attempt_id, wait_event=pending.done
                        )
                    wait_for_other = pending.done
                else:
                    join_task = None
                    if active is not None:
                        active.cancel_event.set()
                        join_task = active.task
                        if active.canceled_terminal is not None:
                            transitions.append(
                                self._commit_terminal_unlocked(
                                    active, active.canceled_terminal
                                )
                            )
                    token = uuid4().hex
                    self._pending = PendingAdmission(
                        attempt_id=attempt_id,
                        token=token,
                        task=task,
                        prior_task=join_task,
                        owner_receipts=owner_receipts,
                        owner_subscriber_key=subscriber.key,
                        subscribers={subscriber.key: subscriber},
                        canceled_terminal=canceled_terminal,
                    )
                    return AdmissionPreparation(
                        attempt_id=attempt_id,
                        token=token,
                        join_task=join_task,
                        transitions=transitions,
                    )
            if wait_for_cleanup is not None:
                await asyncio.shield(wait_for_cleanup.wait())
                continue
            assert wait_for_other is not None
            await asyncio.shield(wait_for_other.wait())

    async def commit_prepared_admission(
        self,
        preparation: AdmissionPreparation,
        *,
        accepted: dict | None,
        progress: dict | None,
        unavailable_terminal: dict,
        readiness: Callable[[], bool],
    ) -> AttemptAdmission:
        """Join outside the gate, then atomically publish ready active state."""
        if preparation.resolved:
            return AttemptAdmission(None, preparation.replay)
        if preparation.wait_event is not None:
            await asyncio.shield(preparation.wait_event.wait())
            async with self._gate:
                self._prune_unlocked()
                terminal = self._terminals.get(preparation.attempt_id)
                if terminal is not None:
                    self._terminals.move_to_end(preparation.attempt_id)
                    return AttemptAdmission(None, [terminal.message])
                active = self._active
                if (
                    active is not None
                    and active.receipt.attempt_id == preparation.attempt_id
                ):
                    return AttemptAdmission(
                        None,
                        [
                            message
                            for message in (active.accepted, active.latest_progress)
                            if message
                        ],
                    )
                raise RuntimeError("pending Fast admission resolved without state")

        if (
            preparation.join_task is not None
            and preparation.join_task is not asyncio.current_task()
        ):
            try:
                await asyncio.shield(preparation.join_task)
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is None or current_task.cancelling() == 0:
                    self._consume_finished_task(preparation.join_task)
                else:
                    cleanup = None
                    async with self._gate:
                        pending = self._pending
                        if (
                            pending is not None
                            and pending.token == preparation.token
                        ):
                            cleanup = self._reserve_cleanup_unlocked(
                                preparation.join_task
                            )
                            if pending.canceled_terminal is not None:
                                self._commit_pending_terminal_unlocked(
                                    pending, pending.canceled_terminal
                                )
                            else:
                                self._pending = None
                                pending.done.set()
                        elif self._cleanup is not None:
                            cleanup = self._cleanup
                    if cleanup is not None:
                        await self._finish_cleanup_uncancelled(cleanup)
                    raise
            except Exception:
                self._consume_finished_task(preparation.join_task)

        async with self._gate:
            self._prune_unlocked()
            terminal = self._terminals.get(preparation.attempt_id)
            pending = self._pending
            if terminal is not None:
                self._terminals.move_to_end(preparation.attempt_id)
                if pending is not None and pending.token == preparation.token:
                    self._pending = None
                    pending.done.set()
                return AttemptAdmission(None, [terminal.message])
            if pending is None or pending.token != preparation.token:
                active = self._active
                if (
                    active is not None
                    and active.receipt.attempt_id == preparation.attempt_id
                ):
                    return AttemptAdmission(
                        None,
                        [
                            message
                            for message in (active.accepted, active.latest_progress)
                            if message
                        ],
                    )
                raise RuntimeError("Fast admission reservation was lost")

            if not readiness():
                self._terminals[preparation.attempt_id] = self._new_terminal(
                    unavailable_terminal
                )
                self._terminals.move_to_end(preparation.attempt_id)
                self._pending = None
                pending.done.set()
                self._prune_unlocked()
                return AttemptAdmission(None, [unavailable_terminal])

            self._generation += 1
            receipt = AttemptReceipt(
                preparation.attempt_id, uuid4().hex, self._generation
            )
            if pending.owner_receipts is not None:
                pending.owner_receipts.add(receipt)
            self._active = ActiveAttempt(
                receipt=receipt,
                task=pending.task,
                cancel_event=asyncio.Event(),
                owner_receipts=pending.owner_receipts,
                owner_subscriber_key=pending.owner_subscriber_key,
                subscribers=pending.subscribers,
                accepted=accepted,
                latest_progress=progress,
                canceled_terminal=pending.canceled_terminal,
            )
            self._pending = None
            pending.done.set()
            return AttemptAdmission(
                receipt,
                [message for message in (accepted, progress) if message],
            )

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
        """Compatibility wrapper for callers without a readiness boundary."""
        preparation = await self.prepare_admission(
            attempt_id=attempt_id,
            websocket=websocket,
            send_lock=send_lock,
            task=task,
            canceled_terminal=canceled_terminal,
            owner_receipts=owner_receipts,
        )
        admission = await self.commit_prepared_admission(
            preparation,
            accepted=accepted,
            progress=progress,
            unavailable_terminal={},
            readiness=lambda: True,
        )
        admission.transitions.extend(preparation.transitions)
        return admission

    async def is_current(self, receipt: AttemptReceipt) -> bool:
        async with self._gate:
            active = self._active
            return (
                active is not None
                and active.receipt == receipt
                and not active.cancel_event.is_set()
            )

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
        while True:
            wait_cleanup = None
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
                        [
                            message
                            for message in (active.accepted, active.latest_progress)
                            if message
                        ],
                    )
                pending = self._pending
                if pending is not None and pending.attempt_id == attempt_id:
                    self._attach_pending_subscriber_unlocked(pending, subscriber)
                    self._commit_pending_terminal_unlocked(pending, terminal)
                    return AttemptAdmission(None, [terminal])
                if self._cleanup is not None:
                    wait_cleanup = self._cleanup.done
                else:
                    self._terminals[attempt_id] = self._new_terminal(terminal)
                    self._terminals.move_to_end(attempt_id)
                    self._prune_unlocked()
                    return AttemptAdmission(None, [terminal])
            assert wait_cleanup is not None
            await asyncio.shield(wait_cleanup.wait())

    async def join_pending_or_reject(
        self,
        *,
        attempt_id: str,
        websocket: Any,
        send_lock: asyncio.Lock,
        terminal: dict,
    ) -> AttemptAdmission:
        """Attach same-ID overflow to pending state; reject only new work."""
        subscriber = AttemptSubscriber(id(websocket), websocket, send_lock)
        while True:
            wait_event = None
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
                        [
                            message
                            for message in (active.accepted, active.latest_progress)
                            if message
                        ],
                    )
                pending = self._pending
                if pending is not None and pending.attempt_id == attempt_id:
                    self._attach_pending_subscriber_unlocked(pending, subscriber)
                    wait_event = pending.done
                elif self._cleanup is not None:
                    wait_event = self._cleanup.done
                else:
                    self._terminals[attempt_id] = self._new_terminal(terminal)
                    self._terminals.move_to_end(attempt_id)
                    self._prune_unlocked()
                    return AttemptAdmission(None, [terminal])
            assert wait_event is not None
            await asyncio.shield(wait_event.wait())

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
        self._terminals[active.receipt.attempt_id] = self._new_terminal(message, result)
        self._terminals.move_to_end(active.receipt.attempt_id)
        subscribers = list(active.subscribers.values())
        self._active = None
        if active.owner_receipts is not None:
            active.owner_receipts.discard(active.receipt)
        self._prune_unlocked()
        return TerminalTransition(message=message, subscribers=subscribers)

    def _commit_pending_terminal_unlocked(
        self, pending: PendingAdmission, message: dict
    ) -> TerminalTransition:
        self._terminals[pending.attempt_id] = self._new_terminal(message)
        self._terminals.move_to_end(pending.attempt_id)
        subscribers = list(pending.subscribers.values())
        if self._pending is pending:
            self._pending = None
        pending.done.set()
        self._prune_unlocked()
        return TerminalTransition(message=message, subscribers=subscribers)

    def _reserve_cleanup_unlocked(
        self, prior_task: asyncio.Task | None
    ) -> CleanupReservation | None:
        if prior_task is None:
            return self._cleanup
        if self._cleanup is None:
            self._cleanup = CleanupReservation(prior_task=prior_task)
        return self._cleanup

    @staticmethod
    def _consume_finished_task(task: asyncio.Task) -> None:
        """Retrieve a finished prior task's outcome without reclassifying it."""
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    async def _finish_cleanup_uncancelled(
        self, cleanup: CleanupReservation
    ) -> None:
        async def finalize() -> None:
            try:
                await asyncio.shield(cleanup.prior_task)
            except asyncio.CancelledError:
                self._consume_finished_task(cleanup.prior_task)
            except Exception:
                self._consume_finished_task(cleanup.prior_task)
                # The prior owner's outcome is already represented by its
                # canonical terminal.  Admission only needs resource closure.
            async with self._gate:
                if self._cleanup is cleanup:
                    self._cleanup = None
                cleanup.done.set()

        finalizer = asyncio.create_task(finalize())
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                continue
        finalizer.result()

    async def cancel_owner_and_join(
        self, receipt: AttemptReceipt
    ) -> TerminalTransition | None:
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
            if self._pending is not None:
                self._pending.subscribers.pop(id(websocket), None)

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
            pending = self._pending
            task = None
            cleanup = self._cleanup
            transition = None
            if active is not None:
                active.cancel_event.set()
                task = active.task
                transition = (
                    self._commit_terminal_unlocked(active, active.canceled_terminal)
                    if active.canceled_terminal is not None
                    else None
                )
            elif pending is not None:
                task = pending.task
                cleanup = self._reserve_cleanup_unlocked(pending.prior_task)
                if pending.canceled_terminal is not None:
                    transition = self._commit_pending_terminal_unlocked(
                        pending, pending.canceled_terminal
                    )
                else:
                    self._pending = None
                    pending.done.set()
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        if cleanup is not None:
            await self._finish_cleanup_uncancelled(cleanup)
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        return transition
