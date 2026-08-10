import asyncio
from uuid import uuid4

from vision.fast_attempt_registry import FastAttemptRegistry


def _message(message_type: str, attempt_id: str, message_id: str) -> dict:
    return {
        "type": message_type,
        "messageId": message_id,
        "payload": {"attemptId": attempt_id, "reason": "attempt_replaced"},
    }


def test_cancel_wins_over_completed_with_one_canonical_terminal():
    """Cancellation CAS clears active work and makes the terminal replayable."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        accepted = _message("vision.try_on.attempt.accepted", attempt_id, "accepted-1")
        progress = _message("vision.try_on.attempt.progress", attempt_id, "progress-1")
        canceled = _message("vision.try_on.attempt.failed", attempt_id, "canceled-1")
        completed = _message("vision.try_on.attempt.completed", attempt_id, "completed-1")
        receipts = set()
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            progress=progress,
            canceled_terminal=canceled,
            owner_receipts=receipts,
        )
        assert admission.is_owner
        assert receipts == {admission.receipt}
        await registry.cancel_owner_and_join(admission.receipt)
        assert receipts == set()

        assert await registry.commit_terminal(admission.receipt, completed) == []
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            progress=progress,
        )

        assert replay.replay == [canceled]

    asyncio.run(scenario())


def test_completed_wins_over_later_cancel_with_one_canonical_terminal():
    """A success committed first cannot be overwritten by a late cancellation."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        accepted = _message("vision.try_on.attempt.accepted", attempt_id, "accepted-2")
        progress = _message("vision.try_on.attempt.progress", attempt_id, "progress-2")
        completed = _message("vision.try_on.attempt.completed", attempt_id, "completed-2")

        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            progress=progress,
            canceled_terminal=_message("vision.try_on.attempt.failed", attempt_id, "canceled-2"),
        )
        assert admission.is_owner
        assert len(await registry.commit_terminal(admission.receipt, completed)) == 1
        await registry.cancel_owner_and_join(admission.receipt)
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            progress=progress,
        )

        assert replay.replay == [completed]

    asyncio.run(scenario())


def test_waiting_replacement_rechecks_after_same_id_backpressure_terminal():
    """A waiting replacement must not overwrite a same-ID terminal won meanwhile."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        first_id = str(uuid4())
        replacement_id = str(uuid4())
        join_released = asyncio.Event()
        first_task_done = asyncio.Event()

        async def first_task():
            await join_released.wait()
            first_task_done.set()

        first_task_handle = asyncio.create_task(first_task())
        first_canceled = _message("vision.try_on.attempt.failed", first_id, "first-canceled")
        first_admission = await registry.admit(
            attempt_id=first_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=first_task_handle,
            accepted=_message("vision.try_on.attempt.accepted", first_id, "first-accepted"),
            progress=_message("vision.try_on.attempt.progress", first_id, "first-progress"),
            canceled_terminal=first_canceled,
        )
        assert first_admission.is_owner

        replacement_task = asyncio.create_task(
            registry.admit(
                attempt_id=replacement_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=_message("vision.try_on.attempt.accepted", replacement_id, "replacement-accepted"),
                progress=_message("vision.try_on.attempt.progress", replacement_id, "replacement-progress"),
            )
        )

        await asyncio.sleep(0)
        backpressure_terminal = _message(
            "vision.try_on.attempt.failed",
            replacement_id,
            "backpressure-winner",
        )
        backpressure = await registry.reject_or_replay(
            attempt_id=replacement_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            terminal=backpressure_terminal,
        )
        assert backpressure.replay == [backpressure_terminal]

        join_released.set()
        replacement_admission = await asyncio.wait_for(replacement_task, timeout=1.0)
        assert not replacement_admission.is_owner
        assert replacement_admission.replay == [backpressure_terminal]
        assert first_task_done.is_set()

    asyncio.run(scenario())


def test_terminal_ttl_prunes_all_expired_records_not_only_lru_head():
    """Replaying one terminal must not hide older expired records behind it."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=0.05, terminal_max_count=10)
        oldest = str(uuid4())
        middle = str(uuid4())
        newest = str(uuid4())
        for attempt_id in (oldest, middle, newest):
            await registry.reject_or_replay(
                attempt_id=attempt_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                terminal=_message("vision.try_on.attempt.failed", attempt_id, attempt_id),
            )
            await asyncio.sleep(0.01)

        # Move the oldest to the OrderedDict tail without extending its TTL.
        replay = await registry.reject_or_replay(
            attempt_id=oldest,
            websocket=object(),
            send_lock=asyncio.Lock(),
            terminal=_message("vision.try_on.attempt.failed", oldest, "ignored"),
        )
        assert replay.replay[0]["messageId"] == oldest
        await asyncio.sleep(0.06)

        retry = await registry.admit(
            attempt_id=oldest,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=_message("vision.try_on.attempt.accepted", oldest, "retry-accepted"),
            progress=_message("vision.try_on.attempt.progress", oldest, "retry-progress"),
        )
        assert retry.is_owner
        await registry.cancel_owner_and_join(retry.receipt)

    asyncio.run(scenario())
