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
