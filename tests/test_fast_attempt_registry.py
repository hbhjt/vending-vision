import asyncio
import gc
from uuid import uuid4

import pytest

from vision.fast_attempt_registry import FastAttemptRegistry


def _message(message_type: str, attempt_id: str, message_id: str) -> dict:
    return {
        "type": message_type,
        "messageId": message_id,
        "payload": {"attemptId": attempt_id, "reason": "replaced"},
    }


def test_cancel_wins_over_completed_with_one_canonical_terminal():
    """Cancellation CAS clears active work and makes the terminal replayable."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        accepted = _message("vision.try_on.attempt.accepted", attempt_id, "accepted-1")
        generating = _message("vision.try_on.attempt.generating", attempt_id, "generating-1")
        canceled = _message("vision.try_on.attempt.failed", attempt_id, "canceled-1")
        completed = _message("vision.try_on.attempt.completed", attempt_id, "completed-1")
        receipts = set()
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            generating=generating,
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
            generating=generating,
        )

        assert replay.replay == [canceled]

    asyncio.run(scenario())


def test_disconnect_keeps_cleanup_barrier_before_another_socket_can_admit():
    """A disconnect terminal fences new owners until the disconnected worker joins."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        old_id, new_id = str(uuid4()), str(uuid4())
        release_old = asyncio.Event()

        async def old_owner():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Simulate the bounded-but-real resource close in the owner
                # finally block: it must finish before a new lease may start.
                await release_old.wait()

        old_task = asyncio.create_task(old_owner())
        await asyncio.sleep(0)
        old = await registry.admit(
            attempt_id=old_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=old_task,
            accepted=_message("vision.try_on.attempt.accepted", old_id, "old-accepted"),
            generating=_message("vision.try_on.attempt.acquiring", old_id, "old-acquiring"),
        )
        assert old.is_owner
        disconnect = _message("vision.try_on.attempt.canceled", old_id, "disconnect")
        disconnect["payload"]["reason"] = "disconnect"
        disconnecting = asyncio.create_task(
            registry.cancel_owner_and_join(old.receipt, disconnect)
        )
        await asyncio.sleep(0)

        replacement = asyncio.create_task(
            registry.prepare_admission(
                attempt_id=new_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
            )
        )
        await asyncio.sleep(0)
        assert not replacement.done(), "new ID became pending before old cleanup joined"

        replay = await registry.prepare_admission(
            attempt_id=old_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
        )
        assert replay.resolved
        assert replay.replay == [disconnect]

        release_old.set()
        transition = await asyncio.wait_for(disconnecting, timeout=1)
        assert transition is not None and transition.message == disconnect
        assert (await asyncio.wait_for(replacement, timeout=1)).is_pending_owner

    asyncio.run(scenario())


def test_replacement_cancels_blocked_owner_and_joins_cleanup_before_admission():
    """Replacement must interrupt a blocked owner, then admit only after its cleanup."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        old_id, new_id = str(uuid4()), str(uuid4())
        cleanup_done = asyncio.Event()

        async def old_owner():
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_done.set()

        old_task = asyncio.create_task(old_owner())
        await asyncio.sleep(0)
        old = await registry.admit(
            attempt_id=old_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=old_task,
            accepted=_message("vision.try_on.attempt.accepted", old_id, "old-accepted"),
            generating=_message("vision.try_on.attempt.acquiring", old_id, "old-acquiring"),
            canceled_terminal=_message("vision.try_on.attempt.canceled", old_id, "old-replaced"),
        )
        assert old.is_owner

        prepared = await registry.prepare_admission(
            attempt_id=new_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
        )
        assert prepared.join_task is old_task
        admitted = await asyncio.wait_for(
            registry.commit_prepared_admission(
                prepared,
                accepted=_message("vision.try_on.attempt.accepted", new_id, "new-accepted"),
                generating=_message("vision.try_on.attempt.acquiring", new_id, "new-acquiring"),
                unavailable_terminal=_message("vision.try_on.attempt.failed", new_id, "new-failed"),
                readiness=lambda: True,
            ),
            timeout=0.25,
        )

        assert cleanup_done.is_set()
        assert admitted.is_owner
        assert admitted.receipt.attempt_id == new_id

    asyncio.run(scenario())


def test_completed_wins_over_later_cancel_with_one_canonical_terminal():
    """A success committed first cannot be overwritten by a late cancellation."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        accepted = _message("vision.try_on.attempt.accepted", attempt_id, "accepted-2")
        generating = _message("vision.try_on.attempt.generating", attempt_id, "generating-2")
        completed = _message("vision.try_on.attempt.completed", attempt_id, "completed-2")

        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            generating=generating,
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
            generating=generating,
        )

        assert replay.replay == [completed]

    asyncio.run(scenario())


def test_explicit_cancel_publishes_before_blocked_owner_cleanup_finishes():
    """A WebSocket cancel must not wait for a blocked renderer to receive ping."""
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        release_owner = asyncio.Event()

        async def owner():
            await release_owner.wait()

        owner_task = asyncio.create_task(owner())
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(), send_lock=asyncio.Lock(), task=owner_task,
            accepted=_message("vision.try_on.attempt.accepted", attempt_id, "accepted"),
            generating=_message("vision.try_on.attempt.acquiring", attempt_id, "acquiring"),
        )
        canceled = _message("vision.try_on.attempt.canceled", attempt_id, "canceled")
        transition = await asyncio.wait_for(
            registry.cancel_current(attempt_id=attempt_id, terminal=canceled), timeout=0.05
        )
        assert transition is not None and transition.message == canceled
        replacement = asyncio.create_task(registry.prepare_admission(
            attempt_id=str(uuid4()), websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
        ))
        release_owner.set()
        await asyncio.gather(owner_task, return_exceptions=True)
        assert (await asyncio.wait_for(replacement, timeout=1)).is_pending_owner

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
            generating=_message("vision.try_on.attempt.generating", first_id, "first-generating"),
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
                generating=_message("vision.try_on.attempt.generating", replacement_id, "replacement-generating"),
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


@pytest.mark.parametrize("prior_outcome", ["return", "cancel", "error"])
@pytest.mark.parametrize("ready", [True, False])
def test_prior_join_outcome_is_consumed_before_pending_admission_recheck(
    prior_outcome, ready
):
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        unhandled_contexts = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: unhandled_contexts.append(context)
        )
        first_id, pending_id, next_id = str(uuid4()), str(uuid4()), str(uuid4())
        prior_release = asyncio.Event()
        pending_prepared = asyncio.Event()

        async def prior_task():
            await prior_release.wait()
            if prior_outcome == "cancel":
                raise asyncio.CancelledError
            if prior_outcome == "error":
                raise RuntimeError("prior resource cleanup failed")

        prior_handle = asyncio.create_task(prior_task())
        first = await registry.admit(
            attempt_id=first_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=prior_handle,
            accepted=_message(
                "vision.try_on.attempt.accepted", first_id, "first-accepted"
            ),
            generating=_message(
                "vision.try_on.attempt.generating", first_id, "first-generating"
            ),
            canceled_terminal=_message(
                "vision.try_on.attempt.failed", first_id, "first-canceled"
            ),
        )
        assert first.is_owner

        accepted = _message(
            "vision.try_on.attempt.accepted", pending_id, "pending-accepted"
        )
        generating = _message(
            "vision.try_on.attempt.generating", pending_id, "pending-generating"
        )
        unavailable = _message(
            "vision.try_on.attempt.failed", pending_id, "pending-unavailable"
        )

        async def pending_owner():
            preparation = await registry.prepare_admission(
                attempt_id=pending_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                canceled_terminal=_message(
                    "vision.try_on.attempt.failed", pending_id, "pending-canceled"
                ),
            )
            pending_prepared.set()
            return await registry.commit_prepared_admission(
                preparation,
                accepted=accepted,
                generating=generating,
                unavailable_terminal=unavailable,
                readiness=lambda: ready,
            )

        pending_task = asyncio.create_task(pending_owner())
        await pending_prepared.wait()
        assert pending_task.cancelling() == 0
        prior_release.set()
        pending = await asyncio.wait_for(pending_task, timeout=1.0)

        replay = await registry.admit(
            attempt_id=pending_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        if ready:
            assert pending.is_owner
            assert replay.replay == [accepted, generating]
            await registry.cancel_owner_and_join(pending.receipt)
        else:
            assert not pending.is_owner
            assert pending.replay == [unavailable]
            assert replay.replay == [unavailable]

        next_admission = await asyncio.wait_for(
            registry.admit(
                attempt_id=next_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=_message(
                    "vision.try_on.attempt.accepted", next_id, "next-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", next_id, "next-generating"
                ),
            ),
            timeout=1.0,
        )
        assert next_admission.is_owner
        await registry.cancel_owner_and_join(next_admission.receipt)
        del prior_handle
        gc.collect()
        await asyncio.sleep(0)
        assert unhandled_contexts == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("initially_ready", "ready_at_commit"),
    [(True, False), (False, True)],
)
def test_prepared_admission_atomically_binds_active_state_to_resolved_accepted(
    initially_ready, ready_at_commit
):
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        attempt_id = str(uuid4())
        accepted = _message(
            "vision.try_on.attempt.accepted", attempt_id, "accepted"
        )
        unavailable = _message(
            "vision.try_on.attempt.failed", attempt_id, "unavailable"
        )
        current = {"ready": initially_ready}
        preparation = await registry.prepare_admission(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
        )

        current["ready"] = ready_at_commit
        admission = await registry.commit_prepared_admission(
            preparation,
            accepted=None,
            generating=None,
            unavailable_terminal=unavailable,
            accepted_resolver=lambda: accepted if current["ready"] else None,
        )
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )

        if ready_at_commit:
            assert admission.is_owner
            assert admission.replay == [accepted]
            assert replay.replay == [accepted]
            await registry.cancel_owner_and_join(admission.receipt)
        else:
            assert not admission.is_owner
            assert admission.replay == [unavailable]
            assert replay.replay == [unavailable]

    asyncio.run(scenario())


def test_cancelled_pending_owner_keeps_cleanup_barrier_until_prior_task_ends():
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        first_id, pending_id, next_id = str(uuid4()), str(uuid4()), str(uuid4())
        prior_release = asyncio.Event()
        prior_done = asyncio.Event()
        pending_prepared = asyncio.Event()

        async def prior_task():
            await prior_release.wait()
            prior_done.set()

        prior_handle = asyncio.create_task(prior_task())
        first_canceled = _message(
            "vision.try_on.attempt.failed", first_id, "first-canceled"
        )
        first = await registry.admit(
            attempt_id=first_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=prior_handle,
            accepted=_message(
                "vision.try_on.attempt.accepted", first_id, "first-accepted"
            ),
            generating=_message(
                "vision.try_on.attempt.generating", first_id, "first-generating"
            ),
            canceled_terminal=first_canceled,
        )
        assert first.is_owner

        pending_canceled = _message(
            "vision.try_on.attempt.failed", pending_id, "pending-canceled"
        )

        async def pending_owner():
            preparation = await registry.prepare_admission(
                attempt_id=pending_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                canceled_terminal=pending_canceled,
            )
            pending_prepared.set()
            return await registry.commit_prepared_admission(
                preparation,
                accepted=_message(
                    "vision.try_on.attempt.accepted", pending_id, "pending-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", pending_id, "pending-generating"
                ),
                unavailable_terminal=_message(
                    "vision.try_on.attempt.failed", pending_id, "pending-unavailable"
                ),
                readiness=lambda: True,
            )

        pending_task = asyncio.create_task(pending_owner())
        await pending_prepared.wait()
        pending_task.cancel()
        await asyncio.sleep(0)

        same_id = await registry.admit(
            attempt_id=pending_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=_message(
                "vision.try_on.attempt.accepted", pending_id, "ignored-accepted"
            ),
            generating=_message(
                "vision.try_on.attempt.generating", pending_id, "ignored-generating"
            ),
        )
        assert same_id.replay == [pending_canceled]

        next_admission_task = asyncio.create_task(
            registry.admit(
                attempt_id=next_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=_message(
                    "vision.try_on.attempt.accepted", next_id, "next-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", next_id, "next-generating"
                ),
            )
        )
        await asyncio.sleep(0.05)
        assert not prior_done.is_set()
        assert not pending_task.done()
        assert not next_admission_task.done()

        prior_release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending_task
        next_admission = await asyncio.wait_for(next_admission_task, timeout=1.0)
        assert prior_done.is_set()
        assert next_admission.is_owner

        replay = await registry.admit(
            attempt_id=pending_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert replay.replay == [pending_canceled]
        await registry.cancel_owner_and_join(next_admission.receipt)

    asyncio.run(scenario())


@pytest.mark.parametrize("prior_outcome", ["cancel", "error"])
def test_repeated_pending_cancel_waits_for_failed_prior_and_keeps_cancelled_error(
    prior_outcome,
):
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        first_id, pending_id, next_id = str(uuid4()), str(uuid4()), str(uuid4())
        prior_release = asyncio.Event()
        prior_exited = asyncio.Event()
        pending_prepared = asyncio.Event()

        async def failed_prior():
            await prior_release.wait()
            prior_exited.set()
            if prior_outcome == "cancel":
                raise asyncio.CancelledError
            raise RuntimeError("prior cleanup failure")

        prior_handle = asyncio.create_task(failed_prior())
        first = await registry.admit(
            attempt_id=first_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=prior_handle,
            accepted=_message(
                "vision.try_on.attempt.accepted", first_id, "first-accepted"
            ),
            generating=_message(
                "vision.try_on.attempt.generating", first_id, "first-generating"
            ),
            canceled_terminal=_message(
                "vision.try_on.attempt.failed", first_id, "first-canceled"
            ),
        )
        assert first.is_owner

        pending_canceled = _message(
            "vision.try_on.attempt.failed", pending_id, "pending-canceled"
        )

        async def pending_owner():
            preparation = await registry.prepare_admission(
                attempt_id=pending_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                canceled_terminal=pending_canceled,
            )
            pending_prepared.set()
            return await registry.commit_prepared_admission(
                preparation,
                accepted=_message(
                    "vision.try_on.attempt.accepted", pending_id, "pending-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", pending_id, "pending-generating"
                ),
                unavailable_terminal=_message(
                    "vision.try_on.attempt.failed", pending_id, "pending-unavailable"
                ),
                readiness=lambda: True,
            )

        pending_task = asyncio.create_task(pending_owner())
        await pending_prepared.wait()
        pending_task.cancel()
        await asyncio.sleep(0)
        pending_task.cancel()

        next_task = asyncio.create_task(
            registry.admit(
                attempt_id=next_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=_message(
                    "vision.try_on.attempt.accepted", next_id, "next-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", next_id, "next-generating"
                ),
            )
        )
        await asyncio.sleep(0.05)
        assert not prior_exited.is_set()
        assert not pending_task.done()
        assert not next_task.done()

        prior_release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending_task
        next_admission = await asyncio.wait_for(next_task, timeout=1.0)
        assert prior_exited.is_set()
        assert next_admission.is_owner

        replay = await registry.admit(
            attempt_id=pending_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert replay.replay == [pending_canceled]
        await registry.cancel_owner_and_join(next_admission.receipt)

    asyncio.run(scenario())


def test_shutdown_during_pending_cancel_waits_for_prior_cleanup_barrier():
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        first_id, pending_id, next_id = str(uuid4()), str(uuid4()), str(uuid4())
        prior_release = asyncio.Event()
        prior_done = asyncio.Event()
        pending_prepared = asyncio.Event()

        async def prior_task():
            await prior_release.wait()
            prior_done.set()

        prior_handle = asyncio.create_task(prior_task())
        first = await registry.admit(
            attempt_id=first_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=prior_handle,
            accepted=_message(
                "vision.try_on.attempt.accepted", first_id, "first-accepted"
            ),
            generating=_message(
                "vision.try_on.attempt.generating", first_id, "first-generating"
            ),
            canceled_terminal=_message(
                "vision.try_on.attempt.failed", first_id, "first-canceled"
            ),
        )
        assert first.is_owner
        pending_canceled = _message(
            "vision.try_on.attempt.failed", pending_id, "pending-canceled"
        )

        async def pending_owner():
            preparation = await registry.prepare_admission(
                attempt_id=pending_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                canceled_terminal=pending_canceled,
            )
            pending_prepared.set()
            return await registry.commit_prepared_admission(
                preparation,
                accepted=_message(
                    "vision.try_on.attempt.accepted", pending_id, "pending-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", pending_id, "pending-generating"
                ),
                unavailable_terminal=_message(
                    "vision.try_on.attempt.failed", pending_id, "pending-unavailable"
                ),
                readiness=lambda: True,
            )

        pending_task = asyncio.create_task(pending_owner())
        await pending_prepared.wait()
        shutdown_task = asyncio.create_task(registry.shutdown())
        await asyncio.sleep(0)
        next_task = asyncio.create_task(
            registry.admit(
                attempt_id=next_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=_message(
                    "vision.try_on.attempt.accepted", next_id, "next-accepted"
                ),
                generating=_message(
                    "vision.try_on.attempt.generating", next_id, "next-generating"
                ),
            )
        )

        same_id = await registry.admit(
            attempt_id=pending_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert same_id.replay == [pending_canceled]
        await asyncio.sleep(0.05)
        assert not prior_done.is_set()
        assert not pending_task.done()
        assert not shutdown_task.done()
        assert not next_task.done()

        prior_release.set()
        transition = await asyncio.wait_for(shutdown_task, timeout=1.0)
        assert transition is not None
        assert transition.message == pending_canceled
        with pytest.raises(asyncio.CancelledError):
            await pending_task
        next_admission = await asyncio.wait_for(next_task, timeout=1.0)
        assert prior_done.is_set()
        assert next_admission.is_owner
        await registry.cancel_owner_and_join(next_admission.receipt)

    asyncio.run(scenario())


def test_repeated_outer_cancel_keeps_cleanup_barrier_until_old_owner_stops():
    async def scenario():
        registry = FastAttemptRegistry(terminal_ttl_seconds=60)
        old_id = str(uuid4())
        new_id = str(uuid4())
        old_stop_released = asyncio.Event()
        old_cleanup_reached = asyncio.Event()

        async def old_owner():
            try:
                await asyncio.Event().wait()
            finally:
                old_cleanup_reached.set()
                await old_stop_released.wait()

        old_task = asyncio.create_task(old_owner())
        old = await registry.admit(
            attempt_id=old_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=old_task,
            accepted=_message("vision.try_on.attempt.accepted", old_id, "old-accepted"),
            generating=_message("vision.try_on.attempt.generating", old_id, "old-generating"),
        )
        assert old.is_owner

        async def replacement():
            prep = await registry.prepare_admission(
                attempt_id=new_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
            )
            assert prep.join_task is old_task
            return await registry.commit_prepared_admission(
                prep,
                accepted=_message("vision.try_on.attempt.accepted", new_id, "new-accepted"),
                generating=_message("vision.try_on.attempt.generating", new_id, "new-generating"),
                unavailable_terminal=_message(
                    "vision.try_on.attempt.failed", new_id, "new-unavailable"
                ),
                readiness=lambda: True,
            )

        replacement_task = asyncio.create_task(replacement())
        await asyncio.sleep(0.08)
        assert old_cleanup_reached.is_set()
        replacement_task.cancel()
        await asyncio.sleep(0)
        replacement_task.cancel()
        await asyncio.sleep(0.05)
        assert not replacement_task.done()

        blocked = asyncio.create_task(
            registry.admit(
                attempt_id=str(uuid4()),
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted=None,
                generating=None,
            )
        )
        await asyncio.sleep(0.05)
        assert not blocked.done(), "new attempt crossed cleanup before old child stopped"

        old_stop_released.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(replacement_task, timeout=1.0)
        admitted = await asyncio.wait_for(blocked, timeout=1.0)
        assert admitted.is_owner
        await registry.cancel_owner_and_join(admitted.receipt)

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
            generating=_message("vision.try_on.attempt.generating", oldest, "retry-generating"),
        )
        assert retry.is_owner
        await registry.cancel_owner_and_join(retry.receipt)

    asyncio.run(scenario())
