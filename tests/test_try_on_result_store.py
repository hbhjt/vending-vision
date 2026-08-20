import asyncio

import pytest

from vision.try_on_attempt_registry import TryOnAttemptRegistry
from vision.try_on_result_store import TryOnResultStore, ResultAdmissionError


def _result(token: str, size: int) -> dict:
    return {
        "token": token,
        "bytes": b"x" * size,
        "reference": f"http://127.0.0.1/r?token={token}",
        "digest": "sha256:" + "a" * 64,
        "contentType": "image/png",
        "width": 1,
        "height": 1,
    }


def test_try_on_result_store_evicts_oldest_entries_to_both_caps():
    async def scenario():
        store = TryOnResultStore(max_count=3, max_bytes=8, ttl_seconds=60)
        for attempt_id, size in (("old-1", 3), ("old-2", 3), ("old-3", 3)):
            await store.admit(attempt_id, _result(attempt_id, size))

        await store.admit("new", _result("new", 4))
        assert [entry.attempt_id for entry in store.snapshot()] == ["old-3", "new"]
        assert store.count == 2
        assert store.aggregate_bytes == 7

    asyncio.run(scenario())


def test_try_on_result_store_reports_every_capacity_eviction_in_order():
    async def scenario():
        store = TryOnResultStore(max_count=3, max_bytes=10, ttl_seconds=60)
        for attempt_id in ("old-1", "old-2", "old-3"):
            await store.admit(attempt_id, _result(attempt_id, 3))

        admission = await store.admit("new", _result("new", 7))

        assert admission.evicted_attempt_ids == ("old-1", "old-2")
        assert [entry.attempt_id for entry in store.snapshot()] == ["old-3", "new"]
        assert store.aggregate_bytes == 10

    asyncio.run(scenario())


def test_try_on_result_store_overwrite_failure_preserves_old_entry():
    async def scenario():
        store = TryOnResultStore(max_count=2, max_bytes=10, ttl_seconds=60)
        await store.admit("same", _result("old-token", 4))
        with pytest.raises(ResultAdmissionError, match="result_store_too_large"):
            await store.admit("same", _result("new-token", 11))
        old = await store.get("same", "old-token")
        assert old is not None and old.bytes == b"x" * 4
        assert await store.get("same", "new-token") is None
        assert store.count == 1 and store.total_bytes == 4

    asyncio.run(scenario())


def test_try_on_result_store_expiry_is_complete_and_reads_do_not_renew_or_reorder():
    async def scenario():
        now = [100.0]
        store = TryOnResultStore(max_count=10, max_bytes=100, ttl_seconds=5, clock=lambda: now[0])
        await store.admit("first", _result("first-token", 2))
        now[0] += 1
        await store.admit("second", _result("second-token", 2))
        assert await store.get("first", "first-token") is not None
        assert [entry.attempt_id for entry in store.snapshot()] == ["first", "second"]
        now[0] = 106
        assert await store.get("first", "first-token") is None
        assert await store.get("second", "second-token") is None
        assert store.count == 0 and store.total_bytes == 0

    asyncio.run(scenario())


def test_terminal_commit_failure_has_one_single_path_failed_terminal_and_no_orphan_grant():
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=2, max_bytes=4, ttl_seconds=60),
        )
        attempt_id = "attempt"
        accepted = {
            "type": "vision.try_on.attempt.accepted",
            "messageId": "accepted",
            "payload": {"attemptId": attempt_id},
        }
        completed = {
            "type": "vision.try_on.attempt.completed",
            "messageId": "completed",
            "payload": {"attemptId": attempt_id, "result": {}},
        }
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=accepted,
            generating=None,
        )
        transition = await registry.commit_terminal_transition(
            admission.receipt,
            completed,
            _result("secret", 5),
        )
        assert transition is not None
        assert transition.message["type"] == "vision.try_on.attempt.failed"
        assert transition.message["payload"] == {
            "attemptId": attempt_id,
            "reason": "try_on_failed",
        }
        assert await registry.get_result(attempt_id, "secret") is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "terminal_type",
    ["vision.try_on.attempt.failed", "vision.try_on.attempt.canceled"],
)
def test_registry_never_admits_a_result_for_a_noncompleted_terminal(terminal_type):
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=2, max_bytes=16, ttl_seconds=60),
        )
        attempt_id = "failed-attempt"
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted={"type": "vision.try_on.attempt.accepted", "payload": {"attemptId": attempt_id}},
            generating=None,
        )

        transition = await registry.commit_terminal_transition(
            admission.receipt,
            {
                "type": terminal_type,
                "payload": {"attemptId": attempt_id, "reason": "try_on_failed"},
            },
            _result("sentinel-result-token", 4),
        )

        assert transition is not None
        assert transition.message["type"] == terminal_type
        assert await registry.get_result(attempt_id, "sentinel-result-token") is None
        replay = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(),
            send_lock=asyncio.Lock(),
            task=asyncio.current_task(),
            accepted=None,
            generating=None,
        )
        assert replay.replay == [transition.message]

    asyncio.run(scenario())


def test_capacity_eviction_removes_dead_completed_replay_before_new_is_visible():
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=2, max_bytes=1024, ttl_seconds=60),
        )

        async def complete(attempt_id, token, size):
            admission = await registry.admit(
                attempt_id=attempt_id,
                websocket=object(),
                send_lock=asyncio.Lock(),
                task=asyncio.current_task(),
                accepted={"type": "vision.try_on.attempt.accepted", "payload": {"attemptId": attempt_id}},
                generating=None,
            )
            assert admission.is_owner
            result = _result(token, size)
            message = {
                "type": "vision.try_on.attempt.completed",
                "payload": {"attemptId": attempt_id, "result": {key: value for key, value in result.items() if key not in {"token", "bytes"}}},
            }
            transition = await registry.commit_terminal_transition(admission.receipt, message, result)
            assert transition is not None
            return transition.message

        old = await complete("old", "old-token", 950)
        new = await complete("new", "new-token", 92)

        assert await registry.get_result("old", "old-token") is None
        assert await registry.get_result("new", "new-token") is not None
        assert (await registry.admit(
            attempt_id="new", websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(), accepted=None, generating=None,
        )).replay == [new]
        old_readmission = await registry.admit(
            attempt_id="old", websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(), accepted=None, generating=None,
        )
        assert old_readmission.is_owner
        assert old_readmission.replay == []
        assert await registry.commit_terminal(
            old_readmission.receipt,
            {"type": "vision.try_on.attempt.failed", "payload": {"attemptId": "old", "reason": "try_on_failed"}},
        )
        assert old["type"] == "vision.try_on.attempt.completed"

    asyncio.run(scenario())


def test_result_expiry_removes_its_completed_terminal_before_same_id_replay():
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=2, max_bytes=16, ttl_seconds=0.01),
        )
        attempt_id = "expired-result"
        admission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(), send_lock=asyncio.Lock(), task=asyncio.current_task(),
            accepted={"type": "vision.try_on.attempt.accepted", "payload": {"attemptId": attempt_id}},
            generating=None,
        )
        result = _result("expired-token", 4)
        await registry.commit_terminal_transition(
            admission.receipt,
            {"type": "vision.try_on.attempt.completed", "payload": {"attemptId": attempt_id, "result": {}}},
            result,
        )
        await asyncio.sleep(0.02)

        readmission = await registry.admit(
            attempt_id=attempt_id,
            websocket=object(), send_lock=asyncio.Lock(), task=asyncio.current_task(),
            accepted=None, generating=None,
        )
        assert readmission.is_owner
        assert readmission.replay == []

    asyncio.run(scenario())


def test_failed_new_result_admission_keeps_existing_terminal_and_grant_intact():
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=1, max_bytes=4, ttl_seconds=60),
        )

        old_admission = await registry.admit(
            attempt_id="old", websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(), accepted=None, generating=None,
        )
        old_result = _result("old-token", 4)
        old_completed = {
            "type": "vision.try_on.attempt.completed",
            "payload": {"attemptId": "old", "result": {}},
        }
        await registry.commit_terminal_transition(
            old_admission.receipt, old_completed, old_result
        )

        new_admission = await registry.admit(
            attempt_id="new", websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(), accepted=None, generating=None,
        )
        transition = await registry.commit_terminal_transition(
            new_admission.receipt,
            {"type": "vision.try_on.attempt.completed", "payload": {"attemptId": "new", "result": {}}},
            _result("new-token", 5),
        )

        assert transition is not None
        assert transition.message["type"] == "vision.try_on.attempt.failed"
        assert await registry.get_result("old", "old-token") is not None
        old_replay = await registry.admit(
            attempt_id="old", websocket=object(), send_lock=asyncio.Lock(),
            task=asyncio.current_task(), accepted=None, generating=None,
        )
        assert old_replay.replay == [old_completed]
        assert await registry.get_result("new", "new-token") is None

    asyncio.run(scenario())


def test_evicted_attempt_duplicate_race_has_one_new_owner_and_no_stale_replay():
    async def scenario():
        registry = TryOnAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=TryOnResultStore(max_count=1, max_bytes=8, ttl_seconds=60),
        )

        async def complete(attempt_id, token):
            admission = await registry.admit(
                attempt_id=attempt_id, websocket=object(), send_lock=asyncio.Lock(),
                task=asyncio.current_task(), accepted=None, generating=None,
            )
            await registry.commit_terminal_transition(
                admission.receipt,
                {"type": "vision.try_on.attempt.completed", "payload": {"attemptId": attempt_id, "result": {}}},
                _result(token, 4),
            )

        await complete("old", "old-token")
        await complete("new", "new-token")
        duplicates = await asyncio.gather(*(
            registry.admit(
                attempt_id="old", websocket=object(), send_lock=asyncio.Lock(),
                task=asyncio.current_task(), accepted=None, generating=None,
            )
            for _ in range(8)
        ))

        owners = [admission for admission in duplicates if admission.is_owner]
        assert len(owners) == 1
        assert all(
            all(message.get("type") != "vision.try_on.attempt.completed" for message in admission.replay)
            for admission in duplicates
        )
        await registry.commit_terminal(
            owners[0].receipt,
            {"type": "vision.try_on.attempt.failed", "payload": {"attemptId": "old", "reason": "try_on_failed"}},
        )

    asyncio.run(scenario())
