import asyncio

import pytest

from vision.fast_attempt_registry import FastAttemptRegistry
from vision.fast_result_store import FastResultStore, ResultAdmissionError


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


def test_fast_result_store_evicts_oldest_entries_to_both_caps():
    async def scenario():
        store = FastResultStore(max_count=3, max_bytes=8, ttl_seconds=60)
        for attempt_id, size in (("old-1", 3), ("old-2", 3), ("old-3", 3)):
            await store.admit(attempt_id, _result(attempt_id, size))

        await store.admit("new", _result("new", 4))
        assert [entry.attempt_id for entry in store.snapshot()] == ["old-3", "new"]
        assert store.count == 2
        assert store.aggregate_bytes == 7

    asyncio.run(scenario())


def test_fast_result_store_overwrite_failure_preserves_old_entry():
    async def scenario():
        store = FastResultStore(max_count=2, max_bytes=10, ttl_seconds=60)
        await store.admit("same", _result("old-token", 4))
        with pytest.raises(ResultAdmissionError, match="result_store_too_large"):
            await store.admit("same", _result("new-token", 11))
        old = await store.get("same", "old-token")
        assert old is not None and old.bytes == b"x" * 4
        assert await store.get("same", "new-token") is None
        assert store.count == 1 and store.total_bytes == 4

    asyncio.run(scenario())


def test_fast_result_store_expiry_is_complete_and_reads_do_not_renew_or_reorder():
    async def scenario():
        now = [100.0]
        store = FastResultStore(max_count=10, max_bytes=100, ttl_seconds=5, clock=lambda: now[0])
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


def test_terminal_commit_failure_has_one_failed_terminal_and_no_orphan_grant():
    async def scenario():
        registry = FastAttemptRegistry(
            terminal_ttl_seconds=60,
            result_store=FastResultStore(max_count=2, max_bytes=4, ttl_seconds=60),
        )
        attempt_id = "attempt"
        accepted = {
            "type": "vision.try_on.attempt.accepted",
            "messageId": "accepted",
            "payload": {"attemptId": attempt_id, "mode": "fast"},
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
            progress=None,
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
            "reason": "fast_failed",
        }
        assert await registry.get_result(attempt_id, "secret") is None

    asyncio.run(scenario())
