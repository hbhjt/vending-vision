import asyncio
import time

import app as vision_app
from vision.fast_attempt_registry import AttemptSubscriber, TerminalTransition


class _HealthySocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class _SlowSocket:
    def __init__(self):
        self.entered = asyncio.Event()

    async def send_json(self, message):
        self.entered.set()
        await asyncio.sleep(10)


def test_fast_terminal_publish_times_out_waiting_for_preoccupied_send_lock(monkeypatch):
    """The terminal deadline includes acquiring a subscriber's send lock."""
    async def scenario():
        monkeypatch.setattr(vision_app, "_FAST_TERMINAL_SEND_TIMEOUT_SECONDS", 0.01)
        detached = []

        async def detach(websocket):
            detached.append(websocket)

        monkeypatch.setattr(vision_app._fast_attempt_registry, "detach_subscriber", detach)
        slow = _SlowSocket()
        send_lock = asyncio.Lock()
        await send_lock.acquire()
        message = {"type": "vision.try_on.attempt.failed", "payload": {"attemptId": "a"}}
        transition = TerminalTransition(
            message=message,
            subscribers=[AttemptSubscriber(1, slow, send_lock)],
        )

        started = time.monotonic()
        await vision_app._publish_fast_transition(transition)
        elapsed = time.monotonic() - started
        send_lock.release()

        assert detached == [slow]
        assert not slow.entered.is_set()
        assert elapsed < 0.2

    asyncio.run(scenario())


def test_fast_terminal_publish_detaches_slow_subscriber_without_blocking_healthy(monkeypatch):
    """A slow terminal subscriber cannot block other subscribers or replacement cleanup."""
    async def scenario():
        monkeypatch.setattr(vision_app, "_FAST_TERMINAL_SEND_TIMEOUT_SECONDS", 0.01)
        detached = []

        async def detach(websocket):
            detached.append(websocket)

        monkeypatch.setattr(vision_app._fast_attempt_registry, "detach_subscriber", detach)
        healthy = _HealthySocket()
        slow = _SlowSocket()
        message = {"type": "vision.try_on.attempt.failed", "payload": {"attemptId": "a"}}
        transition = TerminalTransition(
            message=message,
            subscribers=[
                AttemptSubscriber(1, slow, asyncio.Lock()),
                AttemptSubscriber(2, healthy, asyncio.Lock()),
            ],
        )

        started = time.monotonic()
        await vision_app._publish_fast_transition(transition)
        elapsed = time.monotonic() - started

        assert healthy.messages == [message]
        assert detached == [slow]
        assert elapsed < 0.2

    asyncio.run(scenario())
