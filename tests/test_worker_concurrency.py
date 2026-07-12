import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

import app


class FakePresenceRuntime:
    def __init__(self):
        self.finished = []

    def is_candidate_valid(self, generation):
        return True

    def latest(self):
        return {"occupancy": {"state": "single", "confidence": 0.8}}

    def finish_collection(self, generation, pushed=False):
        self.finished.append((generation, pushed))


class WorkerConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_runtime = app.get_presence_runtime
        self.original_collect = app.collect_front_profile_update
        self.original_broadcast = app.broadcast_profile_update
        self.runtime = FakePresenceRuntime()
        app.get_presence_runtime = lambda: self.runtime
        self.broadcasts = []

        async def record(update):
            self.broadcasts.append(update)

        app.broadcast_profile_update = record

    async def asyncTearDown(self):
        app.get_presence_runtime = self.original_runtime
        app.collect_front_profile_update = self.original_collect
        app.broadcast_profile_update = self.original_broadcast
        app._profile_worker_task = None
        app._profile_cancel_event = None

    def candidate(self):
        return SimpleNamespace(
            generation=1,
            event_id="event-1",
            proximity={"present": True, "close": True},
            track=object(),
            ambient_light=None,
        )

    async def test_slow_profile_collection_does_not_block_event_loop(self):
        def slow_collect(*args):
            time.sleep(0.1)
            return None

        app.collect_front_profile_update = slow_collect
        task = asyncio.create_task(
            app.profile_collection_worker(self.candidate(), threading.Event())
        )
        await asyncio.sleep(0.01)
        self.assertFalse(task.done())
        await task

    async def test_cancelled_collection_does_not_broadcast_stale_result(self):
        def slow_result(*args):
            time.sleep(0.05)
            return {
                "message_type": "vision.profile_result",
                "payload": {"eventId": "event-1"},
            }

        app.collect_front_profile_update = slow_result
        cancel_event = threading.Event()
        task = asyncio.create_task(
            app.profile_collection_worker(self.candidate(), cancel_event)
        )
        await asyncio.sleep(0.01)
        cancel_event.set()
        await task
        self.assertEqual(self.broadcasts, [])


if __name__ == "__main__":
    unittest.main()
