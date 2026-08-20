"""Public tracer bullets for the single-owner acquisition seam."""

import asyncio

import numpy as np
import pytest

from vision.acquisition_observer import AcquisitionObservation
from vision.acquisition_session import AcquisitionSession


def test_slow_observer_does_not_freeze_preview_and_capture_uses_a_newer_checked_frame():
    """A blocked inference boundary cannot turn preview into a stale still."""

    async def scenario():
        first_observation_started = asyncio.Event()
        release_first_observation = asyncio.Event()
        preview_advanced = asyncio.Event()
        previews = []
        published = []
        source_sequence = 0
        observed_sequences = []

        async def read_frame(_timeout):
            nonlocal source_sequence
            source_sequence += 1
            frame = np.full((12, 12, 3), source_sequence, dtype=np.uint8)
            return frame, {"source": "recorded_video", "sequence": source_sequence}

        async def observe(frame, _timeout):
            observed_sequences.append(int(frame[0, 0, 0]))
            if len(observed_sequences) == 1:
                first_observation_started.set()
                await release_first_observation.wait()
            return AcquisitionObservation(b"unused", "single", True)

        async def open_preview(_attempt_id, jpeg):
            previews.append(jpeg)
            return "preview-token"

        async def update_preview(_attempt_id, _token, jpeg):
            previews.append(jpeg)
            if len({bytes(value) for value in previews}) >= 3:
                preview_advanced.set()
            return True

        async def publish(*fact):
            published.append(fact)

        session = AcquisitionSession(
            attempt_id="attempt",
            read_frame=read_frame,
            observe=observe,
            preview_open=open_preview,
            preview_update=update_preview,
            publish=publish,
            stable_seconds=0.02,
            timeout_seconds=2,
            preview_interval_seconds=0.001,
        )
        acquisition = asyncio.create_task(
            session.acquire(manual_requested=lambda: asyncio.sleep(0, result=False), consume_manual=lambda: asyncio.sleep(0))
        )
        await asyncio.wait_for(first_observation_started.wait(), timeout=0.5)
        await asyncio.wait_for(preview_advanced.wait(), timeout=0.5)
        # This assertion occurs before releasing the inference barrier: the
        # public preview adapter has received several distinct JPEG bodies
        # while no inference answer was available.
        preview_identities_during_barrier = {bytes(value) for value in previews}
        assert len(preview_identities_during_barrier) >= 3
        first_checked = observed_sequences[0]
        release_first_observation.set()
        captured = await asyncio.wait_for(acquisition, timeout=1)

        assert captured.source["sequence"] > first_checked
        assert captured.frame_id.startswith("frame-")
        assert published[-1][2] == "counting_down"

    asyncio.run(scenario())


def test_countdown_truth_restarts_after_instability_and_captures_only_after_one_finishes():
    """The public timeline carries a full reset 3→2→1, not a frozen label."""

    async def scenario():
        started = asyncio.get_running_loop().time()
        events = []
        captured_at = None

        async def read_frame(_timeout):
            return np.full((10, 10, 3), 90, dtype=np.uint8), {"source": "recorded_video"}

        async def observe(_frame, _timeout):
            elapsed = asyncio.get_running_loop().time() - started
            # One fact-based unstable interval interrupts the first hold; no
            # call-count sequencing or fixed test sleep is involved.
            return AcquisitionObservation(b"unused", "single", not 0.85 <= elapsed < 1.05)

        async def preview_open(_attempt_id, _jpeg):
            return "preview-token"

        async def preview_update(_attempt_id, _token, _jpeg):
            return True

        async def publish(_token, occupancy, guidance, aligned, remaining):
            events.append((asyncio.get_running_loop().time() - started, occupancy, guidance, aligned, remaining))

        session = AcquisitionSession(
            attempt_id="attempt",
            read_frame=read_frame,
            observe=observe,
            preview_open=preview_open,
            preview_update=preview_update,
            publish=publish,
            stable_seconds=3,
            timeout_seconds=6,
            preview_interval_seconds=0.05,
        )
        captured = await session.acquire(
            manual_requested=lambda: asyncio.sleep(0, result=False),
            consume_manual=lambda: asyncio.sleep(0),
        )
        captured_at = asyncio.get_running_loop().time() - started

        reset_at = next(time for time, _occupancy, guidance, _aligned, _remaining in events if guidance == "align")
        final = [event for event in events if event[0] > reset_at and event[2] == "counting_down"]
        buckets = [(time, max(1, (remaining + 999) // 1000)) for time, _o, _g, _a, remaining in final]
        assert {bucket for _time, bucket in buckets} >= {1, 2, 3}
        for bucket in (3, 2, 1):
            visible = [time for time, value in buckets if value == bucket]
            assert max(visible) - min(visible) >= 0.7
        final_zero = max(time for time, _o, _g, _a, remaining in final if remaining == 0)
        first_final_three = min(time for time, bucket in buckets if bucket == 3)
        assert final_zero - first_final_three >= 2.5
        assert captured_at >= final_zero
        assert captured.frame_id.startswith("frame-")

    asyncio.run(scenario())


def test_manual_capture_does_not_bypass_single_person_alignment():
    async def scenario():
        published = []
        async def frame(_timeout): return np.zeros((8, 8, 3), dtype=np.uint8), {}
        async def observe(_frame, _timeout): return AcquisitionObservation(b"x", "single", False)
        async def open(_id, _jpeg): return "token"
        async def update(_id, _token, _jpeg): return True
        async def publish(*fact): published.append(fact)
        session = AcquisitionSession(attempt_id="a", read_frame=frame, observe=observe, preview_open=open, preview_update=update, publish=publish, timeout_seconds=0.05, preview_interval_seconds=0.001)
        with pytest.raises(asyncio.TimeoutError):
            await session.acquire(manual_requested=lambda: asyncio.sleep(0, result=True), consume_manual=lambda: asyncio.sleep(0))
        assert any(fact[2] == "align" for fact in published)
    asyncio.run(scenario())
