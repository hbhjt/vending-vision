import unittest

from vision.config import settings
from vision.presence_runtime import PresenceRuntime
from vision.profile_state import get_occupancy_gate, reset_active_track


class FakeMonitor:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0

    def check_once(
        self,
        return_image=False,
        camera_role="top",
        return_source=False,
    ):
        value = self.sequence[min(self.index, len(self.sequence) - 1)]
        self.index += 1
        if return_source:
            return value, None, None
        return value, None


def single_proximity(close=False, center_x=0.5):
    return {
        "present": True,
        "close": close,
        "personPresent": True,
        "personCount": 1,
        "faceCount": 0,
        "largestPersonRatio": 0.08,
        "largestPersonBox": {
            "centerX": center_x,
            "centerY": 0.5,
        },
        "topOccupancy": {
            "occupancy": "single",
            "confidence": 0.8,
        },
    }


def empty_proximity():
    return {
        "present": False,
        "close": False,
        "personPresent": False,
        "personCount": 0,
        "faceCount": 0,
        "topOccupancy": {
            "occupancy": "none",
            "confidence": 1.0,
        },
    }


class PresenceRuntimeTest(unittest.TestCase):
    def setUp(self):
        gate = get_occupancy_gate()
        for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
            gate.mark_absent()
        reset_active_track()

    def tearDown(self):
        gate = get_occupancy_gate()
        for _ in range(settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES):
            gate.mark_absent()
        reset_active_track()

    def test_stable_single_starts_pre_sampling_before_close(self):
        runtime = PresenceRuntime(FakeMonitor([single_proximity(close=False)]))

        result = runtime.poll(False, False, False)

        self.assertIsNotNone(result.candidate)
        self.assertFalse(result.candidate.proximity["close"])
        self.assertTrue(runtime.is_candidate_valid(result.candidate.generation))

    def test_target_change_unlocks_profile_after_two_frames(self):
        runtime = PresenceRuntime(
            FakeMonitor([
                single_proximity(center_x=0.9),
                single_proximity(center_x=0.9),
            ])
        )
        runtime.profiled_signature = {
            "source": "person",
            "centerX": 0.1,
            "centerY": 0.5,
            "areaRatio": 0.08,
            "count": 1,
        }
        gate = get_occupancy_gate()
        gate.mark_present()
        gate.mark_pushed("old-profile")

        runtime.poll(False, False, False)
        self.assertFalse(gate.can_trigger())
        runtime.poll(False, False, False)

        self.assertTrue(gate.can_trigger())

    def test_departure_is_emitted_on_first_confirmed_empty_snapshot(self):
        runtime = PresenceRuntime(
            FakeMonitor([
                single_proximity(),
                empty_proximity(),
                empty_proximity(),
            ])
        )
        runtime.poll(False, False, True)

        first_miss = runtime.poll(False, False, True)
        result = runtime.poll(False, False, True)

        self.assertIsNone(first_miss.update)
        self.assertIsNotNone(result.update)
        self.assertEqual(result.update["message_type"], "vision.person_departed")


if __name__ == "__main__":
    unittest.main()
