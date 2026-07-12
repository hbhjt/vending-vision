import unittest

from vision.profile_aggregation import aggregate_samples, build_quality
from vision.profile_messages import build_presence_status
from vision.profile_state import (
    ProfileOccupancyGate,
    ResponsiveOccupancyFilter,
    protocol_occupancy_snapshot,
)
from vision.schema import VisionProfile


def sample(profile, confidence=0.8, valid=True):
    return {
        "profile": profile,
        "protocolProfile": {
            "personPresent": profile.presence,
            "heightCm": profile.height_cm,
            "shoulderWidthCm": profile.shoulder_width_cm,
            "ageRange": "adult" if profile.age else "unknown",
            "gender": profile.gender,
            "bodyType": "regular",
            "upperColor": profile.upper_color,
            "confidence": confidence,
        },
        "quality": {
            "brightness": 100.0,
            "sharpness": 120.0,
        },
        "proximity": {
            "personPresent": True,
            "facePresent": True,
        },
        "source": "close_sample",
        "valid": valid,
    }


class ProfileCoreTest(unittest.TestCase):
    def test_protocol_occupancy_detects_multiple_people(self):
        occupancy = protocol_occupancy_snapshot(
            {
                "present": True,
                "personCount": 2,
                "faceCount": 1,
            }
        )

        self.assertEqual(occupancy["state"], "multiple")
        self.assertGreaterEqual(occupancy["confidence"], 0.7)

    def test_responsive_filter_enters_immediately_and_leaves_after_two_misses(self):
        occupancy_filter = ResponsiveOccupancyFilter(absent_min_frames=2)
        present = {
            "present": True,
            "faceCount": 1,
            "facePresent": True,
            "topOccupancy": {"occupancy": "none", "confidence": 1.0},
        }
        empty = {
            "present": False,
            "faceCount": 0,
            "topOccupancy": {"occupancy": "none", "confidence": 1.0},
        }

        first = occupancy_filter.update(present)
        first_miss = occupancy_filter.update(empty)
        second_miss = occupancy_filter.update(empty)

        self.assertEqual(first["state"], "single")
        self.assertEqual(first_miss["state"], "single")
        self.assertEqual(second_miss["state"], "none")

    def test_occupancy_gate_locks_after_profile_push(self):
        gate = ProfileOccupancyGate()
        gate.mark_present()
        self.assertTrue(gate.can_trigger())

        gate.mark_pushed("event-1")
        self.assertFalse(gate.can_trigger())
        self.assertEqual(gate.public_state()["lastEventId"], "event-1")

        gate.mark_target_changed()
        self.assertTrue(gate.can_trigger())
        self.assertEqual(gate.public_state()["state"], "tracking")

    def test_aggregate_samples_prefers_valid_profile_values(self):
        profile = VisionProfile(
            age=30,
            gender="male",
            height_cm=172.0,
            shoulder_width_cm=43.0,
            body_type="medium",
            upper_color="dark",
            presence=True,
        )

        aggregated = aggregate_samples([sample(profile, confidence=0.82)])

        self.assertIsNotNone(aggregated)
        _, protocol_profile = aggregated
        self.assertEqual(protocol_profile["personPresent"], True)
        self.assertEqual(protocol_profile["heightCm"], 172.0)
        self.assertGreaterEqual(protocol_profile["confidence"], 0.8)

    def test_aggregate_samples_ignores_person_only_frames(self):
        profile = VisionProfile(
            age=None,
            gender="unknown",
            height_cm=None,
            shoulder_width_cm=None,
            body_type="unknown",
            upper_color="unknown",
            presence=True,
        )

        aggregated = aggregate_samples([sample(profile, confidence=0.5)])

        self.assertIsNone(aggregated)

    def test_build_quality_marks_low_confidence_unusable(self):
        quality = build_quality(
            {
                "heightCm": None,
                "bodyType": "unknown",
                "ageRange": "unknown",
                "gender": "unknown",
                "confidence": 0.2,
            },
            samples=[],
            valid_count=0,
            min_valid_frames=1,
        )

        self.assertEqual(quality["overall"], "poor")
        self.assertFalse(quality["profileUsable"])
        self.assertEqual(quality["notUsableReason"], "low_confidence")

    def test_build_quality_requires_multiple_valid_field_frames(self):
        quality = build_quality(
            {
                "heightCm": 172,
                "bodyType": "regular",
                "ageRange": "unknown",
                "gender": "unknown",
                "confidence": 0.6,
            },
            samples=[{"valid": True}],
            valid_count=1,
            min_valid_frames=2,
        )

        self.assertFalse(quality["profileUsable"])
        self.assertEqual(quality["notUsableReason"], "insufficient_quality")

    def test_partial_profile_at_configured_confidence_is_usable(self):
        quality = build_quality(
            {
                "heightCm": None,
                "bodyType": "regular",
                "ageRange": "unknown",
                "gender": "unknown",
                "confidence": 0.3,
            },
            samples=[{"valid": True}, {"valid": True}],
            valid_count=2,
            min_valid_frames=2,
        )

        self.assertEqual(quality["overall"], "fair")
        self.assertTrue(quality["profileUsable"])

    def test_presence_status_omits_ambient_light_when_absent(self):
        payload = build_presence_status(
            event_id="event-1",
            state="empty",
            reason="no_person",
            proximity={"present": False},
        )

        self.assertEqual(payload["state"], "empty")
        self.assertNotIn("ambientLight", payload)


if __name__ == "__main__":
    unittest.main()
