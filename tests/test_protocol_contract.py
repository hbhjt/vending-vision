import unittest

from app import should_deliver_profile_message
from vision.profile_mapper import vision_profile_to_protocol
from vision.profile_messages import build_presence_status
from vision.profile_state import protocol_occupancy_snapshot
from vision.protocol import envelope
from vision.schema import VisionProfile


class ProtocolContractTest(unittest.TestCase):
    def test_profile_contains_only_contract_fields(self):
        profile = vision_profile_to_protocol(
            VisionProfile(
                presence=True,
                height_cm=172,
                shoulder_width_cm=43,
                age=30,
                gender="male",
                body_type="medium",
                upper_color="dark",
            )
        )
        self.assertEqual(
            set(profile),
            {
                "personPresent",
                "heightCm",
                "shoulderWidthCm",
                "ageRange",
                "gender",
                "bodyType",
                "upperColor",
                "confidence",
            },
        )

    def test_capability_filter_includes_profile_push(self):
        self.assertFalse(should_deliver_profile_message("vision.profile_result", set()))
        self.assertTrue(
            should_deliver_profile_message("vision.profile_result", {"profile_push"})
        )
        self.assertFalse(
            should_deliver_profile_message("vision.presence_status", {"profile_push"})
        )

    def test_person_fallback_is_not_overwritten_by_unknown_top_detector(self):
        occupancy = protocol_occupancy_snapshot(
            {
                "present": True,
                "faceCount": 1,
                "topOccupancy": {"occupancy": "unknown", "confidence": 0.0},
            }
        )
        self.assertEqual(occupancy["state"], "single")

    def test_face_or_pose_evidence_overrides_top_yolo_none(self):
        occupancy = protocol_occupancy_snapshot(
            {
                "present": True,
                "bodyPresent": True,
                "topOccupancy": {"occupancy": "none", "confidence": 1.0},
            }
        )

        self.assertEqual(occupancy["state"], "single")
        self.assertEqual(occupancy["confidence"], 0.62)

    def test_multiple_evidence_has_priority_over_single(self):
        occupancy = protocol_occupancy_snapshot(
            {
                "present": True,
                "faceCount": 2,
                "topOccupancy": {"occupancy": "single", "confidence": 1.0},
            }
        )

        self.assertEqual(occupancy["state"], "multiple")

    def test_presence_envelope_keeps_contract_shape(self):
        payload = build_presence_status(
            event_id="event-1",
            state="approach",
            reason="person_present_but_not_close",
            proximity={"present": True},
            occupancy={"state": "single", "confidence": 0.82},
        )
        message = envelope("vision.presence_status", "status-1", payload)
        self.assertEqual(
            set(message), {"protocol", "type", "messageId", "timestamp", "payload"}
        )
        self.assertEqual(message["payload"]["occupancy"]["state"], "single")


if __name__ == "__main__":
    unittest.main()
