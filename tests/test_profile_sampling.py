import unittest
from unittest.mock import patch

import numpy as np

import vision.profile_sampling as profile_sampling


class FakePersonDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


class FakeFaceDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect_faces(self, image):
        return self.detections


class ProfileSamplingTest(unittest.TestCase):
    def tearDown(self):
        profile_sampling._person_detector = None
        profile_sampling._face_detector = None

    def test_quality_score_prefers_good_face_and_person(self):
        image = np.full((240, 320, 3), 128, dtype=np.uint8)
        profile_sampling._person_detector = FakePersonDetector(
            [{"box": [80, 40, 120, 180], "score": 0.9}]
        )
        profile_sampling._face_detector = FakeFaceDetector(
            [{"bbox": (120, 60, 60, 60), "score": 0.9, "landmarks": None}]
        )

        quality = profile_sampling.score_frame_quality(image)

        self.assertTrue(quality["personDetected"])
        self.assertTrue(quality["faceDetected"])
        self.assertGreater(quality["qualityScore"], 0.5)

    def test_low_face_score_is_not_face_detected(self):
        image = np.full((240, 320, 3), 128, dtype=np.uint8)
        profile_sampling._person_detector = FakePersonDetector(
            [{"box": [80, 40, 120, 180], "score": 0.9}]
        )
        profile_sampling._face_detector = FakeFaceDetector(
            [{"bbox": (120, 60, 60, 60), "score": 0.1, "landmarks": None}]
        )

        quality = profile_sampling.score_frame_quality(image)

        self.assertFalse(quality["faceDetected"])
        self.assertLess(quality["faceScore"], 0.45)

    def test_pre_sampling_finishes_with_two_valid_frames_without_close(self):
        sample = {
            "valid": True,
            "quality": {"qualityScore": 0.5},
            "summary": {},
            "source": "candidate",
        }
        responsive_config = {
            "duration_sec": 1.0,
            "early_finish_after_sec": 0.0,
            "target_fps": 100,
            "min_good_frames": 2,
            "max_good_frames": 6,
        }

        with patch.object(
            profile_sampling.settings,
            "PROFILE_SAMPLING_CONFIG",
            responsive_config,
        ), patch.object(
            profile_sampling.settings,
            "FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS",
            1,
        ), patch.object(
            profile_sampling,
            "sample_frame",
            return_value=sample,
        ) as sample_frame:
            samples = profile_sampling.collect_best_profile_samples(
                close_enough=False,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual(sample_frame.call_count, 2)

    def test_close_signal_can_finish_before_pre_sampling_floor(self):
        sample = {
            "valid": True,
            "quality": {"qualityScore": 0.5},
            "summary": {},
            "source": "candidate",
        }
        responsive_config = {
            "duration_sec": 1.0,
            "early_finish_after_sec": 10.0,
            "target_fps": 100,
            "min_good_frames": 2,
            "max_good_frames": 6,
        }

        with patch.object(
            profile_sampling.settings,
            "PROFILE_SAMPLING_CONFIG",
            responsive_config,
        ), patch.object(
            profile_sampling.settings,
            "FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS",
            1,
        ), patch.object(
            profile_sampling,
            "sample_frame",
            return_value=sample,
        ):
            samples = profile_sampling.collect_best_profile_samples(
                close_validator=lambda: True,
            )

        self.assertEqual(len(samples), 2)


if __name__ == "__main__":
    unittest.main()
