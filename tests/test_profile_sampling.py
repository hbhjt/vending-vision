import unittest
import threading
import time
from unittest.mock import patch

import numpy as np

import vision.profile_sampling as profile_sampling
from vision.camera_owner import (
    acquire_front_camera,
    front_camera_io_lock,
    get_front_camera_owner,
    release_front_camera,
)
from vision.schema import VisionProfile


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

    def test_quality_reuses_profile_without_duplicate_detectors(self):
        image = np.full((240, 320, 3), 128, dtype=np.uint8)
        profile_sampling._person_detector = FakePersonDetector(None)
        profile_sampling._face_detector = FakeFaceDetector(None)
        profile = VisionProfile(
            age=30,
            gender="female",
            body_type="medium",
            presence=True,
        )

        quality = profile_sampling.score_frame_quality(image, profile=profile)

        self.assertTrue(quality["personDetected"])
        self.assertTrue(quality["faceDetected"])
        self.assertEqual(quality["personScore"], 1.0)
        self.assertEqual(quality["faceScore"], 1.0)

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

    def test_profile_three_frame_sequence_keeps_try_on_out_until_all_gaps_finish(self):
        frame = np.full((24, 32, 3), 128, dtype=np.uint8)
        gap_started = threading.Event()
        allow_gap = threading.Event()
        sequence_done = threading.Event()
        try_on_acquired = threading.Event()
        try_on_inserted_during_sequence = []
        reads = []
        lease_token = "profile:test-sequence"
        acquired = acquire_front_camera(
            "vision", reason="profile_test", lease_token=lease_token
        )
        self.assertTrue(acquired["ok"])

        def read_frame(_role, warmup_frames=1):
            reads.append(len(reads) + 1)
            return frame.copy(), {"source": "recorded_video", "frameIndex": len(reads)}

        real_sleep = time.sleep

        def interval_barrier(_seconds):
            gap_started.set()
            self.assertTrue(allow_gap.wait(timeout=1.0))

        def try_on_reader():
            self.assertTrue(gap_started.wait(timeout=1.0))
            with front_camera_io_lock():
                try_on_inserted_during_sequence.append(not sequence_done.is_set())
                try_on_acquired.set()

        config = {
            "duration_sec": 1.0,
            "early_finish_after_sec": 0.0,
            "target_fps": 100,
            "min_good_frames": 3,
            "max_good_frames": 3,
        }
        worker_result = {}

        def collect():
            try:
                worker_result["samples"] = profile_sampling.collect_best_profile_samples()
            finally:
                sequence_done.set()

        with patch.object(profile_sampling.settings, "PROFILE_SAMPLING_CONFIG", config), patch.object(
            profile_sampling.settings, "FRONT_CAMERA_PROFILE_SAMPLE_COUNT", 3
        ), patch.object(
            profile_sampling.settings, "FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS", 1
        ), patch.object(
            profile_sampling, "read_camera_with_source", side_effect=read_frame
        ), patch.object(
            profile_sampling, "infer_image", return_value=VisionProfile(age=30, gender="female", presence=True)
        ), patch.object(
            profile_sampling, "score_frame_quality", return_value={
                "faceDetected": True, "qualityScore": 0.8, "brightness": 128.0, "sharpness": 10.0,
            }
        ), patch.object(profile_sampling.time, "sleep", side_effect=interval_barrier):
            collector = threading.Thread(target=collect)
            contender = threading.Thread(target=try_on_reader)
            collector.start()
            contender.start()
            self.assertTrue(gap_started.wait(timeout=1.0))
            real_sleep(0.05)
            self.assertFalse(try_on_acquired.is_set())
            allow_gap.set()
            collector.join(timeout=2.0)
            contender.join(timeout=2.0)

        try:
            self.assertEqual(len(worker_result["samples"]), 3)
            self.assertEqual(reads, [1, 2, 3])
            self.assertEqual(try_on_inserted_during_sequence, [False])
            self.assertEqual(get_front_camera_owner()["leaseToken"], lease_token)
        finally:
            release_front_camera("vision", reason="profile_test_done", lease_token=lease_token)


if __name__ == "__main__":
    unittest.main()
