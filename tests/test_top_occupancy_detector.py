import unittest

import numpy as np

from vision.top_occupancy_detector import TopOccupancyDetector


class FakePersonDetector:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0

    def detect(self, image):
        if self.index >= len(self.sequence):
            return self.sequence[-1]

        value = self.sequence[self.index]
        self.index += 1
        return value

    def status(self):
        return {"backend": "fake", "ready": True}


class UnavailablePersonDetector:
    def detect(self, image):
        return []

    def status(self):
        return {"backend": "missing", "ready": False}


def person_box(x=100, y=100, score=0.9):
    return {"box": [x, y, 80, 160], "score": score, "classId": 0}


class TopOccupancyDetectorTest(unittest.TestCase):
    def make_detector(self, sequence, **overrides):
        config = {
            "roi": [0.0, 0.0, 1.0, 1.0],
            "history_size": 4,
            "present_min_frames": 3,
            "single_min_frames": 3,
            "multiple_min_frames": 3,
            "absent_min_seconds": 0.0,
            "track_iou_threshold": 0.3,
            "track_min_age_frames": 1,
            "track_max_missed_frames": 2,
        }
        config.update(overrides)
        return TopOccupancyDetector(
            person_detector=FakePersonDetector(sequence),
            config=config,
        )

    def test_continuous_absent_votes_none(self):
        detector = self.make_detector([[], [], [], []])
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = None
        for _ in range(4):
            result = detector.detect(image)

        self.assertEqual(result["occupancy"], "none")
        self.assertEqual(result["stableCount"], 0)

    def test_stable_single_votes_single(self):
        detector = self.make_detector([[person_box()]] * 4)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = None
        for _ in range(4):
            result = detector.detect(image)

        self.assertEqual(result["occupancy"], "single")
        self.assertEqual(result["stableCount"], 1)

    def test_responsive_mode_confirms_single_in_one_frame(self):
        detector = self.make_detector(
            [[person_box()]],
            history_size=2,
            present_min_frames=2,
            single_min_frames=1,
            multiple_min_frames=1,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        self.assertEqual(result["occupancy"], "single")

    def test_responsive_mode_confirms_multiple_in_one_frame(self):
        detector = self.make_detector(
            [[person_box(100), person_box(300)]],
            history_size=2,
            present_min_frames=2,
            single_min_frames=1,
            multiple_min_frames=1,
        )
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(result["occupancy"], "multiple")

    def test_responsive_mode_requires_two_empty_frames_to_leave(self):
        detector = self.make_detector(
            [[person_box()], [], []],
            history_size=2,
            present_min_frames=2,
            single_min_frames=1,
            multiple_min_frames=1,
            absent_min_seconds=0.0,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        self.assertEqual(detector.detect(image)["occupancy"], "single")
        self.assertEqual(detector.detect(image)["occupancy"], "single")
        self.assertEqual(detector.detect(image)["occupancy"], "none")

    def test_stable_multiple_votes_multiple(self):
        detector = self.make_detector(
            [[person_box(100), person_box(300)]] * 4
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = None
        for _ in range(4):
            result = detector.detect(image)

        self.assertEqual(result["occupancy"], "multiple")
        self.assertEqual(result["stableCount"], 2)

    def test_roi_filters_outside_person(self):
        detector = self.make_detector(
            [[person_box(20, 20)]] * 4,
            roi=[0.5, 0.5, 1.0, 1.0],
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = None
        for _ in range(4):
            result = detector.detect(image)

        self.assertEqual(result["occupancy"], "none")
        self.assertEqual(result["rawCount"], 0)

    def test_tracker_keeps_id_through_short_miss(self):
        detector = self.make_detector(
            [[person_box(100)], [], [person_box(104)], [person_box(108)]],
            track_max_missed_frames=2,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        first = detector.detect(image)
        track_id = first["tracks"][0]["id"]
        detector.detect(image)
        third = detector.detect(image)

        self.assertEqual(third["tracks"][0]["id"], track_id)

    def test_missing_person_model_reports_unknown_not_none(self):
        detector = TopOccupancyDetector(
            person_detector=UnavailablePersonDetector(),
            config={"enabled": True, "roi": [0.0, 0.0, 1.0, 1.0]},
        )
        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(result["occupancy"], "unknown")


if __name__ == "__main__":
    unittest.main()
