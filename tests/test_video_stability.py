import tempfile
import unittest
from pathlib import Path

from scripts.analyze_video_stability import analyze, read_csv, write_csv


class VideoStabilityTest(unittest.TestCase):
    def make_dataset(self, root):
        top_rows = [
            {
                "timestampMs": timestamp,
                "autoOccupancy": occupancy,
                "present": present,
                "rawCount": raw_count,
                "largestPersonRatio": ratio,
                "imagePath": str(root / f"top-{timestamp}.jpg"),
            }
            for timestamp, occupancy, present, raw_count, ratio in [
                (0, "none", False, 0, 0.0),
                (333, "single", True, 1, 0.1),
                (666, "single", True, 1, 0.1),
            ]
        ]
        front_rows = [
            {
                "timestampMs": timestamp,
                "personDetected": evidence,
                "faceDetected": False,
                "valid": valid,
                "confidence": confidence,
                "bodyType": body_type,
                "upperColor": "light" if valid else "unknown",
                "ageRange": "unknown",
                "gender": "unknown",
                "imagePath": str(root / f"front-{timestamp}.jpg"),
            }
            for timestamp, evidence, valid, confidence, body_type in [
                (0, False, False, 0.3, "unknown"),
                (333, True, True, 0.6, "regular"),
                (666, True, True, 0.6, "regular"),
            ]
        ]
        write_csv(root / "top_occupancy_auto_labels.csv", top_rows)
        write_csv(root / "front_profile_auto_labels.csv", front_rows)

    def test_analyze_builds_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)

            summary = analyze(root)

            self.assertEqual(summary["metrics"]["frontEvidenceCount"], 2)
            self.assertEqual(summary["metrics"]["endToEndCoveragePct"], 100.0)
            self.assertTrue((root / "stability_report.html").exists())
            self.assertTrue((root / "review_labels.csv").exists())

    def test_manual_review_values_survive_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            analyze(root)
            review_path = root / "review_labels.csv"
            rows = read_csv(review_path)
            rows[1]["manualPersonPresent"] = "True"
            rows[1]["manualOccupancy"] = "single"
            rows[1]["manualFrontUsable"] = "True"
            rows[1]["notes"] = "checked"
            write_csv(review_path, rows)

            summary = analyze(root)
            regenerated = read_csv(review_path)

            self.assertEqual(summary["manualReview"]["reviewedRowCount"], 1)
            self.assertEqual(summary["manualReview"]["endToEndRecallPct"], 100.0)
            self.assertEqual(regenerated[1]["notes"], "checked")


if __name__ == "__main__":
    unittest.main()
