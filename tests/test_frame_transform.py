import unittest

import numpy as np

from vision.frame_transform import (
    crop_normalized_roi,
    normalize_rotation,
    rotate_frame,
)


class FrameTransformTest(unittest.TestCase):
    def test_normalize_rotation_accepts_named_values(self):
        self.assertEqual(normalize_rotation("clockwise_90"), 90)
        self.assertEqual(normalize_rotation("counterclockwise_90"), 270)
        self.assertEqual(normalize_rotation(-90), 270)
        self.assertEqual(normalize_rotation(None), 0)

    def test_rotate_frame_counterclockwise(self):
        image = np.array(
            [
                [[1], [2], [3]],
                [[4], [5], [6]],
            ],
            dtype=np.uint8,
        )

        rotated = rotate_frame(image, 270)

        self.assertEqual(rotated.shape[:2], (3, 2))
        self.assertEqual(rotated[0, 0], 3)
        self.assertEqual(rotated[2, 1], 4)

    def test_crop_normalized_roi(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped, roi = crop_normalized_roi(
            image,
            {
                "enabled": True,
                "x": 0.25,
                "y": 0.2,
                "width": 0.5,
                "height": 0.4,
            },
        )

        self.assertEqual(cropped.shape[:2], (40, 100))
        self.assertEqual(roi["pixelX"], 50)
        self.assertEqual(roi["pixelY"], 20)


if __name__ == "__main__":
    unittest.main()
