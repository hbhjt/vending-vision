import os

import cv2
import numpy as np

from vision.config import settings
from vision.logger import logger


class PersonDetector:
    def __init__(self):
        self.enabled = bool(settings.PROXIMITY_PERSON_ENABLED)
        self.model_path = settings.PERSON_DETECTOR_MODEL
        self.net = None
        self.backend = "disabled"
        self.last_error = None

        if not self.enabled:
            return

        if not os.path.exists(self.model_path):
            self.backend = "missing"
            self.last_error = f"person detector model missing: {self.model_path}"
            logger.warning(self.last_error)
            return

        try:
            self.net = cv2.dnn.readNet(self.model_path)
            self.backend = "opencv_dnn"
            logger.info(f"Person detector initialized: {self.model_path}")
        except Exception as e:
            self.backend = "error"
            self.last_error = str(e)
            logger.exception(f"Failed to load person detector: {e}")

    def status(self):
        return {
            "enabled": self.enabled,
            "ready": self.net is not None,
            "backend": self.backend,
            "modelPath": self.model_path,
            "inputWidth": settings.PERSON_DETECTOR_INPUT_WIDTH,
            "inputHeight": settings.PERSON_DETECTOR_INPUT_HEIGHT,
            "scoreThreshold": settings.PERSON_DETECTOR_SCORE_THRESHOLD,
            "nmsThreshold": settings.PERSON_DETECTOR_NMS_THRESHOLD,
            "lastError": self.last_error,
        }

    def detect(self, image):
        if self.net is None or image is None:
            return []

        input_width = settings.PERSON_DETECTOR_INPUT_WIDTH
        input_height = settings.PERSON_DETECTOR_INPUT_HEIGHT

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(input_width, input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)

        try:
            outputs = self.net.forward()
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Person detector inference failed: {e}")
            raise

        detections = self._parse_yolo_output(outputs, image.shape[:2])

        if not detections:
            return []

        boxes = [item["box"] for item in detections]
        scores = [item["score"] for item in detections]
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            settings.PERSON_DETECTOR_SCORE_THRESHOLD,
            settings.PERSON_DETECTOR_NMS_THRESHOLD,
        )

        if len(indexes) == 0:
            return []

        flat_indexes = np.array(indexes).reshape(-1).tolist()
        return [detections[index] for index in flat_indexes]

    def _parse_yolo_output(self, outputs, image_shape):
        h, w = image_shape
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                return []
            outputs = outputs[0]

        output = np.array(outputs)

        if output.ndim == 3:
            output = output[0]

        if output.ndim != 2:
            return []

        # YOLOv8 exports often use (classes + 4, anchors); YOLOv5-style exports
        # often use (anchors, classes + 5). Normalize to one row per anchor.
        if output.shape[0] < output.shape[1] and output.shape[0] in {5, 6, 84, 85}:
            output = output.T

        detections = []

        for row in output:
            parsed = self._parse_yolo_row(row, w, h)
            if parsed is not None:
                detections.append(parsed)

        return detections

    def _parse_yolo_row(self, row, image_width, image_height):
        if len(row) < 6:
            return None

        values = row.astype(float)
        class_id = settings.PERSON_DETECTOR_PERSON_CLASS_ID

        if len(values) >= 85:
            objectness = values[4]
            class_scores = values[5:]
            if class_id >= len(class_scores):
                return None
            score = objectness * class_scores[class_id]
        else:
            class_scores = values[4:]
            if class_id >= len(class_scores):
                return None
            score = class_scores[class_id]

        if score < settings.PERSON_DETECTOR_SCORE_THRESHOLD:
            return None

        cx, cy, box_w, box_h = values[:4]

        # Some exports return normalized coordinates, others return input pixels.
        if max(cx, cy, box_w, box_h) <= 2.0:
            cx *= image_width
            box_w *= image_width
            cy *= image_height
            box_h *= image_height
        else:
            cx *= image_width / float(settings.PERSON_DETECTOR_INPUT_WIDTH)
            box_w *= image_width / float(settings.PERSON_DETECTOR_INPUT_WIDTH)
            cy *= image_height / float(settings.PERSON_DETECTOR_INPUT_HEIGHT)
            box_h *= image_height / float(settings.PERSON_DETECTOR_INPUT_HEIGHT)

        x = int(max(0, cx - box_w / 2.0))
        y = int(max(0, cy - box_h / 2.0))
        box_w = int(min(image_width - x, max(0, box_w)))
        box_h = int(min(image_height - y, max(0, box_h)))

        if box_w <= 0 or box_h <= 0:
            return None

        return {
            "box": [x, y, box_w, box_h],
            "score": round(float(score), 4),
            "classId": class_id,
        }
