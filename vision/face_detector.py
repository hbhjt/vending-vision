import os

import cv2
import mediapipe as mp

from vision.config import settings
from vision.logger import logger


class FaceDetector:
    def __init__(self):
        """
        人脸检测模块。

        优先使用 YuNet ONNX。
        如果 YuNet 模型文件不存在，则自动降级为 OpenCV Haar。
        """

        self.backend = "haar"
        self.detector = None
        self.face_cascade = None

        if os.path.exists(settings.FACE_DETECTOR_MODEL) and hasattr(cv2, "FaceDetectorYN_create"):
            self._init_yunet()
        else:
            self._init_haar()

    def _init_yunet(self):
        try:
            self.detector = cv2.FaceDetectorYN_create(
                settings.FACE_DETECTOR_MODEL,
                "",
                (320, 320),
                settings.FACE_SCORE_THRESHOLD,
                settings.FACE_NMS_THRESHOLD,
                settings.FACE_TOP_K
            )

            self.backend = "yunet"
            logger.info("Face detector initialized: YuNet")

        except Exception as e:
            logger.exception(f"YuNet init failed, fallback to Haar: {e}")
            self._init_haar()

    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Haar 人脸检测模型加载失败")

        self.backend = "haar"
        logger.info("Face detector initialized: Haar")

    def detect(self, image):
        """
        输入 OpenCV BGR 图片。
        返回人脸框列表: [(x, y, w, h), ...]
        """

        if image is None:
            return []

        if self.backend == "yunet":
            return self._detect_yunet(image)

        return self._detect_haar(image)

    def _detect_yunet(self, image):
        h, w = image.shape[:2]

        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        boxes = []

        for face in faces:
            x, y, box_w, box_h = face[:4]

            x = int(max(0, x))
            y = int(max(0, y))
            box_w = int(max(0, box_w))
            box_h = int(max(0, box_h))

            if box_w <= 0 or box_h <= 0:
                continue

            # 防止越界
            if x + box_w > w:
                box_w = w - x

            if y + box_h > h:
                box_h = h - y

            boxes.append((x, y, box_w, box_h))

        return boxes

    def _detect_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return list(faces)

    def draw_faces(self, image, faces):
        """
        在图片上画出人脸框。
        """

        output = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return output

    def crop_largest_face(self, image, faces):
        """
        裁剪最大的人脸。
        后面年龄性别识别会用这个裁剪结果。
        """

        if image is None or len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_face

        return image[y:y + h, x:x + w]

    def crop_face(self, image, face):
        if image is None or face is None:
            return None

        x, y, w, h = face
        return image[y:y + h, x:x + w]

    def _box_center(self, box):
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def _head_center_from_pose(self, image, pose_results):
        if image is None or not pose_results or not pose_results.pose_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = pose_results.pose_landmarks.landmark
        pose_landmark = mp.solutions.pose.PoseLandmark
        indexes = [
            pose_landmark.NOSE,
            pose_landmark.LEFT_EYE,
            pose_landmark.RIGHT_EYE,
            pose_landmark.LEFT_EAR,
            pose_landmark.RIGHT_EAR,
        ]

        points = []

        for index in indexes:
            lm = landmarks[index]
            if lm.visibility >= 0.4:
                points.append((lm.x * w, lm.y * h))

        if not points:
            return None

        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return x, y

    def select_primary_face(self, image, faces, pose_results=None):
        if image is None or not faces:
            return None, {
                "method": "none",
                "faceCount": 0,
                "reason": "no face",
            }

        h, w = image.shape[:2]
        image_diag = (w ** 2 + h ** 2) ** 0.5
        head_center = self._head_center_from_pose(image, pose_results)

        if head_center is not None:
            candidates = []

            for face in faces:
                face_center = self._box_center(face)
                distance = (
                    (face_center[0] - head_center[0]) ** 2
                    + (face_center[1] - head_center[1]) ** 2
                ) ** 0.5
                candidates.append((distance, face))

            distance, face = min(candidates, key=lambda item: item[0])
            distance_ratio = distance / image_diag if image_diag > 0 else 1.0

            if distance_ratio <= settings.PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO:
                return face, {
                    "method": "pose_head_nearest_face",
                    "faceCount": len(faces),
                    "headDistanceRatio": round(distance_ratio, 4),
                    "matched": True,
                }

        image_center = (w / 2.0, h / 2.0)

        def fallback_score(face):
            x, y, box_w, box_h = face
            area_ratio = (box_w * box_h) / float(w * h)
            center = self._box_center(face)
            center_distance = (
                (center[0] - image_center[0]) ** 2
                + (center[1] - image_center[1]) ** 2
            ) ** 0.5
            center_distance_ratio = center_distance / image_diag if image_diag > 0 else 1.0
            return area_ratio - center_distance_ratio * 0.08

        face = max(faces, key=fallback_score)
        return face, {
            "method": "largest_center_weighted_face",
            "faceCount": len(faces),
            "matched": False,
            "reason": "pose head unavailable or too far from face",
        }
