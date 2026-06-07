import cv2
import mediapipe as mp

from vision.config import settings


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=settings.POSE_ENABLE_SEGMENTATION,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        """
        输入 OpenCV BGR 图片，输出 MediaPipe Pose 结果
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def draw_pose(self, image, results):
        """
        在图片上绘制人体关键点和骨架
        """
        output = image.copy()

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return output

    def has_pose(self, results) -> bool:
        """
        判断是否检测到人体姿态
        """
        return results.pose_landmarks is not None
