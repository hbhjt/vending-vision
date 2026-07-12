"""
姿态估计模块

基于 MediaPipe Pose 的姿态检测器。
使用静态图像模式（static_image_mode=True），对每帧独立推理。
可选启用人体分割掩码（segmentation mask），用于体型/身高后备估算。
"""

import cv2
import mediapipe as mp

from vision.config import settings


class PoseEstimator:
    """MediaPipe 姿态估计器。

    特性：
    - 静态图像模式：每帧独立推理，不依赖帧间追踪
    - 模型复杂度 1：平衡精度和速度
    - 可选分割掩码：由 POSE_ENABLE_SEGMENTATION 配置控制
    - 检测置信度阈值：0.5
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,                              # 静态图像模式
            model_complexity=1,                                   # 模型复杂度（0=轻量, 1=标准, 2=高精度）
            enable_segmentation=settings.POSE_ENABLE_SEGMENTATION, # 是否启用人体分割
            min_detection_confidence=0.5                          # 最小检测置信度
        )

    def detect(self, image):
        """对 BGR 图像执行姿态检测。

        Args:
            image: OpenCV BGR 格式图像

        Returns:
            MediaPipe Pose 检测结果（包含 pose_landmarks 和可选的 segmentation_mask）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def draw_pose(self, image, results):
        """在图像上绘制人体关键点和骨架连线（用于调试可视化）。

        Args:
            image: 原始 BGR 图像
            results: MediaPipe Pose 检测结果

        Returns:
            绘制了骨架的图像副本
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
        """判断是否检测到人体姿态。"""
        return results.pose_landmarks is not None
