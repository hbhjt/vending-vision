"""
上衣颜色分类模块

基于 HSV 色彩空间的上衣颜色估计器。
利用 MediaPipe 姿态关键点定位上衣区域（肩部到髋部），
然后对该区域进行 HSV 颜色统计分类。
"""

import cv2
import numpy as np
import mediapipe as mp


class UpperColorEstimator:
    """上衣颜色分类器。

    工作原理：
    1. 利用姿态关键点（左右肩、左右髋）定位上衣 ROI
    2. 将 ROI 转换到 HSV 色彩空间
    3. 根据色相(H)、饱和度(S)、明度(V) 的均值进行分类

    支持的颜色类别：
    dark, light, red, blue, green, yellow, white, black, unknown
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def estimate(self, image, pose_results) -> str:
        """根据姿态关键点截取上衣区域并估算主色。

        要求左右肩和左右髋四个关键点可见度均 >= 0.4。
        ROI 会适当向外扩展 25%（水平）和 10%（垂直），避免截取过窄。

        Args:
            image: BGR 格式的 OpenCV 图像
            pose_results: MediaPipe Pose 检测结果

        Returns:
            颜色字符串: dark / light / red / blue / green / yellow / white / black / unknown
        """
        if image is None or not pose_results.pose_landmarks:
            return "unknown"

        h, w = image.shape[:2]
        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        if (
            left_shoulder.visibility < 0.4
            or right_shoulder.visibility < 0.4
        ):
            return "unknown"

        # 计算上衣区域边界
        if left_hip.visibility >= 0.4 and right_hip.visibility >= 0.4:
            x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
            x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
            y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
            y_max = int(max(left_hip.y, right_hip.y) * h)
        else:
            # 近距离竖屏特写时下身（髋关节）落在画幅之外：退化为
            # 双肩到画面底部的可见躯干区域，仍能采样上衣主色。
            x_min = int(min(left_shoulder.x, right_shoulder.x) * w)
            x_max = int(max(left_shoulder.x, right_shoulder.x) * w)
            y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
            y_max = h

        # 稍微扩大区域，避免只截到身体中心太窄
        pad_x = int((x_max - x_min) * 0.25)
        pad_y = int((y_max - y_min) * 0.10)

        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)

        if x_max <= x_min or y_max <= y_min:
            return "unknown"

        upper_roi = image[y_min:y_max, x_min:x_max]

        if upper_roi.size == 0:
            return "unknown"

        return self._classify_color(upper_roi)

    def _classify_color(self, roi) -> str:
        """对上衣 ROI 进行 HSV 颜色分类。

        分类规则：
        1. 先判断明暗/黑白灰（基于 V 和 S 通道）
        2. 再根据色相 H 判断具体颜色

        OpenCV HSV 范围：H=[0,179], S=[0,255], V=[0,255]
        """
        # 缩小图片以减少计算量
        roi = cv2.resize(roi, (64, 64))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_channel = hsv[:, :, 0]  # 色相
        s_channel = hsv[:, :, 1]  # 饱和度
        v_channel = hsv[:, :, 2]  # 明度

        mean_s = float(np.mean(s_channel))
        mean_v = float(np.mean(v_channel))
        mean_h = float(np.mean(h_channel))

        # 先判断黑白灰和明暗（低饱和度或无彩色）
        if mean_v < 60:
            return "black"

        if mean_v > 200 and mean_s < 45:
            return "white"

        if mean_s < 40:
            if mean_v < 120:
                return "dark"
            else:
                return "light"

        # 基于色相判断具体颜色
        # H 范围 (OpenCV 0-179):
        #   红: 0-10 或 160-179
        #   黄: 10-30
        #   绿: 30-85
        #   蓝: 85-130
        #   红（下半段）: 130-160
        if mean_h < 10 or mean_h >= 160:
            return "red"
        elif 10 <= mean_h < 30:
            return "yellow"
        elif 30 <= mean_h < 85:
            return "green"
        elif 85 <= mean_h < 130:
            return "blue"
        elif 130 <= mean_h < 160:
            return "red"

        return "unknown"
