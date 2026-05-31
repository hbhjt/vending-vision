import cv2
import numpy as np
import mediapipe as mp


class UpperColorEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def estimate(self, image, pose_results) -> str:
        """
        根据人体关键点截取上衣区域，并估算主色。
        返回:
            dark / light / red / blue / green / yellow / white / black / unknown
        """

        if image is None or not pose_results.pose_landmarks:
            return "unknown"

        h, w = image.shape[:2]
        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        needed = [left_shoulder, right_shoulder, left_hip, right_hip]

        if any(lm.visibility < 0.4 for lm in needed):
            return "unknown"

        x_min = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        x_max = int(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * w)
        y_min = int(min(left_shoulder.y, right_shoulder.y) * h)
        y_max = int(max(left_hip.y, right_hip.y) * h)

        # 稍微扩大一点区域，避免只截到身体中心太窄
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
        """
        用 HSV + 亮度规则粗略判断颜色。
        """

        # 缩小图片，减少计算量
        roi = cv2.resize(roi, (64, 64))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # 去掉低饱和度背景影响
        mean_s = float(np.mean(s_channel))
        mean_v = float(np.mean(v_channel))
        mean_h = float(np.mean(h_channel))

        # 先判断黑白灰和明暗
        if mean_v < 60:
            return "black"

        if mean_v > 200 and mean_s < 45:
            return "white"

        if mean_s < 40:
            if mean_v < 120:
                return "dark"
            else:
                return "light"

        # HSV 中 OpenCV H 范围是 0-179
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