import math
import mediapipe as mp
from vision.config import settings

class BodyEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def _get_visible_landmarks(self, results, min_visibility=0.5):
        """
        获取可见度较高的人体关键点。
        """
        if not results.pose_landmarks:
            return []

        landmarks = results.pose_landmarks.landmark

        visible_landmarks = []
        for lm in landmarks:
            if lm.visibility >= min_visibility:
                visible_landmarks.append(lm)

        return visible_landmarks

    def has_person(self, results, min_visible_points=8) -> bool:
        """
        根据 MediaPipe Pose 判断画面中是否有人。

        只要可见关键点数量足够，就认为检测到人体。
        """
        visible_landmarks = self._get_visible_landmarks(results, min_visibility=0.5)
        return len(visible_landmarks) >= min_visible_points

    def estimate_height_cm(self, results, default_height_cm=170.0):
        """
        简化版身高估算。

        注意：
        这里只是根据人体关键点在图片中的比例做粗略估计。
        如果没有真实摄像头标定，不能当作准确身高。
        """
        visible_landmarks = self._get_visible_landmarks(results)

        if not visible_landmarks:
            return None

        ys = [lm.y for lm in visible_landmarks]

        top_y = min(ys)
        bottom_y = max(ys)

        body_ratio = bottom_y - top_y

        if body_ratio <= 0:
            return None

        height_cm = body_ratio * settings.HEIGHT_SCALE + settings.HEIGHT_OFFSET

        if height_cm < 130:
            height_cm = 130.0

        if height_cm > 210:
            height_cm = 210.0

        return round(height_cm, 1)

    def estimate_body_type(self, results):
        """
        根据肩宽 / 身体高度粗略估算体型。
        """
        if not results.pose_landmarks:
            return "unknown"

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        visible_landmarks = self._get_visible_landmarks(results)

        if not visible_landmarks:
            return "unknown"

        ys = [lm.y for lm in visible_landmarks]
        body_height = max(ys) - min(ys)

        if body_height <= 0:
            return "unknown"

        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        ratio = shoulder_width / body_height

        if ratio < settings.BODY_TYPE_THIN_THRESHOLD:
            return "thin"
        elif ratio < settings.BODY_TYPE_FAT_THRESHOLD:
            return "medium"
        else:
            return "fat"

    def estimate_shoulder_width_cm(self, results, height_cm=None):
        """
        简化版肩宽估算。

        思路：
        1. 使用左右肩关键点的归一化距离
        2. 使用人体整体归一化高度作为参考
        3. 根据 身高 * 肩宽比例 粗略换算肩宽

        注意：
        这是原型估算，不是精确测量。
        """
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
            return None

        visible_landmarks = self._get_visible_landmarks(results)

        if not visible_landmarks:
            return None

        ys = [lm.y for lm in visible_landmarks]
        body_height_norm = max(ys) - min(ys)

        if body_height_norm <= 0:
            return None

        shoulder_width_norm = math.sqrt(
            (left_shoulder.x - right_shoulder.x) ** 2
            + (left_shoulder.y - right_shoulder.y) ** 2
        )

        ratio = shoulder_width_norm / body_height_norm

        if height_cm is None:
            height_cm = self.estimate_height_cm(results)

        if height_cm is None:
            return None

        shoulder_width_cm = height_cm * ratio

        # 做一个合理范围限制，防止异常值
        if shoulder_width_cm < 25:
            shoulder_width_cm = 25.0

        if shoulder_width_cm > 65:
            shoulder_width_cm = 65.0

        return round(shoulder_width_cm, 1)

    def has_full_body_for_measurement(self, results) -> bool:
        """
        判断是否适合估算身高、肩宽、体型。

        需要至少检测到：
        - 左右肩
        - 左右髋
        - 至少一个膝盖或脚踝
        """
        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks.landmark

        required_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]

        for point in required_points:
            if landmarks[point].visibility < 0.5:
                return False

        lower_body_points = [
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

        visible_lower_points = 0

        for point in lower_body_points:
            if landmarks[point].visibility >= 0.5:
                visible_lower_points += 1

        return visible_lower_points >= 1