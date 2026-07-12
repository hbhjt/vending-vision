"""
身体测量模块

基于 MediaPipe 姿态关键点和分割掩码，估算人体的身体特征：
- 身高（粗略估计，基于关键点比例 + 配置中的缩放/偏移参数）
- 肩宽（基于身高比例换算）
- 体型（瘦/中等/胖，基于肩宽比或分割掩码宽度比）

支持多级后备策略：
1. 全身关键点测量（需要肩+髋+至少一个膝盖或脚踝）
2. 上半身关键点测量（需要肩+髋）
3. 分割掩码测量（需要启用 POSE_ENABLE_SEGMENTATION）
"""

import math
import mediapipe as mp
import numpy as np

from vision.config import settings

class BodyEstimator:
    """身体特征估算器。

    所有估算均为基于二维图像比例的粗略估计，非精确测量。
    结果受站姿、衣服宽松程度、摄像头角度等因素影响。
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def _get_visible_landmarks(self, results, min_visibility=0.5):
        """获取可见度高于阈值的人体关键点列表。"""
        if not results.pose_landmarks:
            return []

        landmarks = results.pose_landmarks.landmark

        visible_landmarks = []
        for lm in landmarks:
            if lm.visibility >= min_visibility:
                visible_landmarks.append(lm)

        return visible_landmarks

    def has_person(self, results, min_visible_points=8) -> bool:
        """判断画面中是否有人存在。

        只要可见关键点数量 >= min_visible_points 就认为检测到人体。
        """
        visible_landmarks = self._get_visible_landmarks(results, min_visibility=0.5)
        return len(visible_landmarks) >= min_visible_points

    def estimate_height_cm(self, results, default_height_cm=170.0):
        """基于姿态关键点的归一化身体高度估算身高（cm）。

        计算方式：身体在图像中的归一化高度 * HEIGHT_SCALE + HEIGHT_OFFSET。
        结果被限制在 130~210 cm 范围内。

        注意：
        - 这是粗略估算，未做实际摄像头标定。
        - 需要全身关键点（has_full_body_for_measurement 通过）才会估算，
          否则返回 None，由上层走 mask 后备路径。
        """
        # 不满足全身测量条件时不估算，避免用躯干高度错误代替全身高度
        if not self.has_full_body_for_measurement(results):
            return None

        visible_landmarks = self._get_visible_landmarks(results)

        if not visible_landmarks:
            return None

        ys = [lm.y for lm in visible_landmarks]

        top_y = min(ys)     # 最高可见点（通常是头部）
        bottom_y = max(ys)  # 最低可见点（通常是脚部）

        body_ratio = bottom_y - top_y    # 身体在图像中的归一化高度

        if body_ratio <= 0:
            return None

        # 线性映射：body_ratio * scale + offset -> 身高 cm
        height_cm = body_ratio * settings.HEIGHT_SCALE + settings.HEIGHT_OFFSET

        if height_cm < 130:
            height_cm = 130.0

        if height_cm > 210:
            height_cm = 210.0

        return round(height_cm, 1)

    def estimate_body_type(self, results):
        """基于全身关键点的肩宽/身高比估算体型。

        使用左右肩距离 / 身体总高度比值：
        - thin:  比值 < BODY_TYPE_THIN_THRESHOLD
        - medium: BODY_TYPE_THIN_THRESHOLD <= 比值 < BODY_TYPE_FAT_THRESHOLD
        - fat:    比值 >= BODY_TYPE_FAT_THRESHOLD
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

    def has_upper_body_for_body_type(self, results) -> bool:
        """判断是否满足上半身体型估算条件。

        要求左右肩和左右髋四个关键点可见度均 >= 0.45。
        不再要求膝盖或脚踝，更适合售货机中置摄像头的画面。
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

        return all(landmarks[point].visibility >= 0.45 for point in required_points)

    def estimate_upper_body_type(self, results):
        """基于上半身比例估算体型（适用于售货机中部摄像头画面）。

        使用肩宽 / 躯干高度比值：
        躯干高度 = 髋部Y坐标 - 肩部Y坐标
        肩宽 = 左右肩之间的欧几里得距离

        注意：结果受站姿、衣服宽松程度和摄像头角度影响。
        """
        if not self.has_upper_body_for_body_type(results):
            return "unknown"

        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        shoulder_width = math.sqrt(
            (left_shoulder.x - right_shoulder.x) ** 2
            + (left_shoulder.y - right_shoulder.y) ** 2
        )
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        torso_height = hip_y - shoulder_y

        if torso_height <= 0:
            return "unknown"

        ratio = shoulder_width / torso_height

        if ratio < settings.UPPER_BODY_TYPE_THIN_THRESHOLD:
            return "thin"
        elif ratio < settings.UPPER_BODY_TYPE_FAT_THRESHOLD:
            return "medium"
        else:
            return "fat"

    def _segmentation_bbox(self, results):
        """从分割掩码提取人体边界框。

        仅在 BODY_MASK_ENABLED 为 True 且分割掩码可用时返回结果。
        要求掩码区域占比 >= BODY_MASK_MIN_AREA_RATIO。
        """
        if not settings.BODY_MASK_ENABLED:
            return None

        mask = getattr(results, "segmentation_mask", None)

        if mask is None:
            return None

        # 二值化掩码
        person_mask = mask >= settings.BODY_MASK_THRESHOLD
        area_ratio = float(person_mask.mean())

        if area_ratio < settings.BODY_MASK_MIN_AREA_RATIO:
            return None

        ys, xs = np.where(person_mask)

        if len(xs) == 0 or len(ys) == 0:
            return None

        height, width = person_mask.shape[:2]
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())

        box_width = max(x_max - x_min + 1, 1)
        box_height = max(y_max - y_min + 1, 1)

        return {
            "mask": person_mask,
            "width": width,
            "height": height,
            "xMin": x_min,
            "xMax": x_max,
            "yMin": y_min,
            "yMax": y_max,
            "boxWidth": box_width,
            "boxHeight": box_height,
            "areaRatio": area_ratio,
            "heightRatio": box_height / float(height),
            "widthRatio": box_width / float(width),
        }

    def estimate_height_cm_from_mask(self, results):
        """使用人体分割掩码的可见高度做粗略的身高后备估算。

        基于掩码在图像中的高度比 * HEIGHT_SCALE + HEIGHT_OFFSET。
        结果限制在 130~210 cm。
        """
        bbox = self._segmentation_bbox(results)

        if bbox is None:
            return None

        height_cm = (
            bbox["heightRatio"] * settings.HEIGHT_SCALE
            + settings.HEIGHT_OFFSET
        )

        if height_cm < 130:
            height_cm = 130.0

        if height_cm > 210:
            height_cm = 210.0

        return round(height_cm, 1)

    def estimate_body_type_from_mask(self, results):
        """使用人体分割掩码的上半身轮廓粗略估算体型。

        在掩码的 22%~58% 高度范围内取水平切片，
        用切片宽度 / 身体总高度比值判断体型。
        """
        bbox = self._segmentation_bbox(results)

        if bbox is None:
            return "unknown"

        person_mask = bbox["mask"]
        y_min = bbox["yMin"]
        y_max = bbox["yMax"]
        body_height = bbox["boxHeight"]

        # 取上半身的水平切片（22%~58%高度范围，覆盖肩膀到腰部）
        band_top = int(y_min + body_height * 0.22)
        band_bottom = int(y_min + body_height * 0.58)
        band_top = max(y_min, min(band_top, y_max))
        band_bottom = max(band_top + 1, min(band_bottom, y_max + 1))
        band = person_mask[band_top:band_bottom, :]

        ys, xs = np.where(band)

        if len(xs) == 0:
            return "unknown"

        upper_width = int(xs.max() - xs.min() + 1)
        ratio = upper_width / float(body_height)

        if ratio < settings.BODY_MASK_TYPE_THIN_THRESHOLD:
            return "thin"
        elif ratio < settings.BODY_MASK_TYPE_FAT_THRESHOLD:
            return "medium"
        else:
            return "fat"

    def estimate_shoulder_width_cm(self, results, height_cm=None):
        """基于姿态关键点和身高估算肩宽（cm）。

        计算方式：
        1. 取左右肩关键点的归一化距离
        2. 用身体归一化高度作为参考
        3. 肩宽 = 身高 * (肩宽归一化距离 / 身体归一化高度)

        结果限制在 25~65 cm 范围内。
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

        # 合理范围限制，防止异常值
        if shoulder_width_cm < 25:
            shoulder_width_cm = 25.0

        if shoulder_width_cm > 65:
            shoulder_width_cm = 65.0

        return round(shoulder_width_cm, 1)

    def has_full_body_for_measurement(self, results) -> bool:
        """判断是否满足全身测量条件。

        要求：
        - 左右肩和左右髋四个关键点可见度均 >= 0.5
        - 至少有一个膝盖或脚踝可见度 >= 0.5
        """
        if not results.pose_landmarks:
            return False

        landmarks = results.pose_landmarks.landmark

        # 上半身四个必须点
        required_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]

        for point in required_points:
            if landmarks[point].visibility < 0.5:
                return False

        # 下半身至少一个点
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
