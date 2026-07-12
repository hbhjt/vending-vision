"""
视觉推理管线模块

编排完整的画像采集推理链，将各子模块串联为一个统一的推理流程：

姿态检测 -> 人脸检测 -> 主脸选择 -> 身体测量（身高/肩宽/体型）-> 上衣颜色 -> 年龄性别

输出统一的 VisionProfile 数据模型。
"""

from vision.schema import VisionProfile
from vision.pose_estimator import PoseEstimator
from vision.body_estimator import BodyEstimator
from vision.face_detector import FaceDetector
from vision.age_gender_estimator import AgeGenderEstimator
from vision.color_estimator import UpperColorEstimator


class VisionPipeline:
    """视觉推理管线。

    封装了所有视觉模型的推理协调：
    1. 姿态估计 (MediaPipe Pose)
    2. 人脸检测 (SCRFD/YuNet/Haar)
    3. 主脸选择与裁剪
    4. 身体测量：身高、肩宽、体型（基于姿态关键点 + 分割掩码后备）
    5. 上衣颜色分类（基于 HSV）
    6. 年龄性别识别（OpenVINO/Caffe/Mock）

    所有模型在初始化时加载一次，避免每次请求重复加载。
    """

    def __init__(self):
        # 所有子模型只创建一次，避免重复加载
        self.pose_estimator = PoseEstimator()
        self.body_estimator = BodyEstimator()
        self.face_detector = FaceDetector()
        self.age_gender_estimator = AgeGenderEstimator()
        self.upper_color_estimator = UpperColorEstimator()

    def infer(self, image) -> VisionProfile:
        """对单帧图像执行完整推理，返回 VisionProfile。

        推理流程：
        1. 姿态估计 -> 判断是否有人
        2. 人脸检测 -> 选择主脸 -> 裁剪人脸区域
        3. 身体测量（优先使用全身关键点，其次上半身，最后分割掩码）
        4. 上衣颜色分类
        5. 年龄性别识别

        所有测量结果经过异常值过滤（如身高 140~200cm，肩宽 32~55cm）。

        Args:
            image: BGR 格式的 OpenCV 图像

        Returns:
            VisionProfile 数据模型实例
        """
        # 1. 姿态估计
        pose_results = self.pose_estimator.detect(image)

        # 2. 人脸检测与主脸选择
        faces = self.face_detector.detect(image)
        primary_face, _ = self.face_detector.select_primary_face(
            image,
            faces,
            pose_results=pose_results
        )
        face_image = self.face_detector.crop_face(image, primary_face)

        # 3. 判断是否有人存在
        has_pose_person = self.body_estimator.has_person(pose_results)
        has_face_person = len(faces) > 0

        has_person = has_pose_person or has_face_person

        if not has_person:
            # 无人检测到时返回空画像
            return VisionProfile(
                age=None,
                gender="unknown",
                height_cm=None,
                shoulder_width_cm=None,
                body_type="unknown",
                upper_color="unknown",
                presence=False
            )

        # 4. 身体测量：身高、肩宽、体型
        # 判断可用的测量方式（全身 > 上半身 > 分割掩码）
        can_measure_body = self.body_estimator.has_full_body_for_measurement(pose_results)
        can_measure_upper_body = self.body_estimator.has_upper_body_for_body_type(
            pose_results
        )

        height_cm = None
        shoulder_width_cm = None
        body_type = "unknown"

        # 方式1：基于全身关键点测量
        if can_measure_body:
            height_cm = self.body_estimator.estimate_height_cm(pose_results)
            shoulder_width_cm = self.body_estimator.estimate_shoulder_width_cm(
                pose_results,
                height_cm=height_cm
            )
            body_type = self.body_estimator.estimate_body_type(pose_results)

            # 明显异常值过滤
            if height_cm is not None and not (140 <= height_cm <= 200):
                height_cm = None

            if shoulder_width_cm is not None and not (32 <= shoulder_width_cm <= 55):
                shoulder_width_cm = None

        # 方式2：上半身体型估算（当全身测量不可用或体型仍未知时）
        if body_type == "unknown" and can_measure_upper_body:
            body_type = self.body_estimator.estimate_upper_body_type(pose_results)

        # 方式3：基于分割掩码的身高后备估算
        if height_cm is None:
            height_cm = self.body_estimator.estimate_height_cm_from_mask(pose_results)

            if height_cm is not None and not (140 <= height_cm <= 200):
                height_cm = None

        # 方式4：基于分割掩码的体型后备估算
        if body_type == "unknown":
            body_type = self.body_estimator.estimate_body_type_from_mask(pose_results)

        # 5. 上衣颜色分类
        upper_color = self.upper_color_estimator.estimate(image, pose_results)

        # 6. 年龄性别识别（模型缺失或人脸不可用时返回 unknown）
        age, gender = self.age_gender_estimator.predict(face_image)

        profile = VisionProfile(
            age=age,
            gender=gender,
            height_cm=height_cm,
            shoulder_width_cm=shoulder_width_cm,
            body_type=body_type,
            upper_color=upper_color,
            presence=True
        )

        return profile
