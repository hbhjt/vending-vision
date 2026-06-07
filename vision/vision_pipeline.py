from vision.schema import VisionProfile
from vision.pose_estimator import PoseEstimator
from vision.body_estimator import BodyEstimator
from vision.face_detector import FaceDetector
from vision.age_gender_estimator import AgeGenderEstimator
from vision.color_estimator import UpperColorEstimator


class VisionPipeline:
    def __init__(self):
        """
        初始化机器视觉模块。

        这些对象只创建一次，避免每次请求都重新加载。
        """
        self.pose_estimator = PoseEstimator()
        self.body_estimator = BodyEstimator()
        self.face_detector = FaceDetector()
        self.age_gender_estimator = AgeGenderEstimator()
        self.upper_color_estimator = UpperColorEstimator()

    def infer(self, image) -> VisionProfile:
        """
        输入图片，输出 VisionProfile。
        """

        # 1. 姿态估计
        pose_results = self.pose_estimator.detect(image)

        # 2. 人脸检测
        faces = self.face_detector.detect(image)
        primary_face, _ = self.face_detector.select_primary_face(
            image,
            faces,
            pose_results=pose_results
        )
        face_image = self.face_detector.crop_face(image, primary_face)

        # 3. 判断是否有人
        has_pose_person = self.body_estimator.has_person(pose_results)
        has_face_person = len(faces) > 0

        has_person = has_pose_person or has_face_person

        if not has_person:
            return VisionProfile(
                age=None,
                gender="unknown",
                height_cm=None,
                shoulder_width_cm=None,
                body_type="unknown",
                upper_color="unknown",
                presence=False
            )

        # 4. 身高、肩宽、体型、上衣颜色
        can_measure_body = self.body_estimator.has_full_body_for_measurement(pose_results)
        can_measure_upper_body = self.body_estimator.has_upper_body_for_body_type(
            pose_results
        )

        height_cm = None
        shoulder_width_cm = None
        body_type = "unknown"

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

        if body_type == "unknown" and can_measure_upper_body:
            body_type = self.body_estimator.estimate_upper_body_type(pose_results)

        if height_cm is None:
            height_cm = self.body_estimator.estimate_height_cm_from_mask(pose_results)

            if height_cm is not None and not (140 <= height_cm <= 200):
                height_cm = None

        if body_type == "unknown":
            body_type = self.body_estimator.estimate_body_type_from_mask(pose_results)

        upper_color = self.upper_color_estimator.estimate(image, pose_results)

        # 5. 年龄性别识别，模型缺失或人脸不可用时返回 unknown
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
