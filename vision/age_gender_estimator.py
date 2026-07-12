"""
年龄性别识别模块

支持三种推理后端，按优先级自动选择：
1. OpenVINO IR（age-gender-recognition-retail-0013，最高精度）
2. OpenCV DNN + Caffe 模型（后备）
3. Mock 模式（最终降级，返回 unknown）

对输入人脸图像进行质量检查后再推理，低质量图像直接返回 unknown。
"""

import os

import cv2
import numpy as np

from vision.config import settings
from vision.logger import logger


class AgeGenderEstimator:
    """年龄性别识别器。

    输出年龄为整数（估测值），性别为 "male" / "female" / "unknown"。
    当模型不可用时自动降级为 mock 模式，不阻塞服务。
    """

    def __init__(self):
        # Caffe 模型的年龄分段标签
        self.age_list = [
            "(0-2)",
            "(4-6)",
            "(8-12)",
            "(15-20)",
            "(25-32)",
            "(38-43)",
            "(48-53)",
            "(60-100)"
        ]

        self.gender_list = ["male", "female"]

        self.model_ready = False                # 模型是否已加载成功
        self.age_net = None                     # Caffe 年龄网络
        self.gender_net = None                  # Caffe 性别网络
        self.openvino_compiled_model = None     # OpenVINO 编译后的模型
        self.openvino_input = None              # OpenVINO 输入层
        self.openvino_age_output = None         # OpenVINO 年龄输出层
        self.openvino_gender_output = None      # OpenVINO 性别输出层
        self.mode = "mock"                      # 当前模式: openvino / opencv_dnn / mock

        self._load_models()

    def _read_caffe_net(self, proto_path, weights_path):
        """加载 Caffe 模型，兼容不同 OpenCV 版本的 API。"""
        if hasattr(cv2.dnn, "readNetFromCaffe"):
            return cv2.dnn.readNetFromCaffe(proto_path, weights_path)

        if hasattr(cv2.dnn, "readNet"):
            try:
                return cv2.dnn.readNet(weights_path, proto_path, "Caffe")
            except Exception:
                return cv2.dnn.readNet(weights_path, proto_path)

        raise RuntimeError(
            "current OpenCV DNN build does not support Caffe model loading"
        )

    def _load_models(self):
        """按优先级加载模型：OpenVINO -> Caffe -> mock。"""
        # 优先尝试 OpenVINO
        if self._load_openvino_model():
            return

        # 检查 Caffe 模型文件是否完整
        required_files = [
            settings.AGE_MODEL_PROTO,
            settings.AGE_MODEL_WEIGHTS,
            settings.GENDER_MODEL_PROTO,
            settings.GENDER_MODEL_WEIGHTS
        ]

        missing_files = [path for path in required_files if not os.path.exists(path)]

        if missing_files:
            logger.warning(
                f"Age/Gender model files missing, fallback to mock: {missing_files}"
            )
            self.model_ready = False
            return

        try:
            self.age_net = self._read_caffe_net(
                settings.AGE_MODEL_PROTO,
                settings.AGE_MODEL_WEIGHTS
            )

            self.gender_net = self._read_caffe_net(
                settings.GENDER_MODEL_PROTO,
                settings.GENDER_MODEL_WEIGHTS
            )

            self.model_ready = True
            self.mode = "opencv_dnn"
            logger.info("Age/Gender models loaded successfully")

        except Exception as e:
            logger.warning(f"Age/Gender models unavailable, fallback to mock: {e}")
            self.model_ready = False
            self.mode = "mock"

    def _load_openvino_model(self):
        """尝试加载 OpenVINO IR 模型。

        OpenVINO 模型需要 .xml 和 .bin 两个文件。
        使用 retail-0013 模型，输出年龄（0~1 归一化值）和性别（[female, male] 概率）。
        """
        xml_path = settings.OPENVINO_AGE_GENDER_XML
        bin_path = settings.OPENVINO_AGE_GENDER_BIN

        if not (os.path.exists(xml_path) and os.path.exists(bin_path)):
            logger.warning("OpenVINO age/gender model missing, fallback to Caffe")
            return False

        try:
            from openvino.runtime import Core

            core = Core()
            model = core.read_model(model=xml_path, weights=bin_path)
            self.openvino_compiled_model = core.compile_model(model, "CPU")
            self.openvino_input = self.openvino_compiled_model.input(0)
            outputs = list(self.openvino_compiled_model.outputs)
            # 根据输出名称自动匹配年龄和性别输出层
            self.openvino_age_output = next(
                (output for output in outputs if "age" in output.get_any_name().lower()),
                outputs[0],
            )
            self.openvino_gender_output = next(
                (output for output in outputs if "gender" in output.get_any_name().lower()),
                outputs[-1],
            )
            self.model_ready = True
            self.mode = "openvino"
            logger.info(f"Age/Gender OpenVINO model loaded: {xml_path}")
            return True
        except Exception as e:
            logger.warning(f"OpenVINO age/gender unavailable, fallback to Caffe: {e}")
            self.openvino_compiled_model = None
            return False

    def _age_bucket_to_number(self, age_bucket: str):
        """将 Caffe 模型的年龄段标签映射为近似年龄数值。

        映射关系：
        (0-2) -> 1, (4-6) -> 5, (8-12) -> 10, (15-20) -> 18,
        (25-32) -> 28, (38-43) -> 40, (48-53) -> 50, (60-100) -> 65
        """
        mapping = {
            "(0-2)": 1,
            "(4-6)": 5,
            "(8-12)": 10,
            "(15-20)": 18,
            "(25-32)": 28,
            "(38-43)": 40,
            "(48-53)": 50,
            "(60-100)": 65
        }

        return mapping.get(age_bucket)

    def status(self):
        """返回年龄性别模型的状态信息。"""
        return {
            "ok": self.model_ready,
            "mode": self.mode if self.model_ready else "mock",
            "message": (
                f"age/gender models loaded: {self.mode}"
                if self.model_ready
                else "age/gender models not ready, fallback to mock"
            )
        }

    def _face_quality_ok(self, face_image, quality=None):
        """检查人脸图像质量是否满足推理要求。

        检查项：人脸存在性、人脸得分、人脸面积比、模糊度、亮度。
        """
        if face_image is None:
            return False

        if quality is None:
            return True

        if quality.get("faceScore", 1.0) < settings.PROFILE_SAMPLING_CONFIG.get("min_face_score", 0.45):
            return False

        if quality.get("faceAreaRatio", 1.0) < settings.PROFILE_SAMPLING_CONFIG.get("min_face_area_ratio", 0.01):
            return False

        if quality.get("blurScore", 999.0) < settings.PROFILE_SAMPLING_CONFIG.get("min_blur_score", 40.0):
            return False

        brightness = quality.get("brightness", 120.0)
        if brightness < settings.PROFILE_SAMPLING_CONFIG.get("brightness_min", 35):
            return False

        if brightness > settings.PROFILE_SAMPLING_CONFIG.get("brightness_max", 230):
            return False

        return True

    def _predict_openvino(self, face_image):
        """使用 OpenVINO 模型推理年龄和性别。

        retail-0013 模型：输入 62x62，输出年龄为 0~1 归一化值（乘以 100 得到实际年龄），
        性别输出 [female, male] 概率。
        """
        blob = cv2.dnn.blobFromImage(
            face_image,
            scalefactor=1.0,
            size=(62, 62),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        result = self.openvino_compiled_model([blob])
        age_raw = result[self.openvino_age_output]
        gender_raw = result[self.openvino_gender_output]
        age = int(round(float(age_raw.reshape(-1)[0]) * 100.0))
        gender_values = gender_raw.reshape(-1)
        gender = "unknown"

        if len(gender_values) >= 2:
            # retail-0013 通常输出 [female, male] 概率
            gender = "female" if gender_values[0] >= gender_values[1] else "male"

        return age, gender

    def predict(self, face_image, quality=None):
        """对裁剪后的人脸图像预测年龄和性别。

        Args:
            face_image: 裁剪后的人脸 ROI 图像
            quality: 人脸质量信息（可选，用于质量过滤）

        Returns:
            (age, gender) 元组
            - age: int 估测年龄，或 None（质量不足/模型不可用时）
            - gender: "male" / "female" / "unknown"
        """
        # 质量检查
        if not self._face_quality_ok(face_image, quality=quality):
            return None, "unknown"

        if not self.model_ready:
            return None, "unknown"

        # OpenVINO 推理
        if self.mode == "openvino":
            try:
                return self._predict_openvino(face_image)
            except Exception as e:
                logger.exception(f"OpenVINO Age/Gender prediction failed: {e}")
                return None, "unknown"

        # Caffe 推理
        try:
            # Caffe 模型输入：227x227，使用 ImageNet 均值
            blob = cv2.dnn.blobFromImage(
                face_image,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            # 性别预测
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_index = int(gender_preds[0].argmax())
            gender = self.gender_list[gender_index]

            # 年龄预测
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_index = int(age_preds[0].argmax())
            age_bucket = self.age_list[age_index]
            age = self._age_bucket_to_number(age_bucket)

            return age, gender

        except Exception as e:
            logger.exception(f"Age/Gender prediction failed: {e}")
            return None, "unknown"
