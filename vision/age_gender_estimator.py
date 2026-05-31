import os

import cv2
import numpy as np

from vision.config import settings
from vision.logger import logger


class AgeGenderEstimator:
    def __init__(self):
        """
        年龄性别识别模块。

        当前使用 OpenCV DNN + Caffe 模型。
        如果模型文件不存在，则自动降级为 mock。
        """

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

        self.model_ready = False
        self.age_net = None
        self.gender_net = None

        self._load_models()

    def _load_models(self):
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
            self.age_net = cv2.dnn.readNetFromCaffe(
                settings.AGE_MODEL_PROTO,
                settings.AGE_MODEL_WEIGHTS
            )

            self.gender_net = cv2.dnn.readNetFromCaffe(
                settings.GENDER_MODEL_PROTO,
                settings.GENDER_MODEL_WEIGHTS
            )

            self.model_ready = True
            logger.info("Age/Gender models loaded successfully")

        except Exception as e:
            logger.exception(f"Failed to load Age/Gender models: {e}")
            self.model_ready = False

    def _age_bucket_to_number(self, age_bucket: str):
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
        """
        返回年龄性别模型状态。
        """
        return {
            "ok": self.model_ready,
            "mode": "opencv_dnn" if self.model_ready else "mock",
            "message": (
                "age/gender models loaded"
                if self.model_ready
                else "age/gender models not ready, fallback to mock"
            )
        }
    def predict(self, face_image):
        """
        输入裁剪后的人脸图片，输出年龄和性别。

        返回:
            age: int 或 None
            gender: "male" / "female" / "unknown"
        """

        if face_image is None:
            return None, "unknown"

        if not self.model_ready:
            return None, "unknown"

        try:
            blob = cv2.dnn.blobFromImage(
                face_image,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender_index = int(gender_preds[0].argmax())
            gender = self.gender_list[gender_index]

            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age_index = int(age_preds[0].argmax())
            age_bucket = self.age_list[age_index]
            age = self._age_bucket_to_number(age_bucket)

            return age, gender

        except Exception as e:
            logger.exception(f"Age/Gender prediction failed: {e}")
            return None, "unknown"
