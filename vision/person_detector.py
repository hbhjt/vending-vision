"""
人体检测模块

基于 YOLO11/YOLOv8 ONNX 模型的人体检测器。
使用 OpenCV DNN 模块进行推理，支持模型文件缺失时的自动降级。
"""

import os

import cv2
import numpy as np

from vision.config import settings
from vision.logger import logger


class PersonDetector:
    """YOLO 人体检测器。

    主要模型：YOLO11 ONNX (yolo11s.onnx)
    后备模型：YOLOv8 ONNX (person_yolov8n.onnx)
    降级策略：模型文件缺失时自动禁用，不阻塞服务启动

    输出格式：每帧返回检测到的人体边界框列表 [{"box": [x,y,w,h], "score": 0.xx, "classId": 0}, ...]
    """

    def __init__(self):
        self.enabled = bool(settings.PROXIMITY_PERSON_ENABLED)
        self.model_path = settings.PERSON_DETECTOR_MODEL
        self.requested_model_path = settings.PERSON_DETECTOR_MODEL
        self.net = None                       # OpenCV DNN 网络对象
        self.backend = "disabled"             # 当前后端状态
        self.last_error = None                # 最近一次错误信息

        if not self.enabled:
            return

        # 主模型不存在时尝试后备模型
        if not os.path.exists(self.model_path):
            fallback_path = getattr(settings, "PERSON_DETECTOR_FALLBACK_MODEL", None)
            if fallback_path and os.path.exists(fallback_path):
                self.model_path = fallback_path
                self.backend = "fallback_yolo"
                logger.warning(
                    "YOLO11 person detector missing, fallback to "
                    f"{self.model_path}"
                )
            else:
                self.backend = "missing"
                self.last_error = f"person detector model missing: {self.model_path}"
                logger.warning(self.last_error)
                return

        # 二次检查：后备模型也可能不存在
        if not os.path.exists(self.model_path):
            self.backend = "missing"
            self.last_error = f"person detector model missing: {self.model_path}"
            logger.warning(self.last_error)
            return

        try:
            self.net = cv2.dnn.readNet(self.model_path)
            if self.backend != "fallback_yolo":
                self.backend = "opencv_dnn_yolo11"
            logger.info(f"Person detector initialized: {self.model_path}")
        except Exception as e:
            self.backend = "error"
            self.last_error = str(e)
            logger.exception(f"Failed to load person detector: {e}")

    def status(self):
        """返回检测器的状态信息。"""
        return {
            "enabled": self.enabled,
            "ready": self.net is not None,
            "backend": self.backend,
            "requestedModelPath": self.requested_model_path,
            "modelPath": self.model_path,
            "inputWidth": settings.PERSON_DETECTOR_INPUT_WIDTH,
            "inputHeight": settings.PERSON_DETECTOR_INPUT_HEIGHT,
            "scoreThreshold": settings.PERSON_DETECTOR_SCORE_THRESHOLD,
            "nmsThreshold": settings.PERSON_DETECTOR_NMS_THRESHOLD,
            "lastError": self.last_error,
        }

    def detect(self, image):
        """对输入图像执行人体检测。

        处理流程：
        1. 图像预处理（blobFromImage）
        2. DNN 前向推理
        3. 解析 YOLO 输出格式
        4. NMS 非极大值抑制去重

        Returns:
            检测结果列表 [{"box": [x,y,w,h], "score": 0.xx, "classId": 0}, ...]
        """
        if self.net is None or image is None:
            return []

        input_width = settings.PERSON_DETECTOR_INPUT_WIDTH
        input_height = settings.PERSON_DETECTOR_INPUT_HEIGHT

        # 图像预处理：缩放、归一化、通道交换（BGR->RGB）
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(input_width, input_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)

        try:
            outputs = self.net.forward()
            self.last_error = None
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Person detector inference failed: {e}")
            raise

        detections = self._parse_yolo_output(outputs, image.shape[:2])

        if not detections:
            return []

        # NMS 非极大值抑制：去除重叠的重复检测框
        boxes = [item["box"] for item in detections]
        scores = [item["score"] for item in detections]
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            settings.PERSON_DETECTOR_SCORE_THRESHOLD,
            settings.PERSON_DETECTOR_NMS_THRESHOLD,
        )

        if len(indexes) == 0:
            return []

        flat_indexes = np.array(indexes).reshape(-1).tolist()
        return [detections[index] for index in flat_indexes]

    def _parse_yolo_output(self, outputs, image_shape):
        """解析 YOLO 模型的原始输出，适配多种 YOLO 导出格式。

        YOLO 不同版本/导出格式的输出形状可能不同：
        - YOLOv8: (1, 84, 8400) -> 需要转置
        - YOLOv5: (1, 25200, 85) -> 直接使用
        本方法自动检测并适配。
        """
        h, w = image_shape
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                return []
            outputs = outputs[0]

        output = np.array(outputs)

        if output.ndim == 3:
            output = output[0]

        if output.ndim != 2:
            return []

        # 自动检测输出格式并转置：(classes+4, anchors) -> (anchors, classes+4)
        if output.shape[0] < output.shape[1] and output.shape[0] in {5, 6, 84, 85}:
            output = output.T

        detections = []

        for row in output:
            parsed = self._parse_yolo_row(row, w, h)
            if parsed is not None:
                detections.append(parsed)

        return detections

    def _parse_yolo_row(self, row, image_width, image_height):
        """解析 YOLO 输出的单行数据。

        处理两种常见格式：
        - COCO 80类格式（85列）：[cx, cy, w, h, objectness, class_0, ..., class_79]
        - 简化格式（6列）：[cx, cy, w, h, conf, class_id]

        同时自动处理归一化坐标和像素坐标的转换。
        """
        if len(row) < 6:
            return None

        values = row.astype(float)
        class_id = settings.PERSON_DETECTOR_PERSON_CLASS_ID

        # 解析置信度分数
        if len(values) >= 85:
            # COCO 格式：objectness * class_score
            objectness = values[4]
            class_scores = values[5:]
            if class_id >= len(class_scores):
                return None
            score = objectness * class_scores[class_id]
        else:
            # 简化格式：直接用 class_score
            class_scores = values[4:]
            if class_id >= len(class_scores):
                return None
            score = class_scores[class_id]

        if score < settings.PERSON_DETECTOR_SCORE_THRESHOLD:
            return None

        cx, cy, box_w, box_h = values[:4]

        # 坐标转换：归一化坐标（0~2）vs 输入像素坐标
        if max(cx, cy, box_w, box_h) <= 2.0:
            # 归一化坐标：乘以图像实际尺寸
            cx *= image_width
            box_w *= image_width
            cy *= image_height
            box_h *= image_height
        else:
            # 输入尺寸像素坐标：缩放到图像实际尺寸
            cx *= image_width / float(settings.PERSON_DETECTOR_INPUT_WIDTH)
            box_w *= image_width / float(settings.PERSON_DETECTOR_INPUT_WIDTH)
            cy *= image_height / float(settings.PERSON_DETECTOR_INPUT_HEIGHT)
            box_h *= image_height / float(settings.PERSON_DETECTOR_INPUT_HEIGHT)

        # 从中心坐标转为左上角坐标，并裁剪到图像边界
        x = int(max(0, cx - box_w / 2.0))
        y = int(max(0, cy - box_h / 2.0))
        box_w = int(min(image_width - x, max(0, box_w)))
        box_h = int(min(image_height - y, max(0, box_h)))

        if box_w <= 0 or box_h <= 0:
            return None

        return {
            "box": [x, y, box_w, box_h],
            "score": round(float(score), 4),
            "classId": class_id,
        }
