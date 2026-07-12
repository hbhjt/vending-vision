"""
人脸检测模块

三级级联人脸检测器，按优先级自动选择可用模型：
1. SCRFD ONNX（最高精度，首选）
2. YuNet ONNX（OpenCV 内置，后备）
3. Haar Cascade（OpenCV 内置，最终降级）

同时提供主脸选择功能：优先通过姿态关键点匹配，其次按面积+中心位置加权。
"""

import os

import cv2
import mediapipe as mp

from vision.config import settings
from vision.logger import logger


class FaceDetector:
    """三级级联人脸检测器。

    初始化时自动检测可用模型并选择最优后端。
    所有 detect 接口返回统一的人脸框列表 [(x, y, w, h), ...]。
    detect_faces() 返回更丰富的信息（含置信度、关键点、后端名称）。
    """

    def __init__(self):
        self.backend = "haar"          # 当前使用的后端: scrfd / yunet / haar
        self.detector = None           # YuNet 检测器实例
        self.face_cascade = None       # Haar Cascade 分类器实例
        self.scrfd_net = None          # SCRFD DNN 网络实例

        # 按优先级尝试初始化检测器
        if os.path.exists(settings.SCRFD_FACE_DETECTOR_MODEL):
            self._init_scrfd()
        elif os.path.exists(settings.FACE_DETECTOR_MODEL) and hasattr(cv2, "FaceDetectorYN_create"):
            self._init_yunet()
        else:
            self._init_haar()

    def _init_scrfd(self):
        """初始化 SCRFD 人脸检测器（最高精度）。"""
        try:
            self.scrfd_net = cv2.dnn.readNet(settings.SCRFD_FACE_DETECTOR_MODEL)
            self.backend = "scrfd"
            logger.info(f"Face detector initialized: SCRFD {settings.SCRFD_FACE_DETECTOR_MODEL}")
        except Exception as e:
            logger.exception(f"SCRFD init failed, fallback to YuNet/Haar: {e}")
            if os.path.exists(settings.FACE_DETECTOR_MODEL) and hasattr(cv2, "FaceDetectorYN_create"):
                self._init_yunet()
            else:
                self._init_haar()

    def _init_yunet(self):
        """初始化 YuNet 人脸检测器（OpenCV 内置 DNN）。"""
        try:
            self.detector = cv2.FaceDetectorYN_create(
                settings.FACE_DETECTOR_MODEL,
                "",
                (320, 320),
                settings.FACE_SCORE_THRESHOLD,
                settings.FACE_NMS_THRESHOLD,
                settings.FACE_TOP_K
            )

            self.backend = "yunet"
            logger.info("Face detector initialized: YuNet")

        except Exception as e:
            logger.exception(f"YuNet init failed, fallback to Haar: {e}")
            self._init_haar()

    def _init_haar(self):
        """初始化 Haar Cascade 人脸检测器（最终降级方案）。"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Haar 人脸检测模型加载失败")

        self.backend = "haar"
        logger.info("Face detector initialized: Haar")

    def detect(self, image):
        """检测图像中的人脸。

        Args:
            image: OpenCV BGR 图像

        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        if image is None:
            return []

        if self.backend == "scrfd":
            faces = self.detect_faces(image)
            return [face["bbox"] for face in faces]

        if self.backend == "yunet":
            return self._detect_yunet(image)

        return self._detect_haar(image)

    def detect_faces(self, image):
        """统一的人脸检测接口，返回详细信息。

        Returns:
            人脸信息列表 [{"bbox": (x,y,w,h), "score": 0.xx, "landmarks": None, "backend": "scrfd"}, ...]
            非 SCRFD 后端时 score 固定为 1.0。
        """
        if image is None:
            return []

        if self.backend == "scrfd":
            try:
                return self._detect_scrfd_faces(image)
            except Exception as e:
                logger.warning(f"SCRFD inference failed, fallback face boxes: {e}")

        # 降级到 YuNet 或 Haar，score 固定为 1.0
        return [
            {
                "bbox": tuple(box),
                "score": 1.0,
                "landmarks": None,
                "backend": self.backend,
            }
            for box in (
                self._detect_yunet(image)
                if self.backend == "yunet"
                else self._detect_haar(image)
            )
        ]

    def _detect_scrfd_faces(self, image):
        """使用 SCRFD ONNX 模型检测人脸。

        SCRFD ONNX 导出格式存在差异，本方法做通用兜底解析。
        无法识别输出形态时返回空列表，不会让服务崩溃。
        """
        h, w = image.shape[:2]
        input_size = (640, 640)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 128.0,
            size=input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self.scrfd_net.setInput(blob)
        outputs = self.scrfd_net.forward(self.scrfd_net.getUnconnectedOutLayersNames())

        candidates = []
        for output in outputs:
            data = output.reshape(-1, output.shape[-1]) if output.ndim >= 2 else []
            for row in data:
                if len(row) < 5:
                    continue
                score = float(row[4])
                if score < settings.FACE_DETECTOR_CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = [float(item) for item in row[:4]]
                # 自动检测坐标格式：归一化（<=2.0）vs 输入像素坐标
                if max(x1, y1, x2, y2) <= 2.0:
                    x1 *= w
                    x2 *= w
                    y1 *= h
                    y2 *= h
                else:
                    x1 *= w / float(input_size[0])
                    x2 *= w / float(input_size[0])
                    y1 *= h / float(input_size[1])
                    y2 *= h / float(input_size[1])
                x = int(max(0, min(x1, x2)))
                y = int(max(0, min(y1, y2)))
                box_w = int(min(w - x, abs(x2 - x1)))
                box_h = int(min(h - y, abs(y2 - y1)))
                if box_w <= 0 or box_h <= 0:
                    continue
                candidates.append(
                    {
                        "bbox": (x, y, box_w, box_h),
                        "score": round(score, 4),
                        "landmarks": None,
                        "backend": "scrfd",
                    }
                )

        return candidates

    def _detect_yunet(self, image):
        """使用 YuNet 检测人脸。

        自动根据输入图像尺寸调整检测器输入大小。
        检测结果自动裁剪到图像边界内。
        """
        h, w = image.shape[:2]

        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        boxes = []

        for face in faces:
            x, y, box_w, box_h = face[:4]

            x = int(max(0, x))
            y = int(max(0, y))
            box_w = int(max(0, box_w))
            box_h = int(max(0, box_h))

            if box_w <= 0 or box_h <= 0:
                continue

            # 防止越界
            if x + box_w > w:
                box_w = w - x

            if y + box_h > h:
                box_h = h - y

            boxes.append((x, y, box_w, box_h))

        return boxes

    def _detect_haar(self, image):
        """使用 Haar Cascade 检测人脸。

        先将图像转为灰度图，再使用级联分类器检测。
        参数：缩放因子 1.1，最小邻居 5，最小人脸尺寸 50x50。
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return list(faces)

    def draw_faces(self, image, faces):
        """在图像上绘制绿色人脸边界框（用于调试可视化）。"""
        output = image.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        return output

    def crop_largest_face(self, image, faces):
        """裁剪面积最大的人脸（用于年龄性别识别）。"""
        if image is None or len(faces) == 0:
            return None

        largest_face = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_face

        return image[y:y + h, x:x + w]

    def crop_face(self, image, face):
        """裁剪指定的人脸区域。"""
        if image is None or face is None:
            return None

        x, y, w, h = face
        return image[y:y + h, x:x + w]

    def _box_center(self, box):
        """计算边界框的中心点坐标。"""
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def _head_center_from_pose(self, image, pose_results):
        """从 MediaPipe 姿态关键点估算头部中心位置。

        使用鼻子、左右眼、左右耳共 5 个关键点的可见点取平均，
        可见度阈值 >= 0.4。
        """
        if image is None or not pose_results or not pose_results.pose_landmarks:
            return None

        h, w = image.shape[:2]
        landmarks = pose_results.pose_landmarks.landmark
        pose_landmark = mp.solutions.pose.PoseLandmark
        # 头部相关关键点索引
        indexes = [
            pose_landmark.NOSE,
            pose_landmark.LEFT_EYE,
            pose_landmark.RIGHT_EYE,
            pose_landmark.LEFT_EAR,
            pose_landmark.RIGHT_EAR,
        ]

        points = []

        for index in indexes:
            lm = landmarks[index]
            if lm.visibility >= 0.4:
                points.append((lm.x * w, lm.y * h))

        if not points:
            return None

        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        return x, y

    def select_primary_face(self, image, faces, pose_results=None):
        """从多个人脸中选择主脸（最可能是目标用户的人脸）。

        选择策略（按优先级）：
        1. 姿态匹配法：如果有姿态检测结果，选择离头部中心最近的人脸
           （距离比需 <= PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO）
        2. 面积中心加权法：选择面积大且靠近图像中心的人脸

        Returns:
            (face_box, meta_info) 元组
        """
        if image is None or not faces:
            return None, {
                "method": "none",
                "faceCount": 0,
                "reason": "no face",
            }

        h, w = image.shape[:2]
        image_diag = (w ** 2 + h ** 2) ** 0.5
        head_center = self._head_center_from_pose(image, pose_results)

        # 策略1：基于姿态头部中心选择最近的人脸
        if head_center is not None:
            candidates = []

            for face in faces:
                face_center = self._box_center(face)
                distance = (
                    (face_center[0] - head_center[0]) ** 2
                    + (face_center[1] - head_center[1]) ** 2
                ) ** 0.5
                candidates.append((distance, face))

            distance, face = min(candidates, key=lambda item: item[0])
            distance_ratio = distance / image_diag if image_diag > 0 else 1.0

            if distance_ratio <= settings.PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO:
                return face, {
                    "method": "pose_head_nearest_face",
                    "faceCount": len(faces),
                    "headDistanceRatio": round(distance_ratio, 4),
                    "matched": True,
                }

        # 策略2：面积+中心位置加权评分
        image_center = (w / 2.0, h / 2.0)

        def fallback_score(face):
            x, y, box_w, box_h = face
            area_ratio = (box_w * box_h) / float(w * h)
            center = self._box_center(face)
            center_distance = (
                (center[0] - image_center[0]) ** 2
                + (center[1] - image_center[1]) ** 2
            ) ** 0.5
            center_distance_ratio = center_distance / image_diag if image_diag > 0 else 1.0
            # 面积越大越好，离中心越远越差
            return area_ratio - center_distance_ratio * 0.08

        face = max(faces, key=fallback_score)
        return face, {
            "method": "largest_center_weighted_face",
            "faceCount": len(faces),
            "matched": False,
            "reason": "pose head unavailable or too far from face",
        }
