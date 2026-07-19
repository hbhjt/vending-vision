"""
靠近检测模块

使用顶部摄像头检测售货机前是否有人靠近。
综合多种检测手段：
1. 人脸面积比 —— 快速判断是否有人
2. 人体检测（YOLO）—— 更可靠的靠近判断
3. 姿态关键点 —— 后备方案
4. TopOccupancyDetector —— 人员计数

"靠近"状态需要连续多帧确认（防止误触发），通过 close_streak 计数器实现。
"""

import cv2
import threading

from vision.camera_manager import read_camera_with_source
from vision.config import settings
from vision.frame_transform import crop_normalized_roi
from vision.face_detector import FaceDetector
from vision.person_detector import PersonDetector
from vision.pose_estimator import PoseEstimator
from vision.top_occupancy_detector import TopOccupancyDetector


class ProximityMonitor:
    """靠近检测监视器。

    聚合人脸检测、人体检测、姿态检测和人员计数，综合判断：
    - present: 是否有人出现在视野中
    - close: 是否有人足够靠近（连续多帧确认）
    - closeNow: 当前帧是否检测到靠近条件

    所有检测结果归一化后输出为统一格式的字典。
    """

    def __init__(self):
        self.face_detector = FaceDetector()
        self.person_detector = PersonDetector()
        self.top_occupancy_detector = TopOccupancyDetector(
            person_detector=self.person_detector,
        )
        self.pose_estimator = PoseEstimator()
        self.close_streak = 0    # 连续靠近帧计数器
        self.lock = threading.RLock()

    def resize_for_monitor(self, image):
        """将摄像头原图缩小到监控分辨率以提升性能。

        先应用 ROI 裁剪（如果配置了），再缩放到监控尺寸。
        """
        image, roi_status = crop_normalized_roi(
            image,
            settings.TOP_CAMERA_CONFIG.get("roi"),
        )
        width = settings.PROXIMITY_MONITOR_WIDTH
        height = settings.PROXIMITY_MONITOR_HEIGHT

        if not width or not height or width <= 0 or height <= 0:
            return image, roi_status

        return cv2.resize(image, (width, height)), roi_status

    def check_image(self, image):
        """对一帧图像执行完整的靠近检测。

        检测流程：
        1. 缩小图像以提高性能
        2. 人脸检测 -> 计算最大人脸面积比
        3. 人体检测 -> 计算最大人体面积比
        4. 人员计数 -> 判断单人/多人
        5. 姿态关键点 -> 后备方案（当人脸/人体检测均不可用时）
        6. 综合判断 present / close

        Returns:
            包含所有检测结果的详细字典
        """
        # The monitor owns mutable debounce/tracking state.  Serialising this
        # method also protects MediaPipe/OpenCV model instances from concurrent
        # debug and worker calls.
        with self.lock:
            return self._check_image_locked(image)

    def _check_image_locked(self, image):
        monitor_image, roi_status = self.resize_for_monitor(image)
        h, w = monitor_image.shape[:2]

        # ---- 人脸检测 ----
        faces = self.face_detector.detect(monitor_image)

        largest_face_ratio = 0.0
        largest_face_box = None

        if faces:
            largest = max(faces, key=lambda box: box[2] * box[3])
            largest_face_box = self.normalize_box(largest, w, h)
            largest_face_ratio = (largest[2] * largest[3]) / float(w * h)

        face_present = largest_face_ratio >= settings.PROXIMITY_PRESENT_FACE_RATIO
        face_close_now = largest_face_ratio >= settings.PROXIMITY_CLOSE_FACE_RATIO

        # ---- 人体检测 + 人员计数 ----
        person_detections = []
        person_status = self.person_detector.status()
        person_detections_valid = bool(person_status.get("ready"))
        if person_status.get("ready"):
            try:
                person_detections = self.person_detector.detect(monitor_image)
            except Exception:
                person_detections = []
                person_detections_valid = False
                person_status = dict(self.person_detector.status())
                person_status["ready"] = False
                person_status["backend"] = "error"
        top_occupancy = self.top_occupancy_detector.detect(
            monitor_image,
            detections=person_detections,
            detections_valid=person_detections_valid,
        )
        person_result = self.check_person(
            monitor_image,
            detections=person_detections,
            status=person_status,
        )
        occupancy_person_present = top_occupancy["occupancy"] in {"single", "multiple"}
        person_present = (
            settings.PROXIMITY_PERSON_ENABLED
            and person_result["ready"]
            and (
                person_result["largestPersonRatio"]
                >= settings.PROXIMITY_PRESENT_PERSON_RATIO
                or occupancy_person_present
            )
        )
        person_close_now = (
            settings.PROXIMITY_PERSON_ENABLED
            and person_result["ready"]
            and person_result["largestPersonRatio"]
            >= settings.PROXIMITY_CLOSE_PERSON_RATIO
        )

        # ---- 姿态检测（后备方案） ----
        # 如果人脸或人体检测已经可用，则跳过姿态检测以节省计算
        skip_body = face_close_now or person_close_now or person_present
        body_result = self.empty_body_result(skipped=skip_body)

        if settings.PROXIMITY_BODY_ENABLED and not skip_body:
            body_result = self.check_body(monitor_image)

        body_present = (
            settings.PROXIMITY_BODY_ENABLED
            and not body_result["skipped"]
            and body_result["visiblePointCount"]
            >= settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS
            and body_result["bodyBoxRatio"]
            >= settings.PROXIMITY_PRESENT_BODY_RATIO
        )
        body_close_now = (
            settings.PROXIMITY_BODY_ENABLED
            and not body_result["skipped"]
            and body_result["visiblePointCount"]
            >= settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS
            and body_result["bodyBoxRatio"]
            >= settings.PROXIMITY_CLOSE_BODY_RATIO
        )

        # ---- 综合判断 ----
        # present: 任一检测手段认为有人
        present = person_present or face_present or body_present or occupancy_person_present
        # closeNow: 任一检测手段认为靠近
        close_now = person_close_now or face_close_now or body_close_now

        # 连续靠近帧计数（防止单帧误判）
        if close_now:
            self.close_streak += 1
        else:
            self.close_streak = 0

        # close: 连续靠近帧数达到阈值才确认
        close = (
            self.close_streak
            >= settings.PROXIMITY_CLOSE_CONSECUTIVE_FRAMES
        )

        return {
            "present": present,
            "close": close,
            "closeNow": close_now,
            "closeStreak": self.close_streak,
            "requiredCloseStreak": settings.PROXIMITY_CLOSE_CONSECUTIVE_FRAMES,
            # 人脸检测结果
            "faceCount": len(faces),
            "facePresent": face_present,
            "faceCloseNow": face_close_now,
            "largestFaceRatio": round(largest_face_ratio, 5),
            "largestFaceBox": largest_face_box,
            "presentFaceRatio": settings.PROXIMITY_PRESENT_FACE_RATIO,
            "closeFaceRatio": settings.PROXIMITY_CLOSE_FACE_RATIO,
            # 人体检测结果
            "personEnabled": settings.PROXIMITY_PERSON_ENABLED,
            "personReady": person_result["ready"],
            "personBackend": person_result["backend"],
            "personLastError": person_result.get("lastError"),
            "personCount": person_result["personCount"],
            "rawCount": top_occupancy["rawCount"],
            "stableCount": top_occupancy["stableCount"],
            "topOccupancy": top_occupancy,
            "personPresent": person_present,
            "personCloseNow": person_close_now,
            "largestPersonRatio": round(person_result["largestPersonRatio"], 5),
            "largestPersonScore": person_result["largestPersonScore"],
            "largestPersonBox": person_result["largestPersonBox"],
            "presentPersonRatio": settings.PROXIMITY_PRESENT_PERSON_RATIO,
            "closePersonRatio": settings.PROXIMITY_CLOSE_PERSON_RATIO,
            # 姿态检测结果
            "bodyEnabled": settings.PROXIMITY_BODY_ENABLED,
            "bodyPresent": body_present,
            "bodyCloseNow": body_close_now,
            "bodySkipped": body_result["skipped"],
            "bodyVisiblePointCount": body_result["visiblePointCount"],
            "bodyBoxRatio": round(body_result["bodyBoxRatio"], 5),
            "bodyBox": body_result["bodyBox"],
            "presentBodyRatio": settings.PROXIMITY_PRESENT_BODY_RATIO,
            "closeBodyRatio": settings.PROXIMITY_CLOSE_BODY_RATIO,
            # 元信息
            "monitorWidth": w,
            "monitorHeight": h,
            "roi": roi_status,
            "method": (
                "person_detector+face_area_ratio"
                if person_result["ready"]
                else "face_area_ratio+pose_body_bbox_fallback"
            ),
        }

    def check_person(self, image, detections=None, status=None):
        """使用 YOLO 人体检测器检测人体。

        Returns:
            包含 ready, backend, personCount, largestPersonRatio 等字段的字典。
        """
        status = status or self.person_detector.status()
        result = {
            "ready": status["ready"],
            "backend": status["backend"],
            "personCount": 0,
            "largestPersonRatio": 0.0,
            "largestPersonScore": None,
            "largestPersonBox": None,
        }

        if not status["ready"]:
            return result

        if detections is None:
            try:
                detections = self.person_detector.detect(image)
            except Exception:
                status = self.person_detector.status()
                result["ready"] = False
                result["backend"] = "error"
                result["lastError"] = status.get("lastError")
                return result

        result["personCount"] = len(detections)

        if not detections:
            return result

        h, w = image.shape[:2]
        largest = max(detections, key=lambda item: item["box"][2] * item["box"][3])
        _, _, box_w, box_h = largest["box"]
        result["largestPersonRatio"] = (box_w * box_h) / float(w * h)
        result["largestPersonScore"] = largest["score"]
        result["largestPersonBox"] = self.normalize_box(largest["box"], w, h)
        return result

    def empty_body_result(self, skipped: bool = False):
        """返回空的姿态检测结果。"""
        return {
            "visiblePointCount": 0,
            "bodyBoxRatio": 0.0,
            "bodyBox": None,
            "skipped": skipped,
        }

    def check_body(self, image):
        """使用 MediaPipe 姿态关键点检测人体轮廓。

        从姿态关键点的可见点计算出人体边界框，用于判断是否有人靠近。
        需要至少 BODY_MIN_VISIBLE_POINTS 个可见关键点。
        """
        if not settings.PROXIMITY_BODY_ENABLED:
            return self.empty_body_result(skipped=True)

        try:
            results = self.pose_estimator.detect(image)
        except Exception:
            return self.empty_body_result()

        if not results.pose_landmarks:
            return self.empty_body_result()

        h, w = image.shape[:2]
        points = []

        # 收集所有可见度达标的姿态关键点
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility < settings.PROXIMITY_BODY_MIN_VISIBILITY:
                continue

            x = min(max(landmark.x, 0.0), 1.0) * w
            y = min(max(landmark.y, 0.0), 1.0) * h
            points.append((x, y))

        if len(points) < settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS:
            return {
                "visiblePointCount": len(points),
                "bodyBoxRatio": 0.0,
                "bodyBox": None,
                "skipped": False,
            }

        # 计算人体边界框
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min = min(xs)
        y_min = min(ys)
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        body_box_ratio = (box_width * box_height) / float(w * h)

        return {
            "visiblePointCount": len(points),
            "bodyBoxRatio": body_box_ratio,
            "bodyBox": self.normalize_box(
                [x_min, y_min, box_width, box_height],
                w,
                h,
            ),
            "skipped": False,
        }

    def normalize_box(self, box, image_width, image_height):
        """将像素坐标的边界框归一化为 0~1 范围。

        Returns:
            包含 x, y, width, height, centerX, centerY 的字典。
        """
        x, y, box_w, box_h = box
        center_x = (float(x) + float(box_w) / 2.0) / float(image_width)
        center_y = (float(y) + float(box_h) / 2.0) / float(image_height)

        return {
            "x": round(float(x) / float(image_width), 5),
            "y": round(float(y) / float(image_height), 5),
            "width": round(float(box_w) / float(image_width), 5),
            "height": round(float(box_h) / float(image_height), 5),
            "centerX": round(center_x, 5),
            "centerY": round(center_y, 5),
        }

    def check_once(
        self,
        return_image: bool = False,
        camera_role: str = "top",
        return_source: bool = False,
    ):
        """从摄像头读取一帧并执行检测（便捷方法）。"""
        image, source_frame = read_camera_with_source(camera_role, warmup_frames=1)
        result = self.check_image(image)

        if return_source:
            return result, image, source_frame
        if return_image:
            return result, image

        return result


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_monitor = None


def get_proximity_monitor():
    """获取全局 ProximityMonitor 单例（懒初始化）。"""
    global _monitor

    if _monitor is None:
        _monitor = ProximityMonitor()

    return _monitor


def check_proximity_once(camera_role: str = "top"):
    """从顶部摄像头读取一帧并执行靠近检测（便捷函数）。"""
    return get_proximity_monitor().check_once(camera_role=camera_role)


def check_proximity_once_with_image(camera_role: str = "top"):
    """从顶部摄像头读取一帧并执行靠近检测，同时返回图像。"""
    return get_proximity_monitor().check_once(
        return_image=True,
        camera_role=camera_role,
    )
