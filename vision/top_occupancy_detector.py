"""
顶部摄像头人员占用检测模块

基于 YOLO 人体检测 + IOU 跟踪 + 历史平滑投票，判断当前画面中的人数状态：
- none: 无人
- single: 单人
- multiple: 多人
- unknown: 不确定（过渡状态）

设计目标：避免单帧抖动导致误判，通过多帧历史平滑输出稳定结果。
"""

from __future__ import annotations

import time
from collections import Counter, deque
from uuid import uuid4

from vision.config import settings
from vision.person_detector import PersonDetector


def _roi_from_config(config):
    """从配置中解析 ROI 区域。

    支持两种配置格式：
    - 列表: [x1, y1, x2, y2]（归一化坐标 0~1）
    - 字典: {x, y, width, height}
    """
    roi = config.get("roi", [0.0, 0.0, 1.0, 1.0])

    if isinstance(roi, dict):
        return [
            float(roi.get("x", 0.0)),
            float(roi.get("y", 0.0)),
            float(roi.get("x", 0.0)) + float(roi.get("width", 1.0)),
            float(roi.get("y", 0.0)) + float(roi.get("height", 1.0)),
        ]

    if isinstance(roi, list) and len(roi) == 4:
        return [float(item) for item in roi]

    return [0.0, 0.0, 1.0, 1.0]


def _box_center(box):
    """计算边界框的中心点坐标。"""
    x, y, width, height = box
    return x + width / 2.0, y + height / 2.0


def _box_iou(a, b):
    """计算两个边界框的 IOU (Intersection over Union)。

    用于判断两个检测框是否属于同一个跟踪目标。
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter

    if union <= 0:
        return 0.0

    return inter / union


class SimpleTrack:
    """简单的 IOU 跟踪器。

    为每个检测到的人分配唯一 ID，跨帧跟踪其位置。
    支持更新和标记丢失状态。
    """

    def __init__(self, detection):
        self.id = f"top-track-{uuid4()}"               # 唯一跟踪 ID
        self.bbox = list(detection["box"])               # 当前边界框
        self.score = float(detection.get("score") or 0.0) # 检测置信度
        self.age_frames = 1                               # 跟踪持续的帧数
        self.missed_frames = 0                            # 连续丢失的帧数
        self.updated_at = time.time()                     # 最近更新时间

    def update(self, detection):
        """用新的检测结果更新跟踪状态。"""
        self.bbox = list(detection["box"])
        self.score = float(detection.get("score") or 0.0)
        self.age_frames += 1
        self.missed_frames = 0
        self.updated_at = time.time()

    def mark_missed(self):
        """标记当前帧未匹配到检测结果。"""
        self.missed_frames += 1

    def public_state(self):
        """返回跟踪器的公开状态信息。"""
        return {
            "id": self.id,
            "bbox": self.bbox,
            "score": round(self.score, 4),
            "ageFrames": self.age_frames,
            "missedFrames": self.missed_frames,
        }


class TopOccupancyDetector:
    """顶部摄像头的人员占用检测器。

    工作原理：
    1. 在配置的 ROI 区域内运行 YOLO 人体检测
    2. 通过 IOU 跟踪维护每个人员的轨迹
    3. 稳定轨迹（age_frames >= track_min_age_frames）才计入人数
    4. 通过历史投票平滑输出 occupancy 状态
    5. 短暂消失期间保持上一个稳定状态（防止抖动）
    """

    def __init__(self, person_detector=None, config=None):
        self.config = dict(config or settings.TOP_OCCUPANCY_CONFIG)
        self.person_detector = person_detector or PersonDetector()
        # 占用状态历史记录（固定长度队列）
        self.history = deque(maxlen=max(int(self.config.get("history_size", 8)), 1))
        # 活跃跟踪目标列表
        self.tracks = []
        # 最近一次检测到人的时间戳
        self.last_present_at = None
        # 上一个稳定状态（用于短暂消失期间保持状态）
        self.last_stable_occupancy = "none"

    def detect(self, image, detections=None, detections_valid=True):
        """对一帧图像执行占用检测。

        Returns:
            包含 visible, occupancy, rawCount, stableCount, tracks, roi 等信息的字典。
        """
        # 1. ROI 内人体检测.  The caller may provide detections so proximity
        # can reuse one YOLO inference for both occupancy and area checks.
        detector_status = self.person_detector.status()
        if (
            not bool(self.config.get("enabled", True))
            or not detector_status.get("ready")
            or not detections_valid
        ):
            return {
                "visible": False,
                "inInteractionZone": False,
                "rawCount": 0,
                "stableCount": 0,
                "occupancy": "unknown",
                "confidence": 0.0,
                "tracks": [],
                "roi": self._roi_status(),
                "personBackend": detector_status.get("backend"),
            }

        if detections is None:
            try:
                detections = self.person_detector.detect(image)
            except Exception:
                return self.detect(image, detections=[], detections_valid=False)
        detections = self._filter_in_zone(image, detections)
        # 2. 更新跟踪器
        self._update_tracks(detections)

        # 3. 统计稳定跟踪目标
        stable_tracks = [
            track
            for track in self.tracks
            if track.age_frames >= int(self.config.get("track_min_age_frames", 3))
        ]
        raw_count = len(detections)
        stable_count = len(stable_tracks)
        frame_occupancy = self._frame_occupancy(raw_count, stable_count)
        now = time.time()

        if raw_count > 0:
            self.last_present_at = now

        # 4. 记录历史
        self.history.append(
            {
                "occupancy": frame_occupancy,
                "rawCount": raw_count,
                "stableCount": stable_count,
                "seenAt": now,
            }
        )
        # 5. 投票确定最终状态
        occupancy, confidence = self._vote_occupancy(now)
        visible = occupancy in {"single", "multiple"} or raw_count > 0
        in_zone = raw_count > 0 or stable_count > 0

        return {
            "visible": visible,
            "inInteractionZone": in_zone,
            "rawCount": raw_count,
            "stableCount": stable_count,
            "occupancy": occupancy,
            "confidence": confidence,
            "tracks": [track.public_state() for track in stable_tracks],
            "roi": self._roi_status(),
            "personBackend": detector_status.get("backend"),
        }

    def _detect_in_zone(self, image):
        """在 ROI 区域内检测人体。

        只保留中心点落在 ROI 内的检测结果，过滤 ROI 外的误检。
        """
        try:
            detections = self.person_detector.detect(image)
        except Exception:
            return []
        return self._filter_in_zone(image, detections)

    def _filter_in_zone(self, image, detections):
        """Filter already-computed detections by the configured ROI."""

        height, width = image.shape[:2]
        roi = _roi_from_config(self.config)
        x1 = max(0.0, min(roi[0], 1.0)) * width
        y1 = max(0.0, min(roi[1], 1.0)) * height
        x2 = max(0.0, min(roi[2], 1.0)) * width
        y2 = max(0.0, min(roi[3], 1.0)) * height

        in_zone = []
        for detection in detections:
            center_x, center_y = _box_center(detection["box"])
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                in_zone.append(detection)

        return in_zone

    def _update_tracks(self, detections):
        """使用 IOU 匹配更新跟踪器。

        匹配策略：
        - 对每个已有跟踪器，找 IOU 最大的新检测结果
        - IOU >= 阈值则匹配成功，更新跟踪器
        - 未匹配的跟踪器标记为丢失
        - 未匹配的检测结果创建新跟踪器
        - 丢失超过 max_missed_frames 的跟踪器被移除
        """
        threshold = float(self.config.get("track_iou_threshold", 0.3))
        max_missed = int(self.config.get("track_max_missed_frames", 5))
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        matches = []

        # IOU 匹配：每个跟踪器找最佳匹配的检测结果
        for track_index, track in enumerate(self.tracks):
            best_detection = None
            best_iou = 0.0

            for detection_index in unmatched_detections:
                iou = _box_iou(track.bbox, detections[detection_index]["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_detection = detection_index

            if best_detection is not None and best_iou >= threshold:
                matches.append((track_index, best_detection))
                unmatched_tracks.discard(track_index)
                unmatched_detections.discard(best_detection)

        # 应用匹配结果
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detections[detection_index])

        # 标记丢失
        for track_index in unmatched_tracks:
            self.tracks[track_index].mark_missed()

        # 创建新跟踪器
        for detection_index in unmatched_detections:
            self.tracks.append(SimpleTrack(detections[detection_index]))

        # 移除过期跟踪器
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= max_missed
        ]

    def _frame_occupancy(self, raw_count, stable_count):
        """根据原始/稳定检测数判断当前帧的占用状态。

        灵敏模式只使用当前帧检测数。跟踪目标可以继续保留用于身份匹配，
        但不能延长占用状态；短暂漏检由两帧历史窗口统一处理。
        0 -> none, 1 -> single, >=2 -> multiple
        """
        count = raw_count

        if count <= 0:
            return "none"

        if count == 1:
            return "single"

        return "multiple"

    def _vote_occupancy(self, now):
        """通过历史投票确定最终的占用状态。

        规则：
        1. 短暂消失保护：如果刚消失且上次有人的时间在 absent_min_seconds 内，
           保持上一个稳定状态
        2. 历史投票：遍历 multiple -> single -> none 优先级，
           任一状态在历史中出现的帧数 >= 阈值即确认
        3. 否则返回 unknown（避免单帧抖动触发画像采集）
        """
        if not self.history:
            return "unknown", 0.0

        # 短暂消失保护
        absent_min_seconds = float(self.config.get("absent_min_seconds", 1.5))
        if (
            self.last_present_at is not None
            and now - self.last_present_at < absent_min_seconds
            and self.history[-1]["occupancy"] == "none"
        ):
            return self.last_stable_occupancy, 0.55

        # 历史投票
        counts = Counter(item["occupancy"] for item in self.history)
        history_len = len(self.history)

        thresholds = {
            "none": int(self.config.get("present_min_frames", 4)),
            "single": int(self.config.get("single_min_frames", 5)),
            "multiple": int(self.config.get("multiple_min_frames", 4)),
        }

        # 按优先级从高到低检查
        for occupancy in ["multiple", "single", "none"]:
            required = thresholds[occupancy]
            if counts[occupancy] >= required:
                self.last_stable_occupancy = occupancy
                return occupancy, round(counts[occupancy] / float(history_len), 2)

        # 不确定状态：防止抖动帧触发画像采集
        return "unknown", round(max(counts.values()) / float(history_len), 2)

    def _roi_status(self):
        """返回当前 ROI 配置状态。"""
        x1, y1, x2, y2 = _roi_from_config(self.config)
        return {
            "x1": round(x1, 5),
            "y1": round(y1, 5),
            "x2": round(x2, 5),
            "y2": round(y2, 5),
        }
