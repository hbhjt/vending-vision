"""
图像帧变换模块

提供图像旋转和 ROI 裁剪功能。
摄像头可能以不同角度安装，需要根据配置对帧进行旋转归一化处理。
"""

from __future__ import annotations

import cv2


def normalize_rotation(value):
    """将各种旋转表示统一为 0/90/180/270 度。

    支持的输入格式：
    - 字符串: "none", "off", "0", "90", "cw90", "180", "270", "-90", "ccw90" 等
    - 整数: 0, 90, 180, 270（或等价角度）
    - None: 视为 0
    """
    if value is None:
        return 0

    if isinstance(value, str):
        value = value.strip().lower().replace("-", "_")
        mapping = {
            "none": 0,
            "off": 0,
            "0": 0,
            "90": 90,
            "cw90": 90,
            "clockwise_90": 90,
            "180": 180,
            "270": 270,
            "-90": 270,
            "ccw90": 270,
            "counterclockwise_90": 270,
            "counter_clockwise_90": 270,
        }
        return mapping.get(value, 0)

    try:
        degrees = int(value)
    except Exception:
        return 0

    return degrees % 360


def rotate_frame(image, rotation):
    """根据旋转角度对图像进行旋转。

    Args:
        image: 输入图像（numpy array）
        rotation: 旋转角度（支持 0, 90, 180, 270 及各种字符串表示）

    Returns:
        旋转后的图像
    """
    rotation = normalize_rotation(rotation)

    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def camera_rotation(config: dict):
    """从摄像头配置字典中提取并归一化旋转角度。

    支持多种配置键名：rotate > rotation > rotation_degrees。
    """
    return normalize_rotation(
        config.get(
            "rotate",
            config.get("rotation", config.get("rotation_degrees")),
        )
    )


def crop_normalized_roi(image, roi):
    """根据归一化的 ROI 区域裁剪图像。

    ROI 使用 0.0~1.0 的归一化坐标，适配不同分辨率的图像。

    Args:
        image: 输入图像
        roi: ROI 配置字典，包含 x, y, width, height（均为 0~1 归一化值）
             如果 enabled 为 False 或格式无效，返回原图

    Returns:
        (cropped_image, roi_info) 元组
        - cropped_image: 裁剪后的图像（或原图如果 ROI 无效）
        - roi_info: 实际应用的 ROI 信息字典（包含像素坐标）
    """
    if not isinstance(roi, dict) or not roi.get("enabled", True):
        return image, None

    height, width = image.shape[:2]
    x = float(roi.get("x", 0.0))
    y = float(roi.get("y", 0.0))
    roi_width = float(roi.get("width", 1.0))
    roi_height = float(roi.get("height", 1.0))

    # 将归一化坐标转为像素坐标，并 clamp 到有效范围
    x1 = int(max(0.0, min(x, 1.0)) * width)
    y1 = int(max(0.0, min(y, 1.0)) * height)
    x2 = int(max(0.0, min(x + roi_width, 1.0)) * width)
    y2 = int(max(0.0, min(y + roi_height, 1.0)) * height)

    if x2 <= x1 or y2 <= y1:
        return image, None

    cropped = image[y1:y2, x1:x2]
    return cropped, {
        "enabled": True,
        "x": round(x1 / float(width), 5),
        "y": round(y1 / float(height), 5),
        "width": round((x2 - x1) / float(width), 5),
        "height": round((y2 - y1) / float(height), 5),
        "pixelX": x1,
        "pixelY": y1,
        "pixelWidth": x2 - x1,
        "pixelHeight": y2 - y1,
    }
