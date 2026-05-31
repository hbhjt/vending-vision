import os
import re
from datetime import datetime

import cv2

from vision.config import settings
from vision.pose_estimator import PoseEstimator
from vision.logger import logger


def _safe_name(text: str | None) -> str:
    """
    把 sessionId 转成适合文件名的字符串。
    """
    if not text:
        return "no_session"

    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def cleanup_debug_images():
    """
    清理 debug_outputs 中过多的调试图片。

    当前策略：
    - 只统计 jpg / jpeg / png
    - 按修改时间排序
    - 超过 MAX_DEBUG_IMAGES 后删除最旧的图片
    """

    if not os.path.exists(settings.DEBUG_OUTPUT_DIR):
        return

    max_images = settings.MAX_DEBUG_IMAGES

    if max_images <= 0:
        return

    image_files = []

    for filename in os.listdir(settings.DEBUG_OUTPUT_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        file_path = os.path.join(settings.DEBUG_OUTPUT_DIR, filename)

        if os.path.isfile(file_path):
            image_files.append(file_path)

    if len(image_files) <= max_images:
        return

    image_files.sort(key=lambda path: os.path.getmtime(path))

    delete_count = len(image_files) - max_images
    files_to_delete = image_files[:delete_count]

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"Deleted old debug image: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete debug image {file_path}: {e}")


def save_debug_images(image, session_id: str | None = None):
    """
    保存调试图片：
    1. 摄像头原图
    2. 人体骨架图

    返回:
        {
            "rawImagePath": "...",
            "poseImagePath": "..."
        }
    """

    if not settings.SAVE_DEBUG_IMAGES:
        return {}

    if image is None:
        return {}

    os.makedirs(settings.DEBUG_OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session = _safe_name(session_id)

    raw_path = os.path.join(
        settings.DEBUG_OUTPUT_DIR,
        f"raw_{timestamp}_{safe_session}.jpg"
    )

    pose_path = os.path.join(
        settings.DEBUG_OUTPUT_DIR,
        f"pose_{timestamp}_{safe_session}.jpg"
    )

    cv2.imwrite(raw_path, image)

    try:
        pose_estimator = PoseEstimator()
        pose_results = pose_estimator.detect(image)
        pose_image = pose_estimator.draw_pose(image, pose_results)
        cv2.imwrite(pose_path, pose_image)

    except Exception as e:
        logger.exception(f"保存骨架调试图失败: {e}")
        pose_path = None

    logger.info(f"Debug raw image saved: {raw_path}")

    if pose_path:
        logger.info(f"Debug pose image saved: {pose_path}")

    cleanup_debug_images()

    return {
        "rawImagePath": raw_path,
        "poseImagePath": pose_path
    }