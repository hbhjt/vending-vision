"""
摄像头底层操作模块

提供摄像头打开、参数设置、预热读取等底层功能。
支持 DirectShow (dshow) 和 Microsoft Media Foundation (msmf) 后端。
"""

from __future__ import annotations

import math

import cv2

from vision.config import settings


def get_camera_backend(backend_name: str | None = None):
    """根据名称获取 OpenCV 摄像头后端常量。

    支持: any, dshow (DirectShow), msmf (Media Foundation)。
    默认使用 DirectShow。
    """
    name = (backend_name or settings.CAMERA_BACKEND or "dshow").lower()

    mapping = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    return mapping.get(name, cv2.CAP_DSHOW)


def apply_camera_settings(
    cap,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    """Request one exact capture MediaType and return its normalized facts."""
    width = settings.CAMERA_WIDTH if width is None else width
    height = settings.CAMERA_HEIGHT if height is None else height
    fps = settings.CAMERA_FPS if fps is None else fps
    fourcc = settings.CAMERA_FOURCC if fourcc is None else fourcc
    try:
        width = int(width)
        height = int(height)
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("camera media type request is invalid") from exc
    fourcc = str(fourcc or "").upper()
    if width <= 0 or height <= 0 or fps <= 0 or len(fourcc) != 4:
        raise RuntimeError("camera media type request is invalid")

    # OpenCV's DirectShow backend commits width+height through IAMStreamConfig,
    # then may rebuild the graph again for FPS. Apply FOURCC last so the final
    # graph selects the requested subtype at the already-selected size/FPS.
    properties = (
        ("width", cv2.CAP_PROP_FRAME_WIDTH, width),
        ("height", cv2.CAP_PROP_FRAME_HEIGHT, height),
        ("fps", cv2.CAP_PROP_FPS, fps),
        ("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)),
    )
    requested = {"width": width, "height": height, "fps": fps, "fourcc": fourcc}
    for name, prop, value in properties:
        if not cap.set(prop, value):
            raise RuntimeError(
                "camera media type selection failed: "
                f"requested={_media_type_label(requested)}, property={name}"
            )
    return requested


def open_camera(
    camera_index: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    """打开摄像头并应用设置。

    返回 OpenCV VideoCapture 对象。
    如果摄像头无法打开，抛出 RuntimeError。
    """
    if camera_index is None:
        raise RuntimeError("camera capture requires a resolved Vision role binding")

    backend = get_camera_backend(backend_name)
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"camera unavailable, index={camera_index}, "
            f"backend={backend_name or settings.CAMERA_BACKEND}"
        )

    try:
        requested = apply_camera_settings(
            cap, width=width, height=height, fps=fps, fourcc=fourcc
        )
        actual = describe_capture(cap)
        fps_tolerance = max(0.5, requested["fps"] * 0.02)
        matches = (
            actual["width"] == requested["width"]
            and actual["height"] == requested["height"]
            and actual["fps"] is not None
            and abs(actual["fps"] - requested["fps"]) <= fps_tolerance
            and actual["fourcc"] == requested["fourcc"]
        )
        if not matches:
            raise RuntimeError(
                "camera media type negotiation failed: "
                f"requested={_media_type_label(requested)}, "
                f"actual={_media_type_label(actual)}"
            )
    except Exception:
        cap.release()
        raise

    return cap


def read_warmup_frame(cap, warmup_frames: int):
    """读取预热帧，丢弃前几帧不稳定图像。

    摄像头刚打开时前几帧可能曝光不足或为黑帧，
    通过跳过指定数量的帧来获取稳定的图像。
    """
    image = None

    for _ in range(max(1, warmup_frames)):
        ret, frame = cap.read()
        if ret:
            image = frame

    if image is None:
        raise RuntimeError("camera opened but failed to read a valid frame")

    return image


def describe_capture(cap):
    """获取摄像头的实际输出参数（分辨率、帧率、编码格式）。"""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps = round(raw_fps, 2) if math.isfinite(raw_fps) and raw_fps > 0 else None
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    raw_fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4))
    fourcc = raw_fourcc if all(32 <= ord(value) <= 126 for value in raw_fourcc) else None

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fourcc": fourcc,
    }


def _media_type_label(value: dict) -> str:
    width = value.get("width") or "unreported"
    height = value.get("height") or "unreported"
    fps = value.get("fps")
    fps_label = "unreported" if fps is None or fps <= 0 else f"{fps:g}"
    fourcc = value.get("fourcc") or "unreported"
    return f"{width}x{height}@{fps_label} {fourcc}"
