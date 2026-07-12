"""
数据模型（Schema）模块

定义核心的 VisionProfile 数据模型，用于在各模块间
传递人物画像的结构化数据。
"""

from pydantic import BaseModel
from typing import Literal, Optional


class VisionProfile(BaseModel):
    """人物画像数据模型。

    包含视觉模块可以从图像中提取的所有人物特征：
    - age: 估测年龄（整数）
    - gender: 性别（male/female/unknown）
    - height_cm: 估测身高（厘米）
    - shoulder_width_cm: 估测肩宽（厘米）
    - body_type: 体型（thin/medium/fat/unknown）
    - upper_color: 上衣颜色（dark/light/red/blue/green/yellow/white/black/unknown）
    - presence: 是否检测到人物存在
    """
    age: Optional[int] = None
    gender: Optional[Literal["male", "female", "unknown"]] = "unknown"
    height_cm: Optional[float] = None
    shoulder_width_cm: Optional[float] = None
    body_type: Optional[Literal["thin", "medium", "fat", "unknown"]] = "unknown"
    upper_color: Optional[str] = "unknown"
    presence: bool = False
