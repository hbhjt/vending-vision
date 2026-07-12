"""
视觉管线全局单例模块

提供 infer_image() 便捷函数作为向后兼容的旧接口。
内部使用全局 VisionPipeline 实例，避免每次推理都重新加载模型。
"""

from vision.vision_pipeline import VisionPipeline

# 全局管线实例（懒加载单例）
_pipeline = VisionPipeline()


def infer_image(image):
    """对单帧图像执行完整的视觉推理。

    这是一个兼容旧接口的便捷函数。
    外部调用 infer_image(image) 即可，内部使用全局 pipeline 实例。

    Args:
        image: BGR 格式的 OpenCV 图像

    Returns:
        VisionProfile 数据模型实例
    """
    return _pipeline.infer(image)
