from vision.vision_pipeline import VisionPipeline


_pipeline = VisionPipeline()


def infer_image(image):
    """
    兼容旧接口。

    外部仍然调用 infer_image(image)，
    但内部使用全局 pipeline，避免重复初始化模型。
    """
    return _pipeline.infer(image)