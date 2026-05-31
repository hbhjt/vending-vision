def detect_presence(image) -> bool:
    """
    人体靠近检测模块。

    当前是最小可运行版本：
    只要传入了图片，就暂时认为有人。

    后续可以替换为：
    1. PIR 传感器输入
    2. 毫米波雷达输入
    3. 摄像头轻量人体检测
    """
    if image is None:
        return False

    return True