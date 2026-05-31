import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"图片读取失败，请检查路径是否正确: {image_path}")

    return image