from vision.image_loader import load_image
from vision.pose_estimator import PoseEstimator
from vision.color_estimator import UpperColorEstimator


def main():
    image_path = "test_images/test.jpg"

    image = load_image(image_path)

    pose_estimator = PoseEstimator()
    pose_results = pose_estimator.detect(image)

    color_estimator = UpperColorEstimator()
    upper_color = color_estimator.estimate(image, pose_results)

    print("上衣颜色:", upper_color)


if __name__ == "__main__":
    main()