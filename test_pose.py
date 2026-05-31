import cv2

from vision.image_loader import load_image
from vision.pose_estimator import PoseEstimator


def main():
    image_path = "test_images/test.jpg"
    output_path = "output_pose.jpg"

    image = load_image(image_path)

    pose_estimator = PoseEstimator()
    results = pose_estimator.detect(image)

    if pose_estimator.has_pose(results):
        print("检测到人体姿态")
    else:
        print("没有检测到人体姿态")

    output_image = pose_estimator.draw_pose(image, results)
    cv2.imwrite(output_path, output_image)

    print(f"结果图片已保存到: {output_path}")


if __name__ == "__main__":
    main()