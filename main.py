import cv2

from vision.image_loader import load_image
from vision.pipeline import infer_image
from vision.pose_estimator import PoseEstimator


def main():
    image_path = "test_images/test.jpg"
    output_path = "output_pose.jpg"

    image = load_image(image_path)
    profile = infer_image(image)

    print(profile.model_dump_json(indent=2))

    pose_estimator = PoseEstimator()
    results = pose_estimator.detect(image)
    output_image = pose_estimator.draw_pose(image, results)
    cv2.imwrite(output_path, output_image)

    print(f"骨架图已保存到: {output_path}")


if __name__ == "__main__":
    main()