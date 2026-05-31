import cv2

from vision.image_loader import load_image
from vision.face_detector import FaceDetector


def main():
    image_path = "test_images/test.jpg"
    output_path = "output_face.jpg"
    face_crop_path = "output_face_crop.jpg"

    image = load_image(image_path)

    detector = FaceDetector()
    faces = detector.detect(image)

    print(f"检测到 {len(faces)} 张人脸")

    output_image = detector.draw_faces(image, faces)
    cv2.imwrite(output_path, output_image)

    print(f"人脸检测结果已保存到: {output_path}")

    face_crop = detector.crop_largest_face(image, faces)

    if face_crop is not None:
        cv2.imwrite(face_crop_path, face_crop)
        print(f"最大人脸裁剪图已保存到: {face_crop_path}")
    else:
        print("没有裁剪人脸，因为未检测到人脸")


if __name__ == "__main__":
    main()