from vision.image_loader import load_image
from vision.face_detector import FaceDetector
from vision.age_gender_estimator import AgeGenderEstimator


def main():
    image_path = "test_images/test.jpg"

    image = load_image(image_path)

    face_detector = FaceDetector()
    faces = face_detector.detect(image)

    print(f"检测到 {len(faces)} 张人脸")

    face_image = face_detector.crop_largest_face(image, faces)

    estimator = AgeGenderEstimator()
    age, gender = estimator.predict(face_image)

    print("年龄:", age)
    print("性别:", gender)


if __name__ == "__main__":
    main()