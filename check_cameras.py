import cv2


def test_camera_indices(max_index=10):
    available = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                h, w = frame.shape[:2]
                print(f"[OK] index={index}, resolution={w}x{h}")
                available.append(index)

                cv2.putText(
                    frame,
                    f"Camera index = {index}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

                cv2.imshow(f"Camera index {index}", frame)
                cv2.waitKey(800)
                cv2.destroyWindow(f"Camera index {index}")
            else:
                print(f"[FAIL READ] index={index}")

        else:
            print(f"[NO] index={index}")

        cap.release()

    print("\n可用摄像头 index:", available)


if __name__ == "__main__":
    test_camera_indices(max_index=10)
    cv2.destroyAllWindows()