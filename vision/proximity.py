import cv2

from vision.camera import capture_image
from vision.config import settings
from vision.face_detector import FaceDetector


class ProximityMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.close_streak = 0

    def resize_for_monitor(self, image):
        width = settings.PROXIMITY_MONITOR_WIDTH
        height = settings.PROXIMITY_MONITOR_HEIGHT

        if not width or not height or width <= 0 or height <= 0:
            return image

        return cv2.resize(image, (width, height))

    def check_image(self, image):
        monitor_image = self.resize_for_monitor(image)
        h, w = monitor_image.shape[:2]
        faces = self.face_detector.detect(monitor_image)

        largest_face_ratio = 0.0

        if faces:
            largest = max(faces, key=lambda box: box[2] * box[3])
            largest_face_ratio = (largest[2] * largest[3]) / float(w * h)

        present = largest_face_ratio >= settings.PROXIMITY_PRESENT_FACE_RATIO
        close_now = largest_face_ratio >= settings.PROXIMITY_CLOSE_FACE_RATIO

        if close_now:
            self.close_streak += 1
        else:
            self.close_streak = 0

        close = (
            self.close_streak
            >= settings.PROXIMITY_CLOSE_CONSECUTIVE_FRAMES
        )

        return {
            "present": present,
            "close": close,
            "closeNow": close_now,
            "closeStreak": self.close_streak,
            "requiredCloseStreak": settings.PROXIMITY_CLOSE_CONSECUTIVE_FRAMES,
            "faceCount": len(faces),
            "largestFaceRatio": round(largest_face_ratio, 5),
            "presentFaceRatio": settings.PROXIMITY_PRESENT_FACE_RATIO,
            "closeFaceRatio": settings.PROXIMITY_CLOSE_FACE_RATIO,
            "monitorWidth": w,
            "monitorHeight": h,
            "method": "face_area_ratio",
        }

    def check_once(self):
        image = capture_image(warmup_frames=1)
        return self.check_image(image)


_monitor = None


def get_proximity_monitor():
    global _monitor

    if _monitor is None:
        _monitor = ProximityMonitor()

    return _monitor


def check_proximity_once():
    return get_proximity_monitor().check_once()
