import cv2

from vision.camera_manager import read_camera
from vision.config import settings
from vision.face_detector import FaceDetector
from vision.person_detector import PersonDetector
from vision.pose_estimator import PoseEstimator


class ProximityMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.person_detector = PersonDetector()
        self.pose_estimator = PoseEstimator()
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
        largest_face_box = None

        if faces:
            largest = max(faces, key=lambda box: box[2] * box[3])
            largest_face_box = self.normalize_box(largest, w, h)
            largest_face_ratio = (largest[2] * largest[3]) / float(w * h)

        face_present = largest_face_ratio >= settings.PROXIMITY_PRESENT_FACE_RATIO
        face_close_now = largest_face_ratio >= settings.PROXIMITY_CLOSE_FACE_RATIO

        person_result = self.check_person(monitor_image)
        person_present = (
            settings.PROXIMITY_PERSON_ENABLED
            and person_result["ready"]
            and person_result["largestPersonRatio"]
            >= settings.PROXIMITY_PRESENT_PERSON_RATIO
        )
        person_close_now = (
            settings.PROXIMITY_PERSON_ENABLED
            and person_result["ready"]
            and person_result["largestPersonRatio"]
            >= settings.PROXIMITY_CLOSE_PERSON_RATIO
        )

        skip_body = face_close_now or person_close_now or person_result["ready"]
        body_result = self.empty_body_result(skipped=skip_body)

        if settings.PROXIMITY_BODY_ENABLED and not skip_body:
            body_result = self.check_body(monitor_image)

        body_present = (
            settings.PROXIMITY_BODY_ENABLED
            and not body_result["skipped"]
            and body_result["visiblePointCount"]
            >= settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS
            and body_result["bodyBoxRatio"]
            >= settings.PROXIMITY_PRESENT_BODY_RATIO
        )
        body_close_now = (
            settings.PROXIMITY_BODY_ENABLED
            and not body_result["skipped"]
            and body_result["visiblePointCount"]
            >= settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS
            and body_result["bodyBoxRatio"]
            >= settings.PROXIMITY_CLOSE_BODY_RATIO
        )

        present = person_present or face_present or body_present
        close_now = person_close_now or face_close_now or body_close_now

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
            "facePresent": face_present,
            "faceCloseNow": face_close_now,
            "largestFaceRatio": round(largest_face_ratio, 5),
            "largestFaceBox": largest_face_box,
            "presentFaceRatio": settings.PROXIMITY_PRESENT_FACE_RATIO,
            "closeFaceRatio": settings.PROXIMITY_CLOSE_FACE_RATIO,
            "personEnabled": settings.PROXIMITY_PERSON_ENABLED,
            "personReady": person_result["ready"],
            "personBackend": person_result["backend"],
            "personLastError": person_result.get("lastError"),
            "personCount": person_result["personCount"],
            "personPresent": person_present,
            "personCloseNow": person_close_now,
            "largestPersonRatio": round(person_result["largestPersonRatio"], 5),
            "largestPersonScore": person_result["largestPersonScore"],
            "largestPersonBox": person_result["largestPersonBox"],
            "presentPersonRatio": settings.PROXIMITY_PRESENT_PERSON_RATIO,
            "closePersonRatio": settings.PROXIMITY_CLOSE_PERSON_RATIO,
            "bodyEnabled": settings.PROXIMITY_BODY_ENABLED,
            "bodyPresent": body_present,
            "bodyCloseNow": body_close_now,
            "bodySkipped": body_result["skipped"],
            "bodyVisiblePointCount": body_result["visiblePointCount"],
            "bodyBoxRatio": round(body_result["bodyBoxRatio"], 5),
            "bodyBox": body_result["bodyBox"],
            "presentBodyRatio": settings.PROXIMITY_PRESENT_BODY_RATIO,
            "closeBodyRatio": settings.PROXIMITY_CLOSE_BODY_RATIO,
            "monitorWidth": w,
            "monitorHeight": h,
            "method": (
                "person_detector+face_area_ratio"
                if person_result["ready"]
                else "face_area_ratio+pose_body_bbox_fallback"
            ),
        }

    def check_person(self, image):
        status = self.person_detector.status()
        result = {
            "ready": status["ready"],
            "backend": status["backend"],
            "personCount": 0,
            "largestPersonRatio": 0.0,
            "largestPersonScore": None,
            "largestPersonBox": None,
        }

        if not status["ready"]:
            return result

        try:
            detections = self.person_detector.detect(image)
        except Exception:
            status = self.person_detector.status()
            result["ready"] = False
            result["backend"] = "error"
            result["lastError"] = status.get("lastError")
            return result

        result["personCount"] = len(detections)

        if not detections:
            return result

        h, w = image.shape[:2]
        largest = max(detections, key=lambda item: item["box"][2] * item["box"][3])
        _, _, box_w, box_h = largest["box"]
        result["largestPersonRatio"] = (box_w * box_h) / float(w * h)
        result["largestPersonScore"] = largest["score"]
        result["largestPersonBox"] = self.normalize_box(largest["box"], w, h)
        return result

    def empty_body_result(self, skipped: bool = False):
        return {
            "visiblePointCount": 0,
            "bodyBoxRatio": 0.0,
            "bodyBox": None,
            "skipped": skipped,
        }

    def check_body(self, image):
        if not settings.PROXIMITY_BODY_ENABLED:
            return self.empty_body_result(skipped=True)

        try:
            results = self.pose_estimator.detect(image)
        except Exception:
            return self.empty_body_result()

        if not results.pose_landmarks:
            return self.empty_body_result()

        h, w = image.shape[:2]
        points = []

        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility < settings.PROXIMITY_BODY_MIN_VISIBILITY:
                continue

            x = min(max(landmark.x, 0.0), 1.0) * w
            y = min(max(landmark.y, 0.0), 1.0) * h
            points.append((x, y))

        if len(points) < settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS:
            return {
                "visiblePointCount": len(points),
                "bodyBoxRatio": 0.0,
                "bodyBox": None,
                "skipped": False,
            }

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min = min(xs)
        y_min = min(ys)
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        body_box_ratio = (box_width * box_height) / float(w * h)

        return {
            "visiblePointCount": len(points),
            "bodyBoxRatio": body_box_ratio,
            "bodyBox": self.normalize_box(
                [x_min, y_min, box_width, box_height],
                w,
                h,
            ),
            "skipped": False,
        }

    def normalize_box(self, box, image_width, image_height):
        x, y, box_w, box_h = box
        center_x = (float(x) + float(box_w) / 2.0) / float(image_width)
        center_y = (float(y) + float(box_h) / 2.0) / float(image_height)

        return {
            "x": round(float(x) / float(image_width), 5),
            "y": round(float(y) / float(image_height), 5),
            "width": round(float(box_w) / float(image_width), 5),
            "height": round(float(box_h) / float(image_height), 5),
            "centerX": round(center_x, 5),
            "centerY": round(center_y, 5),
        }

    def check_once(self, return_image: bool = False, camera_role: str = "top"):
        image = read_camera(camera_role, warmup_frames=1)
        result = self.check_image(image)

        if return_image:
            return result, image

        return result


_monitor = None


def get_proximity_monitor():
    global _monitor

    if _monitor is None:
        _monitor = ProximityMonitor()

    return _monitor


def check_proximity_once(camera_role: str = "top"):
    return get_proximity_monitor().check_once(camera_role=camera_role)


def check_proximity_once_with_image(camera_role: str = "top"):
    return get_proximity_monitor().check_once(
        return_image=True,
        camera_role=camera_role,
    )
