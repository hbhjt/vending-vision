import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from vision.config import settings
from vision.face_detector import FaceDetector
from vision.logger import logger
from vision.person_detector import PersonDetector
from vision.pose_estimator import PoseEstimator


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value


def cleanup_old_traces():
    root = Path(settings.PROCESS_TRACE_OUTPUT_DIR)
    max_events = settings.PROCESS_TRACE_MAX_EVENTS

    if max_events <= 0 or not root.exists():
        return

    event_dirs = [item for item in root.iterdir() if item.is_dir()]

    if len(event_dirs) <= max_events:
        return

    event_dirs.sort(key=lambda path: path.stat().st_mtime)

    for event_dir in event_dirs[: len(event_dirs) - max_events]:
        try:
            shutil.rmtree(event_dir)
            logger.info(f"Deleted old process trace: {event_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete process trace {event_dir}: {e}")


class ProcessTrace:
    def __init__(self, event_id: str, reason: str = "profile_push"):
        self.enabled = bool(settings.PROCESS_TRACE_ENABLED)
        self.event_id = event_id
        self.reason = reason
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.root = None
        self.samples = []
        self.proximity = None

        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{timestamp}_{_safe_name(event_id)}"
        self.root = Path(settings.PROCESS_TRACE_OUTPUT_DIR) / name

        for subdir in [
            "00_proximity",
            "01_raw_samples",
            "02_inference_frames",
            "03_face_boxes",
            "04_face_crops",
            "05_pose",
        ]:
            (self.root / subdir).mkdir(parents=True, exist_ok=True)

        cleanup_old_traces()

    def _relative(self, path: Path | None):
        if path is None or self.root is None:
            return None

        return path.relative_to(self.root).as_posix()

    def _write_image(self, relative_path: str, image):
        if not self.enabled or self.root is None or image is None:
            return None

        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image)

        if not ok:
            logger.warning(f"Failed to write process trace image: {path}")
            return None

        return path

    def save_proximity(self, image, proximity: dict):
        if not self.enabled:
            return

        self.proximity = _json_safe(proximity)
        raw_path = self._write_image("00_proximity/proximity_raw.jpg", image)
        person_box_path = None

        try:
            face_detector = FaceDetector()
            monitor_width = proximity.get("monitorWidth")
            monitor_height = proximity.get("monitorHeight")
            monitor_image = image

            if monitor_width and monitor_height:
                monitor_image = cv2.resize(image, (monitor_width, monitor_height))

            faces = face_detector.detect(monitor_image)
            boxed_image = face_detector.draw_faces(monitor_image, faces)
            boxed_path = self._write_image(
                "00_proximity/proximity_face_boxes.jpg",
                boxed_image,
            )
        except Exception as e:
            logger.warning(f"Failed to draw proximity trace: {e}")
            boxed_path = None

        try:
            person_detector = PersonDetector()
            if person_detector.status()["ready"]:
                monitor_width = proximity.get("monitorWidth")
                monitor_height = proximity.get("monitorHeight")
                monitor_image = image

                if monitor_width and monitor_height:
                    monitor_image = cv2.resize(image, (monitor_width, monitor_height))

                person_image = monitor_image.copy()
                for item in person_detector.detect(monitor_image):
                    x, y, w, h = item["box"]
                    cv2.rectangle(person_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        person_image,
                        f"person {item['score']:.2f}",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

                person_box_path = self._write_image(
                    "00_proximity/proximity_person_boxes.jpg",
                    person_image,
                )
        except Exception as e:
            logger.warning(f"Failed to draw person proximity trace: {e}")

        self.proximity.update(
            {
                "rawImage": self._relative(raw_path),
                "faceBoxImage": self._relative(boxed_path),
                "personBoxImage": self._relative(person_box_path),
            }
        )

    def save_sample(
        self,
        index: int,
        raw_image,
        inference_image,
        profile,
        protocol_profile: dict,
        frame_quality: dict,
        valid: bool,
    ):
        if not self.enabled:
            return

        base = f"sample_{index:02d}"
        raw_path = self._write_image(f"01_raw_samples/{base}_raw.jpg", raw_image)
        inference_path = self._write_image(
            f"02_inference_frames/{base}_inference.jpg",
            inference_image,
        )

        face_box_path = None
        face_crop_path = None
        pose_path = None
        face_count = 0
        primary_info = {}

        try:
            face_detector = FaceDetector()
            pose_estimator = PoseEstimator()
            pose_results = pose_estimator.detect(inference_image)
            faces = face_detector.detect(inference_image)
            face_count = len(faces)
            primary_face, primary_info = face_detector.select_primary_face(
                inference_image,
                faces,
                pose_results=pose_results,
            )

            face_box_image = inference_image.copy()
            for face_index, (x, y, w, h) in enumerate(faces, start=1):
                is_primary = primary_face == (x, y, w, h)
                color = (0, 0, 255) if is_primary else (0, 255, 0)
                label = "primary" if is_primary else f"face {face_index}"
                cv2.rectangle(face_box_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    face_box_image,
                    label,
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

            face_box_path = self._write_image(
                f"03_face_boxes/{base}_faces.jpg",
                face_box_image,
            )

            if primary_face is not None:
                crop = face_detector.crop_face(inference_image, primary_face)
                face_crop_path = self._write_image(
                    f"04_face_crops/{base}_primary_face.jpg",
                    crop,
                )

            pose_image = pose_estimator.draw_pose(inference_image, pose_results)
            pose_path = self._write_image(f"05_pose/{base}_pose.jpg", pose_image)

        except Exception as e:
            logger.warning(f"Failed to save sample process trace: {e}")

        self.samples.append(
            {
                "index": index,
                "valid": valid,
                "quality": _json_safe(frame_quality),
                "profile": _json_safe(protocol_profile),
                "internalProfile": profile.model_dump() if profile else None,
                "faceCount": face_count,
                "primaryFace": _json_safe(primary_info),
                "rawImage": self._relative(raw_path),
                "inferenceImage": self._relative(inference_path),
                "faceBoxImage": self._relative(face_box_path),
                "primaryFaceCrop": self._relative(face_crop_path),
                "poseImage": self._relative(pose_path),
            }
        )

    def finish(self, status: str, payload: dict | None = None, reason: str | None = None):
        if not self.enabled or self.root is None:
            return None

        manifest = {
            "eventId": self.event_id,
            "reason": self.reason,
            "status": status,
            "statusReason": reason,
            "startedAt": self.started_at,
            "finishedAt": datetime.now().isoformat(timespec="seconds"),
            "proximity": self.proximity,
            "samples": self.samples,
            "payload": _json_safe(payload),
        }

        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Process trace saved: {self.root}")

        return str(self.root)
