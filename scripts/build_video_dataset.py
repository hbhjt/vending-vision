from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from vision.config import settings  # noqa: E402
from vision.frame_transform import camera_rotation, rotate_frame  # noqa: E402
from vision.profile_aggregation import profile_has_detected_field  # noqa: E402
from vision.profile_mapper import vision_profile_to_protocol  # noqa: E402
from vision.profile_sampling import (  # noqa: E402
    resize_for_profile_inference,
    score_frame_quality,
)
from vision.profile_state import (  # noqa: E402
    protocol_occupancy_snapshot,
    ResponsiveOccupancyFilter,
)
from vision.proximity import ProximityMonitor  # noqa: E402
from vision.pipeline import infer_image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build an auto-labeled dataset from recorded top/front vending videos "
            "and run the current vision pipeline against sampled frames."
        )
    )
    parser.add_argument("--top-video", default=None, help="Recorded top camera video.")
    parser.add_argument("--front-video", default=None, help="Recorded front camera video.")
    parser.add_argument(
        "--output-dir",
        default="reports/video_dataset/test01",
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--top-frame-step",
        type=int,
        default=15,
        help="Sample every Nth top-camera frame. Default: 15.",
    )
    parser.add_argument(
        "--front-frame-step",
        type=int,
        default=30,
        help="Sample every Nth front-camera frame. Default: 30.",
    )
    parser.add_argument(
        "--max-top-frames",
        type=int,
        default=0,
        help="Maximum sampled top frames. 0 means no limit.",
    )
    parser.add_argument(
        "--max-front-frames",
        type=int,
        default=0,
        help="Maximum sampled front frames. 0 means no limit.",
    )
    return parser.parse_args()


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else None
        return {
            "path": str(path),
            "frameCount": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "durationSec": round(duration_sec, 3) if duration_sec else None,
        }
    finally:
        cap.release()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    return ensure_dir(path)


def write_csv(path: Path, rows):
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def first_mp4_matching(pattern: str):
    matches = sorted(ROOT.glob(pattern))
    return matches[0] if matches else None


def resolve_video(explicit_path: str | None, fallback_pattern: str):
    if explicit_path:
        return Path(explicit_path)

    path = first_mp4_matching(fallback_pattern)
    if path is None:
        raise RuntimeError(f"no video found for pattern: {fallback_pattern}")
    return path


def save_frame(base_dir: Path, category: str, frame_index: int, frame):
    image_dir = ensure_dir(base_dir / category)
    image_path = image_dir / f"frame_{frame_index:06d}.jpg"
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"failed to write frame: {image_path}")

    return image_path


def process_top_video(video_path: Path, output_dir: Path, frame_step: int, max_frames: int):
    monitor = ProximityMonitor()
    occupancy_filter = ResponsiveOccupancyFilter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open top video: {video_path}")

    rows = []
    frame_index = 0
    sampled = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = max(int(frame_step), 1)
    rotation = camera_rotation(settings.TOP_CAMERA_CONFIG)
    frame_root = output_dir / "top" / "frames_by_occupancy"
    reset_dir(frame_root)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            frame = rotate_frame(frame, rotation)
            result = monitor.check_image(frame)
            top_occupancy = result.get("topOccupancy") or {}
            raw_top_occupancy = top_occupancy.get("occupancy", "unknown")
            occupancy = occupancy_filter.update(
                result,
                protocol_occupancy_snapshot(result),
            )["state"]
            image_path = save_frame(frame_root, occupancy, frame_index, frame)
            timestamp_ms = int(frame_index / fps * 1000) if fps > 0 else None
            rows.append(
                {
                    "sourceVideo": str(video_path),
                    "frameIndex": frame_index,
                    "timestampMs": timestamp_ms,
                    "imagePath": str(image_path),
                    "autoOccupancy": occupancy,
                    "rawTopOccupancy": raw_top_occupancy,
                    "present": bool(result.get("present")),
                    "rawCount": int(top_occupancy.get("rawCount") or 0),
                    "stableCount": int(top_occupancy.get("stableCount") or 0),
                    "occupancyConfidence": float(
                        top_occupancy.get("confidence") or 0.0
                    ),
                    "personCount": int(result.get("personCount") or 0),
                    "faceCount": int(result.get("faceCount") or 0),
                    "largestPersonRatio": float(
                        result.get("largestPersonRatio") or 0.0
                    ),
                    "largestPersonScore": result.get("largestPersonScore"),
                    "largestFaceRatio": float(result.get("largestFaceRatio") or 0.0),
                    "bodyBoxRatio": float(result.get("bodyBoxRatio") or 0.0),
                    "method": result.get("method"),
                }
            )
            sampled += 1
            frame_index += 1

            if max_frames > 0 and sampled >= max_frames:
                break
    finally:
        cap.release()

    return rows


def process_front_video(video_path: Path, output_dir: Path, frame_step: int, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open front video: {video_path}")

    rows = []
    frame_index = 0
    sampled = 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_step = max(int(frame_step), 1)
    rotation = camera_rotation(settings.FRONT_CAMERA_CONFIG)
    frame_root = output_dir / "front" / "frames_by_profile_validity"
    reset_dir(frame_root)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            frame = rotate_frame(frame, rotation)
            inference_frame = resize_for_profile_inference(frame)
            quality = score_frame_quality(inference_frame)
            profile = infer_image(inference_frame)

            if not quality["faceDetected"]:
                profile.age = None
                profile.gender = "unknown"

            protocol_profile = vision_profile_to_protocol(profile)
            has_field = profile_has_detected_field(profile)
            valid = bool(
                protocol_profile["personPresent"]
                and has_field
                and protocol_profile["confidence"] >= settings.PROFILE_MIN_CONFIDENCE
            )
            category = "valid_field_frame" if valid else "invalid_or_empty"
            image_path = save_frame(frame_root, category, frame_index, frame)
            timestamp_ms = int(frame_index / fps * 1000) if fps > 0 else None
            rows.append(
                {
                    "sourceVideo": str(video_path),
                    "frameIndex": frame_index,
                    "timestampMs": timestamp_ms,
                    "imagePath": str(image_path),
                    "valid": valid,
                    "hasProfileField": has_field,
                    "personPresent": bool(protocol_profile["personPresent"]),
                    "confidence": float(protocol_profile["confidence"]),
                    "heightCm": protocol_profile["heightCm"],
                    "shoulderWidthCm": protocol_profile["shoulderWidthCm"],
                    "ageRange": protocol_profile["ageRange"],
                    "gender": protocol_profile["gender"],
                    "bodyType": protocol_profile["bodyType"],
                    "upperColor": protocol_profile["upperColor"],
                    "qualityScore": quality["qualityScore"],
                    "personDetected": bool(quality["personDetected"]),
                    "personScore": quality["personScore"],
                    "personAreaRatio": quality["personAreaRatio"],
                    "faceDetected": bool(quality["faceDetected"]),
                    "faceScore": quality["faceScore"],
                    "faceAreaRatio": quality["faceAreaRatio"],
                    "brightness": quality["brightness"],
                    "sharpness": quality["sharpness"],
                }
            )
            sampled += 1
            frame_index += 1

            if max_frames > 0 and sampled >= max_frames:
                break
    finally:
        cap.release()

    return rows


def count_by(rows, field):
    counts = {}
    for row in rows:
        value = row.get(field)
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def main():
    args = parse_args()
    output_dir = ROOT / args.output_dir
    ensure_dir(output_dir)

    top_video = resolve_video(args.top_video, "*顶部*.mp4")
    front_video = resolve_video(args.front_video, "*中部*.mp4")

    top_rows = process_top_video(
        top_video,
        output_dir=output_dir,
        frame_step=args.top_frame_step,
        max_frames=args.max_top_frames,
    )
    front_rows = process_front_video(
        front_video,
        output_dir=output_dir,
        frame_step=args.front_frame_step,
        max_frames=args.max_front_frames,
    )

    top_csv = output_dir / "top_occupancy_auto_labels.csv"
    front_csv = output_dir / "front_profile_auto_labels.csv"
    summary_json = output_dir / "summary.json"
    manifest_json = output_dir / "manifest.json"

    write_csv(top_csv, top_rows)
    write_csv(front_csv, front_rows)

    top_info = video_info(top_video)
    front_info = video_info(front_video)
    summary = {
        "datasetType": "auto_labeled_review_required",
        "outputDir": str(output_dir),
        "topVideo": top_info,
        "frontVideo": front_info,
        "top": {
            "sampledFrames": len(top_rows),
            "frameStep": args.top_frame_step,
            "occupancyCounts": count_by(top_rows, "autoOccupancy"),
            "presentCount": len([row for row in top_rows if row["present"]]),
            "rawCountMax": max([row["rawCount"] for row in top_rows], default=0),
            "stableCountMax": max([row["stableCount"] for row in top_rows], default=0),
        },
        "front": {
            "sampledFrames": len(front_rows),
            "frameStep": args.front_frame_step,
            "validFrameCount": len([row for row in front_rows if row["valid"]]),
            "hasProfileFieldCount": len(
                [row for row in front_rows if row["hasProfileField"]]
            ),
            "bodyTypeCounts": count_by(front_rows, "bodyType"),
            "upperColorCounts": count_by(front_rows, "upperColor"),
            "ageRangeCounts": count_by(front_rows, "ageRange"),
            "genderCounts": count_by(front_rows, "gender"),
        },
        "files": {
            "topCsv": str(top_csv),
            "frontCsv": str(front_csv),
            "summaryJson": str(summary_json),
            "manifestJson": str(manifest_json),
        },
    }
    write_json(summary_json, summary)
    write_json(
        manifest_json,
        {
            "datasetType": "auto_labeled_review_required",
            "description": (
                "Frames are sampled from recorded vending-machine videos and "
                "labeled by the current model output. Use the CSV files to "
                "manually correct labels before treating this as ground truth."
            ),
            "summary": summary,
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
