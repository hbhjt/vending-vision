import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2

from vision.camera import describe_capture, open_camera, read_warmup_frame
from vision.config import settings


DEFAULT_GROUPS = [
    "front_still",
    "operate_screen",
    "lean_forward",
    "turn_left",
    "turn_right",
]

GROUP_TIPS = {
    "front_still": "Stand naturally, face the screen/camera, arms relaxed.",
    "operate_screen": "Simulate tapping the vending machine screen.",
    "lean_forward": "Lean forward slightly as if reading the screen.",
    "turn_left": "Turn your upper body slightly to the left.",
    "turn_right": "Turn your upper body slightly to the right.",
}


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value)
    return value or "unknown"


def parse_groups(value: str):
    groups = [safe_name(item) for item in value.split(",") if item.strip()]
    return groups or DEFAULT_GROUPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect one person's field camera dataset."
    )
    parser.add_argument("--person-id", required=True, help="anonymous person id, e.g. p001")
    parser.add_argument("--height-cm", type=float, default=None, help="real height in cm")
    parser.add_argument("--age", type=int, default=None, help="real age")
    parser.add_argument(
        "--gender",
        choices=["male", "female", "unknown"],
        default="unknown",
        help="self-reported gender label for testing only",
    )
    parser.add_argument(
        "--body-type",
        choices=["thin", "medium", "fat", "unknown"],
        default="unknown",
        help="rough body type label for testing only",
    )
    parser.add_argument("--note", default="", help="free text note")
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="comma separated capture groups",
    )
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.35, help="seconds between frames")
    parser.add_argument("--countdown", type=int, default=3, help="seconds before each group")
    parser.add_argument("--output-root", default="datasets/field_capture")
    parser.add_argument("--no-prompt", action="store_true", help="do not pause between groups")
    parser.add_argument("--setup-height-cm", type=float, default=180.0)
    parser.add_argument("--setup-distance-cm", type=float, default=20.0)
    parser.add_argument("--setup-tilt-deg", type=float, default=45.0)
    return parser.parse_args()


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


def countdown(seconds: int):
    for remaining in range(seconds, 0, -1):
        print(f"  starting in {remaining}...")
        time.sleep(1)


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    groups = parse_groups(args.groups)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_id = safe_name(args.person_id)
    session_dir = Path(args.output_root) / f"{timestamp}_{person_id}"
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "sessionId": f"{timestamp}_{person_id}",
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "person": {
            "personId": person_id,
            "heightCm": args.height_cm,
            "age": args.age,
            "gender": args.gender,
            "bodyType": args.body_type,
            "note": args.note,
        },
        "setup": {
            "cameraMountHeightCm": args.setup_height_cm,
            "userDistanceCm": args.setup_distance_cm,
            "cameraTiltDeg": args.setup_tilt_deg,
        },
        "cameraConfig": {
            "index": settings.CAMERA_INDEX,
            "backend": settings.CAMERA_BACKEND,
            "width": settings.CAMERA_WIDTH,
            "height": settings.CAMERA_HEIGHT,
            "fps": settings.CAMERA_FPS,
            "fourcc": settings.CAMERA_FOURCC,
            "warmupFrames": settings.CAMERA_WARMUP_FRAMES,
        },
        "capturePlan": {
            "groups": groups,
            "samplesPerGroup": args.samples_per_group,
            "intervalSeconds": args.interval,
        },
    }

    manifest_path = session_dir / "manifest.csv"
    metadata_path = session_dir / "metadata.json"
    write_json(metadata_path, metadata)

    fields = [
        "person_id",
        "group",
        "sample_index",
        "image_path",
        "captured_at",
        "width",
        "height",
        "brightness",
        "sharpness",
        "height_cm",
        "age",
        "gender",
        "body_type",
        "setup_height_cm",
        "setup_distance_cm",
        "setup_tilt_deg",
    ]

    print("================================")
    print("Field Dataset Collection")
    print("================================")
    print(f"session: {session_dir}")
    print(f"person: {person_id}")
    print(f"groups: {', '.join(groups)}")
    print(f"samples per group: {args.samples_per_group}")
    print("Keep the camera at the final vending-machine position.")
    print("================================")

    cap = open_camera()

    try:
        camera_actual = describe_capture(cap)
        metadata["cameraActual"] = camera_actual
        write_json(metadata_path, metadata)

        print(f"camera actual: {camera_actual}")

        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for group in groups:
                tip = GROUP_TIPS.get(group, "Prepare for this capture group.")
                print("\n--------------------------------")
                print(f"group: {group}")
                print(f"tip: {tip}")

                if not args.no_prompt:
                    input("Press Enter when the person is ready...")

                countdown(args.countdown)

                group_dir = images_dir / group
                group_dir.mkdir(parents=True, exist_ok=True)

                for sample_index in range(1, args.samples_per_group + 1):
                    image = read_warmup_frame(cap, settings.CAMERA_WARMUP_FRAMES)
                    h, w = image.shape[:2]
                    quality = image_quality(image)
                    image_path = group_dir / f"{sample_index:03d}.jpg"

                    ok = cv2.imwrite(str(image_path), image)
                    if not ok:
                        raise RuntimeError(f"failed to save image: {image_path}")

                    row = {
                        "person_id": person_id,
                        "group": group,
                        "sample_index": sample_index,
                        "image_path": str(image_path),
                        "captured_at": datetime.now().isoformat(timespec="milliseconds"),
                        "width": w,
                        "height": h,
                        "brightness": quality["brightness"],
                        "sharpness": quality["sharpness"],
                        "height_cm": args.height_cm,
                        "age": args.age,
                        "gender": args.gender,
                        "body_type": args.body_type,
                        "setup_height_cm": args.setup_height_cm,
                        "setup_distance_cm": args.setup_distance_cm,
                        "setup_tilt_deg": args.setup_tilt_deg,
                    }
                    writer.writerow(row)
                    f.flush()

                    print(
                        f"  saved {group}/{sample_index:03d}.jpg "
                        f"brightness={quality['brightness']} sharpness={quality['sharpness']}"
                    )

                    if sample_index < args.samples_per_group:
                        time.sleep(args.interval)

    finally:
        cap.release()

    print("\n================================")
    print("Collection finished")
    print(f"metadata: {metadata_path}")
    print(f"manifest: {manifest_path}")
    print(f"images: {images_dir}")
    print("================================")


if __name__ == "__main__":
    main()
