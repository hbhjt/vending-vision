import argparse
import csv
import os
import time
from datetime import datetime

import cv2

from vision.camera import capture_image, get_configured_camera_status
from vision.logger import logger
from vision.pipeline import infer_image
from vision.profile_mapper import vision_profile_to_protocol


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Real camera batch inference test")
    parser.add_argument("--runs", type=int, default=10, help="number of captures")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds between captures")
    parser.add_argument("--output-dir", default="test_reports", help="CSV output directory")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="save each captured frame beside the CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(args.output_dir, f"real_camera_batch_{timestamp}.csv")
    frame_dir = os.path.join(args.output_dir, f"frames_{timestamp}")

    if args.save_frames:
        ensure_dir(frame_dir)

    fields = [
        "index",
        "time",
        "personPresent",
        "heightCm",
        "shoulderWidthCm",
        "ageRange",
        "gender",
        "bodyType",
        "upperColor",
        "confidence",
        "raw_age",
        "raw_gender",
        "raw_height_cm",
        "raw_shoulder_width_cm",
        "raw_body_type",
        "raw_upper_color",
        "raw_presence",
        "frame_path",
        "error",
    ]

    print("================================")
    print("Real Camera Batch Test")
    print("================================")
    print(f"runs: {args.runs}")
    print(f"interval seconds: {args.interval}")
    print(f"output csv: {output_csv}")

    try:
        camera_status = get_configured_camera_status()
        print("camera status:")
        print(camera_status)
    except Exception as e:
        print(f"camera status failed: {e}")

    print("Please stand in the camera view with a natural front-facing posture.")
    print("================================")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i in range(1, args.runs + 1):
            print(f"\n[{i}/{args.runs}] capturing...")

            row = {
                "index": i,
                "time": datetime.now().isoformat(timespec="seconds"),
                "personPresent": None,
                "heightCm": None,
                "shoulderWidthCm": None,
                "ageRange": None,
                "gender": None,
                "bodyType": None,
                "upperColor": None,
                "confidence": None,
                "raw_age": None,
                "raw_gender": None,
                "raw_height_cm": None,
                "raw_shoulder_width_cm": None,
                "raw_body_type": None,
                "raw_upper_color": None,
                "raw_presence": None,
                "frame_path": "",
                "error": "",
            }

            try:
                image = capture_image()

                if args.save_frames:
                    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, image)
                    row["frame_path"] = frame_path

                profile = infer_image(image)
                protocol_profile = vision_profile_to_protocol(profile)

                row.update(
                    {
                        "personPresent": protocol_profile.get("personPresent"),
                        "heightCm": protocol_profile.get("heightCm"),
                        "shoulderWidthCm": protocol_profile.get("shoulderWidthCm"),
                        "ageRange": protocol_profile.get("ageRange"),
                        "gender": protocol_profile.get("gender"),
                        "bodyType": protocol_profile.get("bodyType"),
                        "upperColor": protocol_profile.get("upperColor"),
                        "confidence": protocol_profile.get("confidence"),
                        "raw_age": profile.age,
                        "raw_gender": profile.gender,
                        "raw_height_cm": profile.height_cm,
                        "raw_shoulder_width_cm": profile.shoulder_width_cm,
                        "raw_body_type": profile.body_type,
                        "raw_upper_color": profile.upper_color,
                        "raw_presence": profile.presence,
                    }
                )

                print("result:")
                print(f"  personPresent: {row['personPresent']}")
                print(f"  heightCm: {row['heightCm']}")
                print(f"  shoulderWidthCm: {row['shoulderWidthCm']}")
                print(f"  bodyType: {row['bodyType']}")
                print(f"  upperColor: {row['upperColor']}")
                print(f"  ageRange: {row['ageRange']}")
                print(f"  gender: {row['gender']}")
                print(f"  confidence: {row['confidence']}")

            except Exception as e:
                row["error"] = str(e)
                logger.exception("Real camera batch test failed")
                print(f"failed: {e}")

            writer.writerow(row)
            f.flush()

            if i < args.runs:
                time.sleep(args.interval)

    print("\n================================")
    print("Batch test finished")
    print(f"CSV saved to: {output_csv}")
    if args.save_frames:
        print(f"Frames saved to: {frame_dir}")
    print("================================")


if __name__ == "__main__":
    main()
