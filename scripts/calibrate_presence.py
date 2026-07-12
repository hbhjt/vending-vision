from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from vision.proximity import ProximityMonitor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze recorded camera videos for presence threshold tuning."
    )
    parser.add_argument("videos", nargs="+", help="Video files to analyze.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Analyze every Nth frame. Default: 5.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum analyzed frames per video. 0 means no limit.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="reports/presence_calibration.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default="reports/presence_calibration_summary.json",
        help="Summary JSON output path.",
    )
    return parser.parse_args()


def percentile(values, pct):
    if not values:
        return None

    values = sorted(values)
    index = int(round((len(values) - 1) * pct / 100.0))
    return round(float(values[index]), 5)


def summarize(rows):
    ratio_fields = [
        "largestPersonRatio",
        "largestFaceRatio",
        "bodyBoxRatio",
    ]
    summary = {
        "frameCount": len(rows),
        "presentCount": len([row for row in rows if row["present"]]),
        "closeNowCount": len([row for row in rows if row["closeNow"]]),
        "multipleCount": len(
            [
                row
                for row in rows
                if max(row["personCount"], row["faceCount"]) > 1
            ]
        ),
    }

    for field in ratio_fields:
        values = [row[field] for row in rows if row[field] is not None]
        if not values:
            summary[field] = {}
            continue

        summary[field] = {
            "min": round(min(values), 5),
            "p50": percentile(values, 50),
            "p75": percentile(values, 75),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "max": round(max(values), 5),
            "mean": round(float(statistics.mean(values)), 5),
        }

    return summary


def analyze_video(path: Path, monitor: ProximityMonitor, frame_step: int, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")

    rows = []
    frame_index = 0
    analyzed = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % max(frame_step, 1) != 0:
                frame_index += 1
                continue

            result = monitor.check_image(frame)
            rows.append(
                {
                    "video": str(path),
                    "frameIndex": frame_index,
                    "present": bool(result.get("present")),
                    "close": bool(result.get("close")),
                    "closeNow": bool(result.get("closeNow")),
                    "closeStreak": int(result.get("closeStreak") or 0),
                    "personReady": bool(result.get("personReady")),
                    "personCount": int(result.get("personCount") or 0),
                    "faceCount": int(result.get("faceCount") or 0),
                    "personPresent": bool(result.get("personPresent")),
                    "facePresent": bool(result.get("facePresent")),
                    "bodyPresent": bool(result.get("bodyPresent")),
                    "largestPersonRatio": float(
                        result.get("largestPersonRatio") or 0.0
                    ),
                    "largestFaceRatio": float(result.get("largestFaceRatio") or 0.0),
                    "bodyBoxRatio": float(result.get("bodyBoxRatio") or 0.0),
                    "method": result.get("method"),
                }
            )
            analyzed += 1
            frame_index += 1

            if max_frames > 0 and analyzed >= max_frames:
                break
    finally:
        cap.release()

    return {
        "video": str(path),
        "analyzedFrames": analyzed,
        "elapsedMs": int((time.time() - started) * 1000),
        "summary": summarize(rows),
        "rows": rows,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    monitor = ProximityMonitor()
    all_rows = []
    video_summaries = []
    failures = []

    for video in args.videos:
        try:
            result = analyze_video(
                Path(video),
                monitor=monitor,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
            )
        except Exception as exc:
            failures.append(
                {
                    "video": video,
                    "error": str(exc),
                }
            )
            print(f"{video}: failed: {exc}")
            continue

        all_rows.extend(result["rows"])
        video_summaries.append(
            {
                "video": result["video"],
                "analyzedFrames": result["analyzedFrames"],
                "elapsedMs": result["elapsedMs"],
                "summary": result["summary"],
            }
        )
        print(
            f"{result['video']}: analyzed={result['analyzedFrames']} "
            f"present={result['summary']['presentCount']} "
            f"closeNow={result['summary']['closeNowCount']} "
            f"multiple={result['summary']['multipleCount']}"
        )

    summary = {
        "videos": video_summaries,
        "failures": failures,
        "combined": summarize(all_rows),
    }

    write_csv(Path(args.csv_path), all_rows)
    write_json(Path(args.json_path), summary)
    print(f"CSV: {args.csv_path}")
    print(f"JSON: {args.json_path}")

    if failures and not all_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
