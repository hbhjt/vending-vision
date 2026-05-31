import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from vision.proximity import check_proximity_once


def parse_args():
    parser = argparse.ArgumentParser(description="Test proximity detection")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output-dir", default="test_reports/proximity")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"proximity_{timestamp}.json"
    records = []

    for index in range(1, args.runs + 1):
        result = check_proximity_once()
        record = {
            "index": index,
            "capturedAt": datetime.now().isoformat(timespec="milliseconds"),
            "result": result,
        }
        records.append(record)
        print(f"[{index}/{args.runs}]")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if index < args.runs:
            time.sleep(args.interval)

    summary = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "runs": args.runs,
        "interval": args.interval,
        "closeCount": sum(1 for item in records if item["result"].get("close")),
        "presentCount": sum(1 for item in records if item["result"].get("present")),
        "records": records,
    }

    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
