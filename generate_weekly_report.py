import argparse
import csv
import json
import shutil
from datetime import date, datetime
from pathlib import Path

import cv2

from vision.face_detector import FaceDetector
from vision.pipeline import infer_image
from vision.pose_estimator import PoseEstimator
from vision.profile_mapper import vision_profile_to_protocol


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual intermediate results and a weekly report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="image file or directory, e.g. datasets/field_capture",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory, default reports/weekly_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="max images to process",
    )
    parser.add_argument(
        "--title",
        default="机器视觉模块周报",
        help="report title",
    )
    return parser.parse_args()


def find_images(path: Path, limit: int):
    if path.is_file():
        images = [path]
    else:
        images = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]

    return images[:limit]


def safe_stem(path: Path, index: int):
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    return f"{index:03d}_{stem}"


def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": round(float(gray.mean()), 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def save_face_crops(image, faces, output_dir: Path, base_name: str):
    paths = []

    for index, (x, y, w, h) in enumerate(faces, start=1):
        crop = image[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        path = output_dir / f"{base_name}_face_{index}.jpg"
        cv2.imwrite(str(path), crop)
        paths.append(path)

    return paths


def draw_face_boxes(image, faces, primary_face=None):
    output = image.copy()

    for index, (x, y, w, h) in enumerate(faces, start=1):
        is_primary = primary_face == (x, y, w, h)
        color = (0, 0, 255) if is_primary else (0, 255, 0)
        label = "primary" if is_primary else f"face {index}"

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return output


def relative(path: Path, root: Path):
    return path.relative_to(root).as_posix()


def process_images(images, output_dir: Path):
    original_dir = output_dir / "original"
    face_box_dir = output_dir / "face_boxes"
    face_crop_dir = output_dir / "face_crops"
    pose_dir = output_dir / "pose"

    for directory in [original_dir, face_box_dir, face_crop_dir, pose_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    face_detector = FaceDetector()
    pose_estimator = PoseEstimator()

    rows = []

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            rows.append(
                {
                    "index": index,
                    "source": str(image_path),
                    "error": "failed to read image",
                }
            )
            continue

        base_name = safe_stem(image_path, index)
        original_path = original_dir / f"{base_name}.jpg"
        face_box_path = face_box_dir / f"{base_name}_faces.jpg"
        pose_path = pose_dir / f"{base_name}_pose.jpg"

        cv2.imwrite(str(original_path), image)

        quality = image_quality(image)
        pose_results = pose_estimator.detect(image)
        faces = face_detector.detect(image)
        primary_face, primary_info = face_detector.select_primary_face(
            image,
            faces,
            pose_results=pose_results,
        )

        face_box_image = draw_face_boxes(image, faces, primary_face=primary_face)
        cv2.imwrite(str(face_box_path), face_box_image)

        face_crop_paths = save_face_crops(image, faces, face_crop_dir, base_name)

        pose_image = pose_estimator.draw_pose(image, pose_results)
        cv2.imwrite(str(pose_path), pose_image)

        profile = infer_image(image)
        protocol_profile = vision_profile_to_protocol(profile)

        rows.append(
            {
                "index": index,
                "source": str(image_path),
                "original": str(original_path),
                "face_box": str(face_box_path),
                "pose": str(pose_path),
                "face_crops": ";".join(str(path) for path in face_crop_paths),
                "face_count": len(faces),
                "primary_face": str(primary_face) if primary_face else "",
                "primary_method": primary_info.get("method", ""),
                "primary_matched": primary_info.get("matched", ""),
                "brightness": quality["brightness"],
                "sharpness": quality["sharpness"],
                "personPresent": protocol_profile["personPresent"],
                "heightCm": protocol_profile["heightCm"],
                "shoulderWidthCm": protocol_profile["shoulderWidthCm"],
                "ageRange": protocol_profile["ageRange"],
                "gender": protocol_profile["gender"],
                "bodyType": protocol_profile["bodyType"],
                "upperColor": protocol_profile["upperColor"],
                "confidence": protocol_profile["confidence"],
                "error": "",
            }
        )

    return rows


def write_csv(rows, output_dir: Path):
    csv_path = output_dir / "weekly_results.csv"
    fields = [
        "index",
        "source",
        "original",
        "face_box",
        "pose",
        "face_crops",
        "face_count",
        "primary_face",
        "primary_method",
        "primary_matched",
        "brightness",
        "sharpness",
        "personPresent",
        "heightCm",
        "shoulderWidthCm",
        "ageRange",
        "gender",
        "bodyType",
        "upperColor",
        "confidence",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def summarize(rows):
    valid_rows = [row for row in rows if not row.get("error")]
    present_rows = [row for row in valid_rows if row.get("personPresent") is True]
    face_rows = [row for row in valid_rows if int(row.get("face_count") or 0) > 0]
    confidence_values = [
        float(row["confidence"])
        for row in valid_rows
        if row.get("confidence") not in (None, "")
    ]

    avg_confidence = (
        round(sum(confidence_values) / len(confidence_values), 2)
        if confidence_values
        else None
    )

    return {
        "total": len(rows),
        "valid": len(valid_rows),
        "person_present": len(present_rows),
        "face_detected": len(face_rows),
        "avg_confidence": avg_confidence,
    }


def write_report(title, input_path: Path, output_dir: Path, rows, csv_path: Path):
    report_path = output_dir / "WEEKLY_REPORT.md"
    summary = summarize(rows)
    examples = [row for row in rows if not row.get("error")][:8]

    lines = [
        f"# {title}",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"数据来源：`{input_path}`",
        "",
        "## 本周进展",
        "",
        "- 完成现场图片中间结果生成，包括人脸框、姿态骨架、人脸裁剪和画像结果。",
        "- 当前流程已切换为视觉端主动推送：靠近检测通过后，多帧采样并聚合结果。",
        "- 当前推理前会将图像缩放到低分辨率，降低 N150 工控机负载。",
        "",
        "## 数据概览",
        "",
        f"- 处理图片数：{summary['total']}",
        f"- 成功读取图片数：{summary['valid']}",
        f"- 检测到人的图片数：{summary['person_present']}",
        f"- 检测到人脸的图片数：{summary['face_detected']}",
        f"- 平均置信度：{summary['avg_confidence']}",
        f"- 结果 CSV：[{csv_path.name}]({relative(csv_path, output_dir)})",
        "",
        "## 中间结果示例",
        "",
    ]

    for row in examples:
        lines.extend(
            [
                f"### 样例 {row['index']}",
                "",
                f"- 原图：[{Path(row['original']).name}]({relative(Path(row['original']), output_dir)})",
                f"- 人脸框：[{Path(row['face_box']).name}]({relative(Path(row['face_box']), output_dir)})",
                f"- 姿态骨架：[{Path(row['pose']).name}]({relative(Path(row['pose']), output_dir)})",
                f"- 人脸数量：{row['face_count']}",
                f"- 主用户选择：{row.get('primary_method', '')}, matched={row.get('primary_matched', '')}",
                f"- 画像：person={row['personPresent']}, height={row['heightCm']}, body={row['bodyType']}, age={row['ageRange']}, gender={row['gender']}, color={row['upperColor']}, confidence={row['confidence']}",
                "",
                f"![face box]({relative(Path(row['face_box']), output_dir)})",
                "",
                f"![pose]({relative(Path(row['pose']), output_dir)})",
                "",
            ]
        )

    lines.extend(
        [
            "## 风险与下周计划",
            "",
            "- 45度俯拍和20cm近距离会导致人体比例变化，身高厘米值仍需谨慎使用。",
            "- 性别和年龄在弱光、侧脸、遮挡条件下不稳定，建议作为弱字段或返回 unknown。",
            "- 下周重点：基于现场数据调整靠近阈值，评估是否需要轻量人体检测模型替代人脸面积判断。",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    args = parse_args()
    input_path = Path(args.input)

    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / f"weekly_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(input_path, args.limit)

    if not images:
        raise RuntimeError(f"no images found: {input_path}")

    rows = process_images(images, output_dir)
    csv_path = write_csv(rows, output_dir)
    report_path = write_report(args.title, input_path, output_dir, rows, csv_path)

    manifest = {
        "input": str(input_path),
        "output": str(output_dir),
        "imageCount": len(images),
        "csv": str(csv_path),
        "report": str(report_path),
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
