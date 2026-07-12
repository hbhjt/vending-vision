from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a synchronized stability report from a video dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default="reports/video_dataset/test01",
        help="Directory produced by build_video_dataset.py.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percent(numerator, denominator):
    if not denominator:
        return 0.0
    return round(numerator / denominator * 100.0, 1)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(float(ordered[index]), 4)


def state_runs(rows, value_getter):
    if not rows:
        return []
    runs = []
    start = 0
    value = value_getter(rows[0])
    for index in range(1, len(rows)):
        current = value_getter(rows[index])
        if current == value:
            continue
        runs.append(make_run(rows, start, index - 1, value))
        start = index
        value = current
    runs.append(make_run(rows, start, len(rows) - 1, value))
    return runs


def make_run(rows, start, end, value):
    start_ms = as_int(rows[start].get("timestampMs"))
    end_ms = as_int(rows[end].get("timestampMs"))
    if end + 1 < len(rows):
        end_ms = as_int(rows[end + 1].get("timestampMs"))
    elif end > start:
        end_ms += as_int(rows[end].get("timestampMs")) - as_int(
            rows[end - 1].get("timestampMs")
        )
    else:
        end_ms += 333
    return {
        "value": value,
        "startMs": start_ms,
        "endMs": end_ms,
        "durationMs": max(end_ms - start_ms, 0),
        "sampleCount": end - start + 1,
    }


def nearest_row(rows, timestamps, timestamp_ms, tolerance_ms=500):
    index = bisect.bisect_left(timestamps, timestamp_ms)
    candidates = []
    if index < len(rows):
        candidates.append(rows[index])
    if index > 0:
        candidates.append(rows[index - 1])
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(as_int(row["timestampMs"]) - timestamp_ms))
    if abs(as_int(nearest["timestampMs"]) - timestamp_ms) > tolerance_ms:
        return None
    return nearest


def relative_image_path(dataset_dir: Path, value):
    if not value:
        return ""
    path = Path(value)
    try:
        return path.resolve().relative_to(dataset_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def field_summary(rows, field, missing_values):
    values = [row.get(field, "") for row in rows]
    available = [value for value in values if value not in missing_values]
    counts = Counter(available)
    dominant, dominant_count = counts.most_common(1)[0] if counts else (None, 0)
    adjacent = 0
    flips = 0
    previous = None
    previous_ms = None
    for row in rows:
        value = row.get(field, "")
        timestamp_ms = as_int(row.get("timestampMs"))
        if value in missing_values:
            continue
        if previous is not None and timestamp_ms - previous_ms <= 1000:
            adjacent += 1
            if value != previous:
                flips += 1
        previous = value
        previous_ms = timestamp_ms
    return {
        "availableCount": len(available),
        "availabilityPct": percent(len(available), len(rows)),
        "dominantValue": dominant,
        "dominantPct": percent(dominant_count, len(available)),
        "adjacentComparisons": adjacent,
        "flipCount": flips,
        "flipPct": percent(flips, adjacent),
        "counts": dict(counts),
    }


def write_csv(path: Path, rows, fieldnames=None, encoding="utf-8-sig"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding=encoding) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_timeline_svg(paired_rows, duration_ms):
    width = 1000
    left = 115
    plot_width = 850
    row_height = 25
    rows = [
        ("顶部占用", "topOccupancy"),
        ("顶部有人", "topPresent"),
        ("正面人体证据", "frontEvidence"),
        ("正面有效画像", "frontValid"),
        ("端到端可用", "coordinatedUsable"),
    ]
    colors = {
        "none": "#d1d5db",
        "unknown": "#f59e0b",
        "single": "#22c55e",
        "multiple": "#ef4444",
        True: "#2563eb",
        False: "#e5e7eb",
    }
    rectangles = []
    for row_index, (label, field) in enumerate(rows):
        y = 28 + row_index * row_height
        rectangles.append(
            f'<text x="5" y="{y + 14}" font-size="13" fill="#334155">{label}</text>'
        )
        for index, item in enumerate(paired_rows):
            start_ms = as_int(item["timestampMs"])
            next_ms = (
                as_int(paired_rows[index + 1]["timestampMs"])
                if index + 1 < len(paired_rows)
                else duration_ms
            )
            x = left + start_ms / duration_ms * plot_width
            item_width = max((next_ms - start_ms) / duration_ms * plot_width, 1.0)
            value = item[field]
            color = colors.get(value, "#94a3b8")
            title = html.escape(
                f"{start_ms / 1000:.1f}s | {label}: {value}", quote=True
            )
            rectangles.append(
                f'<rect x="{x:.2f}" y="{y}" width="{item_width + 0.3:.2f}" '
                f'height="18" fill="{color}"><title>{title}</title></rect>'
            )
    for second in range(0, int(duration_ms / 1000) + 1, 10):
        x = left + second * 1000 / duration_ms * plot_width
        rectangles.append(
            f'<line x1="{x:.2f}" y1="20" x2="{x:.2f}" y2="155" '
            'stroke="#94a3b8" stroke-width="0.5" opacity="0.5" />'
        )
        rectangles.append(
            f'<text x="{x - 7:.2f}" y="170" font-size="11" fill="#64748b">{second}s</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} 180" role="img" '
        'aria-label="双摄像头稳定性时间轴">'
        + "".join(rectangles)
        + "</svg>"
    )


def report_html(dataset_dir: Path, summary, paired_rows, review_rows):
    metrics = summary["metrics"]
    verdict = summary["verdict"]
    duration_ms = summary["durationMs"]
    timeline = make_timeline_svg(paired_rows, duration_ms)
    bad_examples = [
        row
        for row in review_rows
        if row["frontEvidence"] == "True" and row["topOccupancy"] != "single"
    ][:8]
    cards = [
        ("稳定性评分", f'{summary["score"]}/100'),
        ("顶部来人覆盖", f'{metrics["topPresentCoveragePct"]}%'),
        ("顶部单人门控覆盖", f'{metrics["topSingleCoveragePct"]}%'),
        ("正面画像有效率", f'{metrics["frontValidRatePct"]}%'),
        ("端到端可用率", f'{metrics["endToEndCoveragePct"]}%'),
        ("顶部状态短抖动", str(metrics["topShortRunCount"])),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )
    recommendation_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in summary["recommendations"]
    )
    example_html = "".join(
        '<article class="example">'
        f'<div><strong>{as_int(row["timestampMs"]) / 1000:.1f}s</strong> — '
        f'顶部={html.escape(row["topOccupancy"])}，正面有效={row["frontValid"]}</div>'
        '<div class="shots">'
        f'<a href="{html.escape(row["topImage"], quote=True)}"><img src="{html.escape(row["topImage"], quote=True)}" alt="顶部帧"></a>'
        f'<a href="{html.escape(row["frontImage"], quote=True)}"><img src="{html.escape(row["frontImage"], quote=True)}" alt="正面帧"></a>'
        "</div></article>"
        for row in bad_examples
    )
    field_rows = "".join(
        "<tr>"
        f"<td>{html.escape(field)}</td>"
        f"<td>{data['availabilityPct']}%</td>"
        f"<td>{html.escape(str(data['dominantValue']))}</td>"
        f"<td>{data['dominantPct']}%</td>"
        f"<td>{data['flipPct']}%</td>"
        "</tr>"
        for field, data in summary["frontFields"].items()
    )
    summary_json = html.escape(
        json.dumps(summary, ensure_ascii=False, indent=2), quote=False
    )
    manual = summary["manualReview"]
    manual_html = (
        '<p class="muted">尚未填写人工标签。自动稳定性评分只能用于快速筛查。</p>'
        if manual["reviewedRowCount"] == 0
        else (
            f'<p>已复核 {manual["reviewedRowCount"]} 行；'
            f'顶部单人召回率 {manual["topSingleRecallPct"]}%；'
            f'正面可用召回率 {manual["frontUsableRecallPct"]}%；'
            f'端到端召回率 {manual["endToEndRecallPct"]}%。</p>'
        )
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>双摄像头稳定性报告</title>
<style>
body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}
main{{max-width:1180px;margin:auto;padding:24px}} h1,h2{{margin:.3em 0}} .muted{{color:#64748b}}
.verdict{{padding:16px 20px;border-radius:12px;background:{verdict['color']};color:white;margin:16px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
.card,.panel{{background:white;border-radius:12px;padding:16px;box-shadow:0 1px 3px #0001}}
.label{{color:#64748b;font-size:14px}} .value{{font-size:28px;font-weight:700;margin-top:5px}}
.panel{{margin-top:16px}} svg{{width:100%;height:auto}} table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #e2e8f0}} .shots{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}
.shots img{{width:100%;height:220px;object-fit:contain;background:#0f172a;border-radius:8px}}
.example{{margin:14px 0;padding-bottom:14px;border-bottom:1px solid #e2e8f0}}
code{{background:#e2e8f0;padding:2px 5px;border-radius:4px}} details pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style>
</head>
<body><main>
<h1>双摄像头离线稳定性报告</h1>
<p class="muted">数据集：{html.escape(str(dataset_dir))} · 时长 {duration_ms / 1000:.1f}s · 自动标注，需要人工复核</p>
<div class="verdict"><strong>{html.escape(verdict['label'])}</strong><br>{html.escape(verdict['reason'])}</div>
<section class="cards">{card_html}</section>
<section class="panel"><h2>同步时间轴</h2><p class="muted">绿色=单人，橙色=未知，灰色=无人；蓝色表示对应条件成立。鼠标悬停可看时间。</p>{timeline}</section>
<section class="panel"><h2>主要结论与建议</h2><ol>{recommendation_html}</ol></section>
<section class="panel"><h2>画像字段稳定性</h2><table><thead><tr><th>字段</th><th>可用率</th><th>主要值</th><th>主要值占比</th><th>相邻跳变率</th></tr></thead><tbody>{field_rows}</tbody></table></section>
<section class="panel"><h2>顶部门控漏失示例</h2><p class="muted">下列时刻正面摄像头检测到人体证据，但顶部没有确认单人。点击图片可查看原帧。</p>{example_html or '<p>未发现此类样本。</p>'}</section>
<section class="panel"><h2>人工复核</h2>{manual_html}<p>使用 Excel 打开 <code>review_labels.csv</code>，填写 <code>manualPersonPresent</code>、<code>manualOccupancy</code>、<code>manualFrontUsable</code> 和 <code>notes</code>，保存后重新运行分析脚本即可得到人工标签召回率。自动预测不能代替真实标签。</p></section>
<details class="panel"><summary>完整 JSON 指标</summary><pre>{summary_json}</pre></details>
</main></body></html>"""


def analyze(dataset_dir: Path):
    top_rows = read_csv(dataset_dir / "top_occupancy_auto_labels.csv")
    front_rows = read_csv(dataset_dir / "front_profile_auto_labels.csv")
    if not top_rows or not front_rows:
        raise RuntimeError("dataset CSV files are empty")
    top_rows.sort(key=lambda row: as_int(row["timestampMs"]))
    front_rows.sort(key=lambda row: as_int(row["timestampMs"]))
    front_timestamps = [as_int(row["timestampMs"]) for row in front_rows]
    duration_ms = max(
        as_int(top_rows[-1]["timestampMs"]),
        as_int(front_rows[-1]["timestampMs"]),
    ) + 333

    review_path = dataset_dir / "review_labels.csv"
    existing_review = {}
    if review_path.exists():
        for row in read_csv(review_path):
            existing_review[as_int(row.get("timestampMs"))] = row

    paired_rows = []
    review_rows = []
    for top in top_rows:
        timestamp_ms = as_int(top["timestampMs"])
        front = nearest_row(front_rows, front_timestamps, timestamp_ms)
        if front is None:
            continue
        front_evidence = as_bool(front["personDetected"]) or as_bool(
            front["faceDetected"]
        )
        front_valid = as_bool(front["valid"])
        top_single = top["autoOccupancy"] == "single"
        paired = {
            "timestampMs": timestamp_ms,
            "topOccupancy": top["autoOccupancy"],
            "topPresent": as_bool(top["present"]),
            "frontEvidence": front_evidence,
            "frontValid": front_valid,
            "coordinatedUsable": top_single and front_valid,
        }
        paired_rows.append(paired)
        previous_review = existing_review.get(timestamp_ms, {})
        review_rows.append(
            {
                **{key: str(value) for key, value in paired.items()},
                "topRawCount": top["rawCount"],
                "topLargestPersonRatio": top["largestPersonRatio"],
                "frontConfidence": front["confidence"],
                "frontBodyType": front["bodyType"],
                "frontUpperColor": front["upperColor"],
                "topImage": relative_image_path(dataset_dir, top["imagePath"]),
                "frontImage": relative_image_path(dataset_dir, front["imagePath"]),
                "manualPersonPresent": previous_review.get(
                    "manualPersonPresent", ""
                ),
                "manualOccupancy": previous_review.get("manualOccupancy", ""),
                "manualFrontUsable": previous_review.get(
                    "manualFrontUsable", ""
                ),
                "notes": previous_review.get("notes", ""),
            }
        )

    evidence_rows = [row for row in paired_rows if row["frontEvidence"]]
    evidence_count = len(evidence_rows)
    top_present_count = sum(row["topPresent"] for row in evidence_rows)
    top_single_count = sum(row["topOccupancy"] == "single" for row in evidence_rows)
    front_valid_count = sum(row["frontValid"] for row in evidence_rows)
    coordinated_count = sum(row["coordinatedUsable"] for row in evidence_rows)
    top_runs = state_runs(paired_rows, lambda row: row["topOccupancy"])
    short_runs = [run for run in top_runs if run["durationMs"] < 1000]
    single_runs = [run for run in top_runs if run["value"] == "single"]
    confidence_values = [as_float(row["confidence"]) for row in front_rows]
    valid_front_rows = [row for row in front_rows if as_bool(row["valid"])]

    metrics = {
        "pairedSampleCount": len(paired_rows),
        "frontEvidenceCount": evidence_count,
        "topPresentDuringEvidenceCount": top_present_count,
        "topSingleDuringEvidenceCount": top_single_count,
        "frontValidDuringEvidenceCount": front_valid_count,
        "coordinatedUsableCount": coordinated_count,
        "topPresentCoveragePct": percent(top_present_count, evidence_count),
        "topSingleCoveragePct": percent(top_single_count, evidence_count),
        "frontValidRatePct": percent(front_valid_count, evidence_count),
        "endToEndCoveragePct": percent(coordinated_count, evidence_count),
        "topTransitionCount": max(len(top_runs) - 1, 0),
        "topShortRunCount": len(short_runs),
        "topSingleLongestMs": max(
            [run["durationMs"] for run in single_runs], default=0
        ),
        "frontConfidence": {
            "mean": round(statistics.mean(confidence_values), 4),
            "p50": percentile(confidence_values, 0.5),
            "p90": percentile(confidence_values, 0.9),
        },
    }
    score = round(
        metrics["topPresentCoveragePct"] * 0.2
        + metrics["topSingleCoveragePct"] * 0.5
        + metrics["frontValidRatePct"] * 0.3
    )
    if score >= 75:
        verdict = {
            "label": "稳定",
            "reason": "双摄协同覆盖率达到当前离线验收目标。",
            "color": "#15803d",
        }
    elif score >= 45:
        verdict = {
            "label": "需要调优",
            "reason": "能形成部分可用结果，但存在明显门控或画像漏失。",
            "color": "#b45309",
        }
    else:
        verdict = {
            "label": "当前录像下不稳定",
            "reason": "双摄端到端门控覆盖不足，实际运行可能无法产生画像推荐。",
            "color": "#b91c1c",
        }

    recommendations = []
    if evidence_count and metrics["topSingleCoveragePct"] < 60:
        recommendations.append(
            "最高优先级：顶部人体 YOLO 对俯视角适配不足。正面确认有人时，顶部虽然经常 present=true，但 occupancy 未稳定为 single，严格门控会阻止画像。"
        )
    if evidence_count and metrics["frontValidRatePct"] < 50:
        recommendations.append(
            "正面画像有效帧偏少。建议优先改善人物进入正面 ROI 的位置和停留时间，再评估人体存在字段与部分画像判定。"
        )
    if metrics["topShortRunCount"]:
        recommendations.append(
            "顶部状态仍存在不足 1 秒的短区间；当前已使用两帧离开确认，如现场欢迎语或会话仍发生抖动，可将无人确认增加到三帧。"
        )
    if metrics["endToEndCoveragePct"] < 50:
        recommendations.append(
            "本轮先不要以字段准确率作为主要验收项；先把“顶部单人门控 + 正面两帧有效”端到端覆盖率提升到 70% 以上。"
        )
    recommendations.append(
        "在 review_labels.csv 中人工标记真实有人/单人/正面可用后重新统计，自动标签仅用于快速发现问题，不能作为最终准确率真值。"
    )

    front_fields = {
        "bodyType": field_summary(valid_front_rows, "bodyType", {"", "unknown"}),
        "upperColor": field_summary(valid_front_rows, "upperColor", {"", "unknown"}),
        "ageRange": field_summary(valid_front_rows, "ageRange", {"", "unknown"}),
        "gender": field_summary(valid_front_rows, "gender", {"", "unknown"}),
    }
    manually_reviewed = [
        row
        for row in review_rows
        if row["manualOccupancy"].strip()
        or row["manualFrontUsable"].strip()
        or row["manualPersonPresent"].strip()
    ]
    manual_single = [
        row for row in review_rows if row["manualOccupancy"].strip() == "single"
    ]
    manual_front_usable = [
        row for row in review_rows if as_bool(row["manualFrontUsable"])
    ]
    manual_end_to_end = [
        row
        for row in review_rows
        if row["manualOccupancy"].strip() == "single"
        and as_bool(row["manualFrontUsable"])
    ]
    manual_review = {
        "reviewedRowCount": len(manually_reviewed),
        "manualSingleCount": len(manual_single),
        "manualFrontUsableCount": len(manual_front_usable),
        "manualEndToEndCount": len(manual_end_to_end),
        "topSingleRecallPct": percent(
            sum(row["topOccupancy"] == "single" for row in manual_single),
            len(manual_single),
        ),
        "frontUsableRecallPct": percent(
            sum(as_bool(row["frontValid"]) for row in manual_front_usable),
            len(manual_front_usable),
        ),
        "endToEndRecallPct": percent(
            sum(as_bool(row["coordinatedUsable"]) for row in manual_end_to_end),
            len(manual_end_to_end),
        ),
    }
    summary = {
        "datasetType": "auto_labeled_review_required",
        "durationMs": duration_ms,
        "score": score,
        "verdict": verdict,
        "metrics": metrics,
        "frontFields": front_fields,
        "manualReview": manual_review,
        "topRuns": top_runs,
        "recommendations": recommendations,
    }
    write_csv(dataset_dir / "stability_timeline.csv", review_rows)
    write_csv(review_path, review_rows)
    (dataset_dir / "stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (dataset_dir / "stability_report.html").write_text(
        report_html(dataset_dir, summary, paired_rows, review_rows), encoding="utf-8"
    )
    return summary


def main():
    args = parse_args()
    dataset_dir = (ROOT / args.dataset_dir).resolve()
    summary = analyze(dataset_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"REPORT={dataset_dir / 'stability_report.html'}")


if __name__ == "__main__":
    main()
