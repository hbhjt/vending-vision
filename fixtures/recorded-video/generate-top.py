"""确定性重生成 1080p top presence 夹具。

旧 320x320 top 夹具没有出处；把它作为已提交的源（sources/top-legacy-320.mp4），
逐帧无损语义地重采样到 1920x1080，保持“有人走近→离开后全黑”的 approach/departure
行为与生产 YOLO presence 检测语义。重采样只改变画幅，不改变检测输入内容
（人物检测仍统一缩放到 640 输入）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from common import FPS, TOP_FRAME_SIZE, TOP_LEGACY_SOURCE


def main() -> None:
    capture = cv2.VideoCapture(str(TOP_LEGACY_SOURCE))
    if not capture.isOpened():
        raise SystemExit(f"missing legacy top source: {TOP_LEGACY_SOURCE}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            cv2.resize(
                frame,
                TOP_FRAME_SIZE,
                interpolation=cv2.INTER_AREA,
            )
        )
    capture.release()
    if not frames:
        raise SystemExit("legacy top source has no frames")

    output = Path(__file__).with_name("top.mp4")
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, TOP_FRAME_SIZE
    )
    if not writer.isOpened():
        raise SystemExit("could not open recorded-video writer")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    print(f"{output.name} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
