import argparse
import json
from pathlib import Path

import cv2

from vision.camera import capture_image, probe_cameras


def parse_args():
    parser = argparse.ArgumentParser(description="Camera probe and capture tool")
    parser.add_argument("--probe", action="store_true", help="probe camera indexes")
    parser.add_argument("--max-index", type=int, default=8, help="max camera index to probe")
    parser.add_argument("--index", type=int, default=None, help="camera index")
    parser.add_argument("--backend", default=None, help="camera backend: dshow, msmf, any")
    parser.add_argument("--width", type=int, default=None, help="requested frame width")
    parser.add_argument("--height", type=int, default=None, help="requested frame height")
    parser.add_argument("--fps", type=int, default=None, help="requested FPS")
    parser.add_argument("--fourcc", default=None, help="requested FourCC, e.g. MJPG or YUY2")
    parser.add_argument("--warmup", type=int, default=None, help="warmup frames")
    parser.add_argument("--output", default="output_camera.jpg", help="output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.probe:
        results = probe_cameras(max_index=args.max_index, backend_name=args.backend)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    image = capture_image(
        camera_index=args.index,
        backend_name=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        warmup_frames=args.warmup,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)

    if not ok:
        raise RuntimeError(f"failed to save image: {output_path}")

    h, w = image.shape[:2]
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "width": w,
                "height": h,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
