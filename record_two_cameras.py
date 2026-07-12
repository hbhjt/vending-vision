import cv2
import time
import threading
from datetime import datetime


# ============================================================
# 1. 摄像头 index 映射
# ============================================================

CAMERA_MAP = {
    "builtin": 1,  # 电脑自带摄像头
    "left": 0,     # 左边外接摄像头
    "right": 2,    # 右边外接摄像头
}

# 实际录制哪两个摄像头
RECORD_CAMERAS = ["left", "right"]

# 如果左右反了，只需要把上面的 CAMERA_MAP 改成：
# CAMERA_MAP = {
#     "builtin": 0,
#     "left": 2,
#     "right": 1,
# }


# ============================================================
# 2. 录制参数
# ============================================================

DURATION_SEC = 60

# 建议先用 15，更稳定
# 如果你的电脑和摄像头性能足够，可以改成 30
TARGET_FPS = 15

CAM_WIDTH = 1280
CAM_HEIGHT = 720

SHOW_PREVIEW = True


class CameraWorker:
    def __init__(self, logical_name, index, width, height, fps):
        self.logical_name = logical_name
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.last_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头：{self.logical_name}, index={self.index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 尽量减少缓存延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = self.cap.read()

        if not ret:
            self.cap.release()
            raise RuntimeError(
                f"摄像头可以打开，但读取失败：{self.logical_name}, index={self.index}"
            )

        h, w = frame.shape[:2]

        self.width = w
        self.height = h

        frame = self.add_label(frame)

        with self.lock:
            self.last_frame = frame

        print(
            f"[Camera OK] {self.logical_name}: "
            f"index={self.index}, size={self.width}x{self.height}"
        )

    def add_label(self, frame):
        frame = frame.copy()

        cv2.putText(
            frame,
            f"{self.logical_name} | index={self.index}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3,
        )

        return frame

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                frame = self.add_label(frame)

                with self.lock:
                    self.last_frame = frame

            time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None

            return self.last_frame.copy()

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("当前摄像头映射：")
    for name, index in CAMERA_MAP.items():
        print(f"  {name}: index={index}")

    print("\n本次录制摄像头：")
    for name in RECORD_CAMERAS:
        print(f"  {name}: index={CAMERA_MAP[name]}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    workers = {}
    writers = {}
    output_paths = {}
    written_frames = {}

    try:
        # ====================================================
        # 打开摄像头和视频写入器
        # ====================================================

        for logical_name in RECORD_CAMERAS:
            index = CAMERA_MAP[logical_name]

            worker = CameraWorker(
                logical_name=logical_name,
                index=index,
                width=CAM_WIDTH,
                height=CAM_HEIGHT,
                fps=TARGET_FPS,
            )

            worker.open()

            output_path = f"camera_{logical_name}_index{index}_{timestamp}.mp4"

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                TARGET_FPS,
                (worker.width, worker.height),
            )

            if not writer.isOpened():
                raise RuntimeError(f"视频写入器打开失败：{output_path}")

            workers[logical_name] = worker
            writers[logical_name] = writer
            output_paths[logical_name] = output_path
            written_frames[logical_name] = 0

        # 启动摄像头采集线程
        for worker in workers.values():
            worker.start()

        print("\n开始录制...")
        print(f"计划录制时长：{DURATION_SEC} 秒")
        print(f"目标输出 FPS：{TARGET_FPS}")
        print("按 q 可以提前停止录制\n")

        start_time = time.time()
        total_target_frames = int(DURATION_SEC * TARGET_FPS)

        # ====================================================
        # 按固定 FPS 写入视频
        # ====================================================

        for frame_id in range(total_target_frames):
            target_time = start_time + frame_id / TARGET_FPS

            now = time.time()
            if now < target_time:
                time.sleep(target_time - now)

            elapsed = time.time() - start_time

            for logical_name in RECORD_CAMERAS:
                frame = workers[logical_name].get_frame()

                if frame is None:
                    continue

                writers[logical_name].write(frame)
                written_frames[logical_name] += 1

                if SHOW_PREVIEW:
                    index = CAMERA_MAP[logical_name]
                    cv2.imshow(f"{logical_name} | index={index}", frame)

            if SHOW_PREVIEW:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("手动停止录制")
                    break

            if frame_id % (TARGET_FPS * 5) == 0 and frame_id > 0:
                print(f"已录制 {elapsed:.1f}s / {DURATION_SEC}s")

    finally:
        for writer in writers.values():
            writer.release()

        for worker in workers.values():
            worker.stop()

        cv2.destroyAllWindows()

    real_duration = time.time() - start_time

    print("\n录制完成。")
    print(f"真实运行时长：{real_duration:.2f} 秒")
    print(f"目标输出 FPS：{TARGET_FPS}")

    for logical_name in RECORD_CAMERAS:
        path = output_paths[logical_name]
        frames = written_frames[logical_name]
        video_duration = frames / TARGET_FPS

        print(
            f"{logical_name}: {path}, "
            f"写入帧数={frames}, "
            f"理论视频时长={video_duration:.2f}s"
        )


if __name__ == "__main__":
    main()