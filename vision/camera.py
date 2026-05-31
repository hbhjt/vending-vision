import cv2

from vision.config import settings


def get_camera_backend(backend_name: str | None = None):
    name = (backend_name or settings.CAMERA_BACKEND or "dshow").lower()

    mapping = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    return mapping.get(name, cv2.CAP_DSHOW)


def apply_camera_settings(
    cap,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    width = settings.CAMERA_WIDTH if width is None else width
    height = settings.CAMERA_HEIGHT if height is None else height
    fps = settings.CAMERA_FPS if fps is None else fps
    fourcc = settings.CAMERA_FOURCC if fourcc is None else fourcc

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))

    if width and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def open_camera(
    camera_index: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    if camera_index is None:
        camera_index = settings.CAMERA_INDEX

    backend = get_camera_backend(backend_name)
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"camera unavailable, index={camera_index}, backend={backend_name or settings.CAMERA_BACKEND}"
        )

    apply_camera_settings(cap, width=width, height=height, fps=fps, fourcc=fourcc)
    return cap


def read_warmup_frame(cap, warmup_frames: int):
    image = None

    for _ in range(max(1, warmup_frames)):
        ret, frame = cap.read()
        if ret:
            image = frame

    if image is None:
        raise RuntimeError("camera opened but failed to read a valid frame")

    return image


def capture_image(
    camera_index: int | None = None,
    warmup_frames: int | None = None,
    backend_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    fourcc: str | None = None,
):
    if warmup_frames is None:
        warmup_frames = settings.CAMERA_WARMUP_FRAMES

    cap = open_camera(
        camera_index=camera_index,
        backend_name=backend_name,
        width=width,
        height=height,
        fps=fps,
        fourcc=fourcc,
    )

    try:
        return read_warmup_frame(cap, warmup_frames)
    finally:
        cap.release()


def get_configured_camera_status():
    cap = open_camera()

    try:
        image = read_warmup_frame(cap, settings.CAMERA_WARMUP_FRAMES)
        capture = describe_capture(cap)
        h, w = image.shape[:2]

        return {
            "ok": True,
            "index": settings.CAMERA_INDEX,
            "backend": settings.CAMERA_BACKEND,
            "requested": {
                "width": settings.CAMERA_WIDTH,
                "height": settings.CAMERA_HEIGHT,
                "fps": settings.CAMERA_FPS,
                "fourcc": settings.CAMERA_FOURCC,
            },
            "actual": capture,
            "frame": {
                "width": w,
                "height": h,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
            },
        }
    finally:
        cap.release()


def describe_capture(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()

    return {
        "width": width,
        "height": height,
        "fps": round(float(fps), 2),
        "fourcc": fourcc,
    }


def probe_cameras(max_index: int = 8, backend_name: str | None = None):
    results = []

    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, get_camera_backend(backend_name))
        opened = cap.isOpened()
        frame_ok = False
        shape = None
        capture = {}

        if opened:
            ret, frame = cap.read()
            frame_ok = bool(ret)
            capture = describe_capture(cap)

            if ret and frame is not None:
                shape = list(frame.shape)

        cap.release()

        results.append(
            {
                "index": index,
                "opened": opened,
                "frameOk": frame_ok,
                "frameShape": shape,
                "capture": capture,
            }
        )

    return results
