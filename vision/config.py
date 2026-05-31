import json
import os


def load_json_config():
    config_path = os.getenv("VISION_CONFIG_FILE", "config.json")

    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_json_config = load_json_config()


def get_config_value(key, env_key, default):
    """
    配置优先级：
    1. 环境变量
    2. config.json
    3. 默认值
    """
    if env_key in os.environ:
        value = os.getenv(env_key)

        if isinstance(default, bool):
            return str(value).lower() == "true"

        if isinstance(default, int):
            return int(value)

        if isinstance(default, float):
            return float(value)

        return value

    return _json_config.get(key, default)


class Settings:
    APP_NAME = "Vending Vision Module"
    APP_VERSION = "0.2.0"
    PROTOCOL = "vem.vision.v1"

    HOST = get_config_value("host", "VISION_HOST", "127.0.0.1")
    PORT = get_config_value("port", "VISION_PORT", 7892)
    DEFAULT_TIMEOUT_MS = get_config_value(
        "default_timeout_ms",
        "VISION_DEFAULT_TIMEOUT_MS",
        8000
    )
    PROFILE_PUSH_ENABLED = get_config_value(
        "profile_push_enabled",
        "VISION_PROFILE_PUSH_ENABLED",
        True
    )
    PROFILE_PUSH_INTERVAL_MS = get_config_value(
        "profile_push_interval_ms",
        "VISION_PROFILE_PUSH_INTERVAL_MS",
        1000
    )
    PROFILE_PUSH_COOLDOWN_MS = get_config_value(
        "profile_push_cooldown_ms",
        "VISION_PROFILE_PUSH_COOLDOWN_MS",
        8000
    )
    PROFILE_SAMPLE_COUNT = get_config_value(
        "profile_sample_count",
        "VISION_PROFILE_SAMPLE_COUNT",
        5
    )
    PROFILE_SAMPLE_INTERVAL_MS = get_config_value(
        "profile_sample_interval_ms",
        "VISION_PROFILE_SAMPLE_INTERVAL_MS",
        300
    )
    PROFILE_MIN_CONFIDENCE = get_config_value(
        "profile_min_confidence",
        "VISION_PROFILE_MIN_CONFIDENCE",
        0.45
    )
    PROFILE_MIN_VALID_FRAMES = get_config_value(
        "profile_min_valid_frames",
        "VISION_PROFILE_MIN_VALID_FRAMES",
        2
    )
    PROFILE_DETECTION_WIDTH = get_config_value(
        "profile_detection_width",
        "VISION_PROFILE_DETECTION_WIDTH",
        416
    )
    PROFILE_DETECTION_HEIGHT = get_config_value(
        "profile_detection_height",
        "VISION_PROFILE_DETECTION_HEIGHT",
        234
    )
    PROXIMITY_ENABLED = get_config_value(
        "proximity_enabled",
        "VISION_PROXIMITY_ENABLED",
        True
    )
    PROXIMITY_MONITOR_WIDTH = get_config_value(
        "proximity_monitor_width",
        "VISION_PROXIMITY_MONITOR_WIDTH",
        416
    )
    PROXIMITY_MONITOR_HEIGHT = get_config_value(
        "proximity_monitor_height",
        "VISION_PROXIMITY_MONITOR_HEIGHT",
        234
    )
    PROXIMITY_PRESENT_FACE_RATIO = get_config_value(
        "proximity_present_face_ratio",
        "VISION_PROXIMITY_PRESENT_FACE_RATIO",
        0.003
    )
    PROXIMITY_CLOSE_FACE_RATIO = get_config_value(
        "proximity_close_face_ratio",
        "VISION_PROXIMITY_CLOSE_FACE_RATIO",
        0.015
    )
    PROXIMITY_CLOSE_CONSECUTIVE_FRAMES = get_config_value(
        "proximity_close_consecutive_frames",
        "VISION_PROXIMITY_CLOSE_CONSECUTIVE_FRAMES",
        2
    )
    PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO = get_config_value(
        "primary_face_max_head_distance_ratio",
        "VISION_PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO",
        0.18
    )

    CAMERA_INDEX = get_config_value("camera_index", "VISION_CAMERA_INDEX", 0)
    CAMERA_BACKEND = get_config_value("camera_backend", "VISION_CAMERA_BACKEND", "dshow")
    CAMERA_WIDTH = get_config_value("camera_width", "VISION_CAMERA_WIDTH", 0)
    CAMERA_HEIGHT = get_config_value("camera_height", "VISION_CAMERA_HEIGHT", 0)
    CAMERA_FPS = get_config_value("camera_fps", "VISION_CAMERA_FPS", 0)
    CAMERA_FOURCC = get_config_value("camera_fourcc", "VISION_CAMERA_FOURCC", "")
    CAMERA_WARMUP_FRAMES = get_config_value(
        "camera_warmup_frames",
        "VISION_CAMERA_WARMUP_FRAMES",
        5
    )

    HEIGHT_SCALE = get_config_value("height_scale", "VISION_HEIGHT_SCALE", 100.0)
    HEIGHT_OFFSET = get_config_value("height_offset", "VISION_HEIGHT_OFFSET", 70.0)

    BODY_TYPE_THIN_THRESHOLD = get_config_value(
        "body_type_thin_threshold",
        "VISION_BODY_TYPE_THIN_THRESHOLD",
        0.2
    )
    BODY_TYPE_FAT_THRESHOLD = get_config_value(
        "body_type_fat_threshold",
        "VISION_BODY_TYPE_FAT_THRESHOLD",
        0.3
    )

    MOCK_SCENARIO = get_config_value("mock_scenario", "VISION_MOCK_SCENARIO", "off")
    MOCK_PUSH_INTERVAL_MS = get_config_value(
        "mock_push_interval_ms",
        "VISION_MOCK_PUSH_INTERVAL_MS",
        1000
    )

    FACE_DETECTOR_MODEL = os.getenv(
        "VISION_FACE_DETECTOR_MODEL",
        "models/face_detection/face_detection_yunet_2023mar.onnx"
    )

    FACE_SCORE_THRESHOLD = float(os.getenv("VISION_FACE_SCORE_THRESHOLD", "0.6"))
    FACE_NMS_THRESHOLD = float(os.getenv("VISION_FACE_NMS_THRESHOLD", "0.3"))
    FACE_TOP_K = int(os.getenv("VISION_FACE_TOP_K", "5000"))

    AGE_MODEL_PROTO = os.getenv(
        "VISION_AGE_MODEL_PROTO",
        "models/age_gender/age_deploy.prototxt"
    )
    AGE_MODEL_WEIGHTS = os.getenv(
        "VISION_AGE_MODEL_WEIGHTS",
        "models/age_gender/age_net.caffemodel"
    )
    GENDER_MODEL_PROTO = os.getenv(
        "VISION_GENDER_MODEL_PROTO",
        "models/age_gender/gender_deploy.prototxt"
    )
    GENDER_MODEL_WEIGHTS = os.getenv(
        "VISION_GENDER_MODEL_WEIGHTS",
        "models/age_gender/gender_net.caffemodel"
    )

    LOG_DIR = os.getenv("VISION_LOG_DIR", "logs")
    LOG_FILE = os.getenv("VISION_LOG_FILE", "logs/vision.log")

    DEBUG_OUTPUT_DIR = get_config_value(
        "debug_output_dir",
        "VISION_DEBUG_OUTPUT_DIR",
        "debug_outputs"
    )
    SAVE_DEBUG_IMAGES = get_config_value(
        "save_debug_images",
        "VISION_SAVE_DEBUG_IMAGES",
        True
    )
    MAX_DEBUG_IMAGES = get_config_value(
        "max_debug_images",
        "VISION_MAX_DEBUG_IMAGES",
        200
    )


settings = Settings()
