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


def get_json_config_value(key, env_key, default):
    if env_key in os.environ:
        try:
            return json.loads(os.getenv(env_key) or "")
        except Exception:
            return default

    value = _json_config.get(key, default)
    return value if isinstance(value, type(default)) else default


def build_camera_config(cameras, role, fallback):
    config = dict(fallback)
    role_config = cameras.get(role, {}) if isinstance(cameras, dict) else {}

    if isinstance(role_config, dict):
        for key, value in role_config.items():
            if value is not None:
                config[key] = value

    config.setdefault("role", role)
    return config


def bool_config_value(value, default=False):
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    return str(value).lower() == "true"


class Settings:
    APP_NAME = "Vending Vision Module"
    APP_VERSION = "0.2.0"
    PROTOCOL = "vem.vision.v1"
    POSE_ENABLE_SEGMENTATION = get_config_value(
        "pose_enable_segmentation",
        "VISION_POSE_ENABLE_SEGMENTATION",
        True
    )

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
        300
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
    PROFILE_BODY_BUFFER_MAX_FRAMES = get_config_value(
        "profile_body_buffer_max_frames",
        "VISION_PROFILE_BODY_BUFFER_MAX_FRAMES",
        8
    )
    PROFILE_BODY_BUFFER_TTL_MS = get_config_value(
        "profile_body_buffer_ttl_ms",
        "VISION_PROFILE_BODY_BUFFER_TTL_MS",
        4000
    )
    PROFILE_TRACK_ENABLED = get_config_value(
        "profile_track_enabled",
        "VISION_PROFILE_TRACK_ENABLED",
        True
    )
    PROFILE_TRACK_MAX_MISSING_FRAMES = get_config_value(
        "profile_track_max_missing_frames",
        "VISION_PROFILE_TRACK_MAX_MISSING_FRAMES",
        2
    )
    PROFILE_TRACK_MAX_CENTER_SHIFT = get_config_value(
        "profile_track_max_center_shift",
        "VISION_PROFILE_TRACK_MAX_CENTER_SHIFT",
        0.35
    )
    PROFILE_TRACK_MIN_MATCH_SCORE = get_config_value(
        "profile_track_min_match_score",
        "VISION_PROFILE_TRACK_MIN_MATCH_SCORE",
        0.45
    )
    PROFILE_OCCUPANCY_GATE_ENABLED = get_config_value(
        "profile_occupancy_gate_enabled",
        "VISION_PROFILE_OCCUPANCY_GATE_ENABLED",
        True
    )
    PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES = get_config_value(
        "profile_occupancy_reset_absent_frames",
        "VISION_PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES",
        6
    )
    PROFILE_MIN_CONFIDENCE = get_config_value(
        "profile_min_confidence",
        "VISION_PROFILE_MIN_CONFIDENCE",
        0.45
    )
    PROFILE_MIN_VALID_FRAMES = get_config_value(
        "profile_min_valid_frames",
        "VISION_PROFILE_MIN_VALID_FRAMES",
        1
    )
    PROFILE_FACE_VOTE_ENABLED = get_config_value(
        "profile_face_vote_enabled",
        "VISION_PROFILE_FACE_VOTE_ENABLED",
        True
    )
    PROFILE_FACE_VOTE_SAMPLE_COUNT = get_config_value(
        "profile_face_vote_sample_count",
        "VISION_PROFILE_FACE_VOTE_SAMPLE_COUNT",
        3
    )
    PROFILE_FACE_VOTE_INTERVAL_MS = get_config_value(
        "profile_face_vote_interval_ms",
        "VISION_PROFILE_FACE_VOTE_INTERVAL_MS",
        120
    )
    PROFILE_FACE_VOTE_MIN_SHARPNESS = get_config_value(
        "profile_face_vote_min_sharpness",
        "VISION_PROFILE_FACE_VOTE_MIN_SHARPNESS",
        30.0
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
    PROXIMITY_PERSON_ENABLED = get_config_value(
        "proximity_person_enabled",
        "VISION_PROXIMITY_PERSON_ENABLED",
        True
    )
    PROXIMITY_PRESENT_PERSON_RATIO = get_config_value(
        "proximity_present_person_ratio",
        "VISION_PROXIMITY_PRESENT_PERSON_RATIO",
        0.04
    )
    PROXIMITY_CLOSE_PERSON_RATIO = get_config_value(
        "proximity_close_person_ratio",
        "VISION_PROXIMITY_CLOSE_PERSON_RATIO",
        0.18
    )
    PROXIMITY_BODY_ENABLED = get_config_value(
        "proximity_body_enabled",
        "VISION_PROXIMITY_BODY_ENABLED",
        True
    )
    PROXIMITY_BODY_MIN_VISIBILITY = get_config_value(
        "proximity_body_min_visibility",
        "VISION_PROXIMITY_BODY_MIN_VISIBILITY",
        0.45
    )
    PROXIMITY_BODY_MIN_VISIBLE_POINTS = get_config_value(
        "proximity_body_min_visible_points",
        "VISION_PROXIMITY_BODY_MIN_VISIBLE_POINTS",
        6
    )
    PROXIMITY_PRESENT_BODY_RATIO = get_config_value(
        "proximity_present_body_ratio",
        "VISION_PROXIMITY_PRESENT_BODY_RATIO",
        0.08
    )
    PROXIMITY_CLOSE_BODY_RATIO = get_config_value(
        "proximity_close_body_ratio",
        "VISION_PROXIMITY_CLOSE_BODY_RATIO",
        0.22
    )
    PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO = get_config_value(
        "primary_face_max_head_distance_ratio",
        "VISION_PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO",
        0.18
    )
    AMBIENT_LIGHT_DARK_LUMA = get_config_value(
        "ambient_light_dark_luma",
        "VISION_AMBIENT_LIGHT_DARK_LUMA",
        50.0
    )
    AMBIENT_LIGHT_DIM_LUMA = get_config_value(
        "ambient_light_dim_luma",
        "VISION_AMBIENT_LIGHT_DIM_LUMA",
        110.0
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
        1
    )
    CAMERA_KEEP_OPEN = get_config_value(
        "camera_keep_open",
        "VISION_CAMERA_KEEP_OPEN",
        True
    )
    CAMERA_READ_RETRY_COUNT = get_config_value(
        "camera_read_retry_count",
        "VISION_CAMERA_READ_RETRY_COUNT",
        2
    )
    CAMERA_RECONNECT_DELAY_MS = get_config_value(
        "camera_reconnect_delay_ms",
        "VISION_CAMERA_RECONNECT_DELAY_MS",
        300
    )
    CAMERAS = get_json_config_value("cameras", "VISION_CAMERAS", {})
    LEGACY_CAMERA_CONFIG = {
        "index": CAMERA_INDEX,
        "backend": CAMERA_BACKEND,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
        "fourcc": CAMERA_FOURCC,
        "keep_open": CAMERA_KEEP_OPEN,
    }
    TOP_CAMERA_CONFIG = build_camera_config(CAMERAS, "top", LEGACY_CAMERA_CONFIG)
    FRONT_CAMERA_CONFIG = build_camera_config(CAMERAS, "front", LEGACY_CAMERA_CONFIG)
    TOP_CAMERA_INDEX = int(TOP_CAMERA_CONFIG.get("index", CAMERA_INDEX))
    TOP_CAMERA_BACKEND = TOP_CAMERA_CONFIG.get("backend", CAMERA_BACKEND)
    TOP_CAMERA_WIDTH = int(TOP_CAMERA_CONFIG.get("width", CAMERA_WIDTH) or 0)
    TOP_CAMERA_HEIGHT = int(TOP_CAMERA_CONFIG.get("height", CAMERA_HEIGHT) or 0)
    TOP_CAMERA_FPS = int(TOP_CAMERA_CONFIG.get("fps", CAMERA_FPS) or 0)
    TOP_CAMERA_FOURCC = TOP_CAMERA_CONFIG.get("fourcc", CAMERA_FOURCC)
    TOP_CAMERA_KEEP_OPEN = bool_config_value(
        TOP_CAMERA_CONFIG.get("keep_open"),
        CAMERA_KEEP_OPEN
    )
    FRONT_CAMERA_INDEX = int(FRONT_CAMERA_CONFIG.get("index", CAMERA_INDEX))
    FRONT_CAMERA_BACKEND = FRONT_CAMERA_CONFIG.get("backend", CAMERA_BACKEND)
    FRONT_CAMERA_WIDTH = int(FRONT_CAMERA_CONFIG.get("width", CAMERA_WIDTH) or 0)
    FRONT_CAMERA_HEIGHT = int(FRONT_CAMERA_CONFIG.get("height", CAMERA_HEIGHT) or 0)
    FRONT_CAMERA_FPS = int(FRONT_CAMERA_CONFIG.get("fps", CAMERA_FPS) or 0)
    FRONT_CAMERA_FOURCC = FRONT_CAMERA_CONFIG.get("fourcc", CAMERA_FOURCC)
    FRONT_CAMERA_KEEP_OPEN = bool_config_value(
        FRONT_CAMERA_CONFIG.get("keep_open"),
        False
    )
    FRONT_CAMERA_PROFILE_MAX_WAIT_MS = get_config_value(
        "front_camera_profile_max_wait_ms",
        "VISION_FRONT_CAMERA_PROFILE_MAX_WAIT_MS",
        3000
    )
    FRONT_CAMERA_PROFILE_SAMPLE_COUNT = get_config_value(
        "front_camera_profile_sample_count",
        "VISION_FRONT_CAMERA_PROFILE_SAMPLE_COUNT",
        6
    )
    FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS = get_config_value(
        "front_camera_profile_sample_interval_ms",
        "VISION_FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS",
        250
    )
    FRONT_CAMERA_OWNER_TIMEOUT_MS = get_config_value(
        "front_camera_owner_timeout_ms",
        "VISION_FRONT_CAMERA_OWNER_TIMEOUT_MS",
        120000
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
    UPPER_BODY_TYPE_THIN_THRESHOLD = get_config_value(
        "upper_body_type_thin_threshold",
        "VISION_UPPER_BODY_TYPE_THIN_THRESHOLD",
        0.65
    )
    UPPER_BODY_TYPE_FAT_THRESHOLD = get_config_value(
        "upper_body_type_fat_threshold",
        "VISION_UPPER_BODY_TYPE_FAT_THRESHOLD",
        0.9
    )
    BODY_MASK_ENABLED = get_config_value(
        "body_mask_enabled",
        "VISION_BODY_MASK_ENABLED",
        True
    )
    BODY_MASK_THRESHOLD = get_config_value(
        "body_mask_threshold",
        "VISION_BODY_MASK_THRESHOLD",
        0.55
    )
    BODY_MASK_MIN_AREA_RATIO = get_config_value(
        "body_mask_min_area_ratio",
        "VISION_BODY_MASK_MIN_AREA_RATIO",
        0.03
    )
    BODY_MASK_TYPE_THIN_THRESHOLD = get_config_value(
        "body_mask_type_thin_threshold",
        "VISION_BODY_MASK_TYPE_THIN_THRESHOLD",
        0.32
    )
    BODY_MASK_TYPE_FAT_THRESHOLD = get_config_value(
        "body_mask_type_fat_threshold",
        "VISION_BODY_MASK_TYPE_FAT_THRESHOLD",
        0.46
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

    PERSON_DETECTOR_MODEL = get_config_value(
        "person_detector_model",
        "VISION_PERSON_DETECTOR_MODEL",
        "models/person_detection/person_yolov8n.onnx"
    )
    PERSON_DETECTOR_INPUT_WIDTH = get_config_value(
        "person_detector_input_width",
        "VISION_PERSON_DETECTOR_INPUT_WIDTH",
        640
    )
    PERSON_DETECTOR_INPUT_HEIGHT = get_config_value(
        "person_detector_input_height",
        "VISION_PERSON_DETECTOR_INPUT_HEIGHT",
        640
    )
    PERSON_DETECTOR_SCORE_THRESHOLD = get_config_value(
        "person_detector_score_threshold",
        "VISION_PERSON_DETECTOR_SCORE_THRESHOLD",
        0.35
    )
    PERSON_DETECTOR_NMS_THRESHOLD = get_config_value(
        "person_detector_nms_threshold",
        "VISION_PERSON_DETECTOR_NMS_THRESHOLD",
        0.45
    )
    PERSON_DETECTOR_PERSON_CLASS_ID = get_config_value(
        "person_detector_person_class_id",
        "VISION_PERSON_DETECTOR_PERSON_CLASS_ID",
        0
    )

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


settings = Settings()
