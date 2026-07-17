"""
配置管理模块

负责加载和解析应用配置，支持三层优先级：
1. 环境变量（最高优先级）
2. config.json 文件
3. 代码中的默认值

同时处理路径解析（支持 PyInstaller 打包后的运行时路径和开发时的当前目录路径）。
"""

import json
import os
import sys
from pathlib import Path

import jsonschema

from vision._build_version import APP_VERSION as BUILD_APP_VERSION

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------

# 运行时基准目录：PyInstaller 打包后使用 _MEIPASS 临时目录，开发时使用项目根目录
RUNTIME_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
# 配置文件基准目录：默认从当前工作目录开始查找
CONFIG_BASE_DIR = Path.cwd()
MANAGED_CONFIG_MODE = os.getenv("VISION_CONFIG_MODE", "").strip().lower() == "managed"


class ConfigError(RuntimeError):
    """VEM 托管的外部现场配置无效。"""


def _validate_managed_config(config, config_path):
    if not isinstance(config, dict):
        raise ConfigError(f"managed configuration must be an object: {config_path}")

    schema_path = RUNTIME_BASE_DIR / "config" / "vending-vision-site-config-v1.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        allowed_keys = set(schema["properties"])
    except Exception as exc:
        raise ConfigError("打包的现场配置 schema 不可用") from exc
    try:
        jsonschema.Draft202012Validator(schema).validate(config)
    except jsonschema.ValidationError as exc:
        raise ConfigError(f"托管现场配置不符合 schema: {exc.message}") from exc
    unknown = sorted(set(config) - allowed_keys)
    if unknown:
        raise ConfigError(f"managed configuration contains unknown keys: {', '.join(unknown)}")
    if config.get("schemaVersion") != "vending-vision-site-config/v1":
        raise ConfigError("managed configuration schemaVersion must be vending-vision-site-config/v1")
    host = config.get("host")
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ConfigError("managed configuration host must be loopback")
    port = config.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("managed configuration port must be an integer from 1 to 65535")
    origins = config.get("allowed_origins")
    if not isinstance(origins, list) or not origins or not all(
        isinstance(item, str) and item.startswith("http://") for item in origins
    ):
        raise ConfigError("managed configuration allowed_origins must contain local HTTP origins")

    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != {"top", "front"}:
        raise ConfigError("managed configuration must define exactly top and front cameras")
    camera_keys = {
        "backend", "width", "height", "fps", "fourcc", "role",
        "keep_open", "rotate", "roi", "source", "video_path", "loop",
    }
    expected_roles = {"top": "presence", "front": "profile_tryon"}
    camera_sources = {}
    for camera_name, camera in cameras.items():
        if not isinstance(camera, dict):
            raise ConfigError(f"camera {camera_name} must be an object")
        camera_unknown = sorted(set(camera) - camera_keys)
        if camera_unknown:
            raise ConfigError(
                f"camera {camera_name} contains unknown keys: {', '.join(camera_unknown)}"
            )
        if camera.get("role") != expected_roles[camera_name]:
            raise ConfigError(f"camera {camera_name}.role is invalid")
        if camera.get("rotate", 0) not in {0, 90, 180, 270}:
            raise ConfigError(f"camera {camera_name}.rotate must be 0, 90, 180, or 270")
        source = str(camera.get("source", "dshow")).lower()
        camera_sources[camera_name] = source
        if source not in {"dshow", "recorded_video"}:
            raise ConfigError(f"camera {camera_name}.source is invalid")
        if source == "recorded_video" and not (
            isinstance(camera.get("video_path"), str)
            and camera["video_path"].strip()
        ):
            raise ConfigError(f"camera {camera_name}.video_path is required for recorded_video")

    if len(set(camera_sources.values())) != 1:
        raise ConfigError("managed configuration cannot mix recorded_video and dshow camera sources")

    return config


def runtime_path(path):
    """将配置中的相对路径解析为运行时实际路径。

    解析顺序：
    1. 如果是绝对路径，直接返回
    2. 如果相对于当前工作目录存在，返回该路径
    3. 否则相对于运行时基准目录（PyInstaller 打包目录）返回
    """
    value = Path(path)

    if value.is_absolute():
        return str(value)

    cwd_path = Path.cwd() / value
    if cwd_path.exists():
        return str(cwd_path)

    return str(RUNTIME_BASE_DIR / value)


def _config_candidates():
    """生成配置文件候选路径列表。

    优先使用 VISION_CONFIG_FILE 环境变量指定的路径，
    其次使用当前目录下的 config.json，
    最后使用打包目录中的 config.json。
    """
    env_config = os.getenv("VISION_CONFIG_FILE")

    if env_config:
        yield Path(env_config)
        return

    yield Path.cwd() / "config.json"
    bundled_config = RUNTIME_BASE_DIR / "config.json"

    if bundled_config != Path.cwd() / "config.json":
        yield bundled_config


def load_json_config():
    """从候选路径中加载 JSON 配置文件。

    遍历所有候选路径，返回第一个成功解析的配置字典。
    同时更新 CONFIG_BASE_DIR 为配置文件所在目录。
    """
    global CONFIG_BASE_DIR

    candidates = list(_config_candidates())
    if MANAGED_CONFIG_MODE:
        if not os.getenv("VISION_CONFIG_FILE"):
            raise ConfigError("managed launch requires --config")
        config_path = candidates[0]
        if not config_path.is_file():
            raise ConfigError(f"managed configuration does not exist: {config_path}")
        try:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"managed configuration is not valid UTF-8 JSON: {config_path}") from exc
        CONFIG_BASE_DIR = config_path.parent
        return _validate_managed_config(parsed, config_path)

    for config_path in candidates:
        if not config_path.exists():
            continue

        try:
            CONFIG_BASE_DIR = config_path.parent
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}


# 全局配置缓存：在模块加载时一次性读取 JSON 配置文件
_json_config = load_json_config()

# ---------------------------------------------------------------------------
# 配置读取工具函数
# ---------------------------------------------------------------------------


def get_config_value(key, env_key, default):
    """
    获取单个配置值，按优先级：环境变量 > config.json > 默认值。

    自动根据默认值类型进行类型转换（bool/int/float/str）。
    """
    if not MANAGED_CONFIG_MODE and env_key in os.environ:
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
    """获取 JSON 类型的配置值（如字典、列表），支持环境变量覆盖。"""
    if not MANAGED_CONFIG_MODE and env_key in os.environ:
        try:
            return json.loads(os.getenv(env_key) or "")
        except Exception:
            return default

    value = _json_config.get(key, default)
    return value if isinstance(value, type(default)) else default


def get_nested_config_value(section, key, env_key, default):
    """获取嵌套配置值（从 config.json 的某个 section 下读取 key）。"""
    if not MANAGED_CONFIG_MODE and env_key in os.environ:
        value = os.getenv(env_key)

        if isinstance(default, bool):
            return str(value).lower() == "true"

        if isinstance(default, int):
            return int(value)

        if isinstance(default, float):
            return float(value)

        try:
            if isinstance(default, (dict, list)):
                return json.loads(value or "")
        except Exception:
            return default

        return value

    section_value = _json_config.get(section, {})
    if not isinstance(section_value, dict):
        return default

    value = section_value.get(key, default)
    return value if isinstance(value, type(default)) else default


def get_model_config(name, default):
    """获取模型配置，将 JSON 中的模型参数与默认值合并。"""
    models = _json_config.get("models", {})

    if not isinstance(models, dict):
        return dict(default)

    value = models.get(name, {})
    config = dict(default)

    if isinstance(value, dict):
        for key, item in value.items():
            if item is not None:
                config[key] = item

    return config


def get_section_config(section, default):
    """获取整个 section 的配置，将 JSON 值与默认值合并覆盖。"""
    value = _json_config.get(section, {})
    config = dict(default)

    if isinstance(value, dict):
        for key, item in value.items():
            if item is not None:
                config[key] = item

    return config


def get_path_config_value(key, env_key, default):
    """获取路径类型的配置值，自动解析为运行时实际路径。"""
    if not MANAGED_CONFIG_MODE and env_key in os.environ:
        return runtime_path(os.getenv(env_key) or default)

    value = _json_config.get(key, default)
    path = Path(value)

    if path.is_absolute():
        return str(path)

    config_path = CONFIG_BASE_DIR / path
    if config_path.exists():
        return str(config_path)

    return runtime_path(value)


def build_camera_config(cameras, role, fallback):
    """根据角色（top/front）构建摄像头配置，未指定的字段使用后备值。"""
    config = dict(fallback)
    role_config = cameras.get(role, {}) if isinstance(cameras, dict) else {}

    if isinstance(role_config, dict):
        for key, value in role_config.items():
            if value is not None:
                if key == "video_path" and isinstance(value, str):
                    path = Path(value)
                    config[key] = str(path if path.is_absolute() else CONFIG_BASE_DIR / path)
                    continue
                config[key] = value

    config.setdefault("role", role)
    return config


def bool_config_value(value, default=False):
    """安全地将配置值转换为布尔类型。"""
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    return str(value).lower() == "true"


class Settings:
    APP_NAME = "Vending Vision Module"
    APP_VERSION = BUILD_APP_VERSION
    PROTOCOL = "vem.vision.v1"
    PERSON_DETECTOR_CONFIG = get_model_config(
        "person_detector",
        {
            "type": "yolo11",
            "path": "models/person_detection/yolo11s.onnx",
            "fallback_path": "models/person_detection/person_yolov8n.onnx",
            "input_size": 640,
            "conf_threshold": 0.35,
            "iou_threshold": 0.45,
            "person_class_id": 0,
        },
    )
    FACE_DETECTOR_CONFIG = get_model_config(
        "face_detector",
        {
            "type": "scrfd",
            "path": "models/face_detection/scrfd_10g.onnx",
            "fallback_type": "yunet",
            "fallback_path": "models/face_detection/face_detection_yunet_2023mar.onnx",
            "conf_threshold": 0.45,
        },
    )
    AGE_GENDER_CONFIG = get_model_config(
        "age_gender",
        {
            "type": "openvino",
            "xml_path": "models/age_gender/age-gender-recognition-retail-0013.xml",
            "bin_path": "models/age_gender/age-gender-recognition-retail-0013.bin",
            "fallback_type": "caffe",
        },
    )
    POSE_CONFIG = get_model_config(
        "pose",
        {
            "type": "yolo11_pose",
            "path": "models/pose/yolo11s-pose.onnx",
            "fallback_type": "mediapipe",
            "conf_threshold": 0.35,
        },
    )
    TOP_OCCUPANCY_CONFIG = get_section_config(
        "top_occupancy",
        {
            "enabled": True,
            "roi": [0.0, 0.0, 1.0, 1.0],
            "history_size": 2,
            "present_min_frames": 2,
            "single_min_frames": 1,
            "multiple_min_frames": 1,
            "absent_min_seconds": 0.6,
            "track_iou_threshold": 0.3,
            "track_min_age_frames": 1,
            "track_max_missed_frames": 5,
        },
    )
    PROFILE_SAMPLING_CONFIG = get_section_config(
        "profile_sampling",
        {
            "duration_sec": 2.0,
            "early_finish_after_sec": 1.0,
            "target_fps": 8,
            "min_good_frames": 2,
            "max_good_frames": 10,
            "min_face_area_ratio": 0.006,
            "min_face_score": 0.35,
            "min_person_score": 0.30,
            "min_blur_score": 25.0,
            "brightness_min": 35,
            "brightness_max": 230,
        },
    )
    POSE_ENABLE_SEGMENTATION = get_config_value(
        "pose_enable_segmentation",
        "VISION_POSE_ENABLE_SEGMENTATION",
        True
    )

    HOST = get_config_value("host", "VISION_HOST", "127.0.0.1")
    PORT = get_config_value("port", "VISION_PORT", 7892)
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
        0.25
    )
    PROFILE_OCCUPANCY_GATE_ENABLED = get_config_value(
        "profile_occupancy_gate_enabled",
        "VISION_PROFILE_OCCUPANCY_GATE_ENABLED",
        True
    )
    PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES = get_config_value(
        "profile_occupancy_reset_absent_frames",
        "VISION_PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES",
        1
    )
    PROFILE_MIN_CONFIDENCE = get_config_value(
        "profile_min_confidence",
        "VISION_PROFILE_MIN_CONFIDENCE",
        0.25
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
        0.005
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
        0.07
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
        0.08
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
        "backend": CAMERA_BACKEND,
        "width": CAMERA_WIDTH,
        "height": CAMERA_HEIGHT,
        "fps": CAMERA_FPS,
        "fourcc": CAMERA_FOURCC,
        "keep_open": CAMERA_KEEP_OPEN,
    }
    TOP_CAMERA_CONFIG = build_camera_config(CAMERAS, "top", LEGACY_CAMERA_CONFIG)
    FRONT_CAMERA_CONFIG = build_camera_config(CAMERAS, "front", LEGACY_CAMERA_CONFIG)
    TOP_CAMERA_BACKEND = TOP_CAMERA_CONFIG.get("backend", CAMERA_BACKEND)
    TOP_CAMERA_WIDTH = int(TOP_CAMERA_CONFIG.get("width", CAMERA_WIDTH) or 0)
    TOP_CAMERA_HEIGHT = int(TOP_CAMERA_CONFIG.get("height", CAMERA_HEIGHT) or 0)
    TOP_CAMERA_FPS = int(TOP_CAMERA_CONFIG.get("fps", CAMERA_FPS) or 0)
    TOP_CAMERA_FOURCC = TOP_CAMERA_CONFIG.get("fourcc", CAMERA_FOURCC)
    TOP_CAMERA_KEEP_OPEN = bool_config_value(
        TOP_CAMERA_CONFIG.get("keep_open"),
        CAMERA_KEEP_OPEN
    )
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
        125
    )
    FRONT_CAMERA_OWNER_TIMEOUT_MS = get_config_value(
        "front_camera_owner_timeout_ms",
        "VISION_FRONT_CAMERA_OWNER_TIMEOUT_MS",
        120000
    )
    TRY_ON_SESSION_TTL_MS = get_config_value(
        "try_on_session_ttl_ms",
        "VISION_TRY_ON_SESSION_TTL_MS",
        10 * 60 * 1000,
    )
    TRY_ON_SESSION_HISTORY_LIMIT = get_config_value(
        "try_on_session_history_limit",
        "VISION_TRY_ON_SESSION_HISTORY_LIMIT",
        32,
    )
    TRY_ON_MAX_STREAM_CLIENTS = get_config_value(
        "try_on_max_stream_clients",
        "VISION_TRY_ON_MAX_STREAM_CLIENTS",
        2,
    )
    WEBSOCKET_SEND_TIMEOUT_MS = get_config_value(
        "websocket_send_timeout_ms",
        "VISION_WEBSOCKET_SEND_TIMEOUT_MS",
        2000,
    )
    WEBSOCKET_QUEUE_SIZE = get_config_value(
        "websocket_queue_size",
        "VISION_WEBSOCKET_QUEUE_SIZE",
        16,
    )
    ALLOWED_ORIGINS = (
        tuple(get_json_config_value("allowed_origins", "VISION_ALLOWED_ORIGINS_JSON", []))
        if MANAGED_CONFIG_MODE
        else tuple(
            item.strip()
            for item in str(os.getenv("VISION_ALLOWED_ORIGINS", "")).split(",")
            if item.strip()
        )
    )

    DEVELOPMENT_DASHBOARD_ENABLED = (
        not MANAGED_CONFIG_MODE
        and bool_config_value(
            get_config_value(
                "development_dashboard", "VISION_DEVELOPMENT_DASHBOARD", False
            )
        )
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

    FACE_DETECTOR_MODEL = get_path_config_value(
        "face_detector_model",
        "VISION_FACE_DETECTOR_MODEL",
        FACE_DETECTOR_CONFIG.get(
            "fallback_path",
            "models/face_detection/face_detection_yunet_2023mar.onnx",
        )
    )
    SCRFD_FACE_DETECTOR_MODEL = get_path_config_value(
        "scrfd_face_detector_model",
        "VISION_SCRFD_FACE_DETECTOR_MODEL",
        FACE_DETECTOR_CONFIG.get(
            "path",
            "models/face_detection/scrfd_10g.onnx",
        )
    )
    FACE_DETECTOR_CONF_THRESHOLD = float(
        FACE_DETECTOR_CONFIG.get("conf_threshold", 0.45)
    )

    FACE_SCORE_THRESHOLD = float(os.getenv("VISION_FACE_SCORE_THRESHOLD", "0.6"))
    FACE_NMS_THRESHOLD = float(os.getenv("VISION_FACE_NMS_THRESHOLD", "0.3"))
    FACE_TOP_K = int(os.getenv("VISION_FACE_TOP_K", "5000"))

    PERSON_DETECTOR_MODEL = get_path_config_value(
        "person_detector_model",
        "VISION_PERSON_DETECTOR_MODEL",
        PERSON_DETECTOR_CONFIG.get("path", "models/person_detection/yolo11s.onnx")
    )
    PERSON_DETECTOR_FALLBACK_MODEL = get_path_config_value(
        "person_detector_fallback_model",
        "VISION_PERSON_DETECTOR_FALLBACK_MODEL",
        PERSON_DETECTOR_CONFIG.get(
            "fallback_path",
            "models/person_detection/person_yolov8n.onnx",
        )
    )
    PERSON_DETECTOR_INPUT_WIDTH = get_config_value(
        "person_detector_input_width",
        "VISION_PERSON_DETECTOR_INPUT_WIDTH",
        int(PERSON_DETECTOR_CONFIG.get("input_size", 640))
    )
    PERSON_DETECTOR_INPUT_HEIGHT = get_config_value(
        "person_detector_input_height",
        "VISION_PERSON_DETECTOR_INPUT_HEIGHT",
        int(PERSON_DETECTOR_CONFIG.get("input_size", 640))
    )
    PERSON_DETECTOR_SCORE_THRESHOLD = get_config_value(
        "person_detector_score_threshold",
        "VISION_PERSON_DETECTOR_SCORE_THRESHOLD",
        float(PERSON_DETECTOR_CONFIG.get("conf_threshold", 0.35))
    )
    PERSON_DETECTOR_NMS_THRESHOLD = get_config_value(
        "person_detector_nms_threshold",
        "VISION_PERSON_DETECTOR_NMS_THRESHOLD",
        float(PERSON_DETECTOR_CONFIG.get("iou_threshold", 0.45))
    )
    PERSON_DETECTOR_PERSON_CLASS_ID = get_config_value(
        "person_detector_person_class_id",
        "VISION_PERSON_DETECTOR_PERSON_CLASS_ID",
        int(PERSON_DETECTOR_CONFIG.get("person_class_id", 0))
    )

    AGE_MODEL_PROTO = get_path_config_value(
        "age_model_proto",
        "VISION_AGE_MODEL_PROTO",
        "models/age_gender/age_deploy.prototxt"
    )
    AGE_MODEL_WEIGHTS = get_path_config_value(
        "age_model_weights",
        "VISION_AGE_MODEL_WEIGHTS",
        "models/age_gender/age_net.caffemodel"
    )
    GENDER_MODEL_PROTO = get_path_config_value(
        "gender_model_proto",
        "VISION_GENDER_MODEL_PROTO",
        "models/age_gender/gender_deploy.prototxt"
    )
    GENDER_MODEL_WEIGHTS = get_path_config_value(
        "gender_model_weights",
        "VISION_GENDER_MODEL_WEIGHTS",
        "models/age_gender/gender_net.caffemodel"
    )
    OPENVINO_AGE_GENDER_XML = get_path_config_value(
        "openvino_age_gender_xml",
        "VISION_OPENVINO_AGE_GENDER_XML",
        AGE_GENDER_CONFIG.get(
            "xml_path",
            "models/age_gender/age-gender-recognition-retail-0013.xml",
        )
    )
    OPENVINO_AGE_GENDER_BIN = get_path_config_value(
        "openvino_age_gender_bin",
        "VISION_OPENVINO_AGE_GENDER_BIN",
        AGE_GENDER_CONFIG.get(
            "bin_path",
            "models/age_gender/age-gender-recognition-retail-0013.bin",
        )
    )

    LOG_DIR = os.getenv("VISION_LOG_DIR", "logs")
    LOG_FILE = os.getenv("VISION_LOG_FILE", "logs/vision.log")
    LOG_LEVEL = get_config_value("log_level", "VISION_LOG_LEVEL", "INFO")
    LOG_MAX_BYTES = get_config_value(
        "log_max_bytes",
        "VISION_LOG_MAX_BYTES",
        5 * 1024 * 1024
    )
    LOG_BACKUP_COUNT = get_config_value(
        "log_backup_count",
        "VISION_LOG_BACKUP_COUNT",
        5
    )


settings = Settings()
