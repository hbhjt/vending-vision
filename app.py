import asyncio
import json
from json import JSONDecodeError
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from vision.camera import (
    capture_image,
    get_configured_camera_status,
    probe_cameras,
    release_camera_stream,
    reset_camera_stream,
)
from vision.config import settings
from vision.debug_saver import save_debug_images
from vision.logger import logger
from vision.pipeline import infer_image
from vision.profile_push import collect_profile_update
from vision.profile_mapper import vision_profile_to_protocol
from vision.protocol import APP_VERSION, PROTOCOL, envelope, error_envelope, now_iso
from vision.proximity import check_proximity_once
from vision.schema import VisionProfile
from vision.self_check import run_self_check


app = FastAPI(
    title=settings.APP_NAME,
    description="Smart vending machine vision API",
    version=APP_VERSION,
)


busy = False
startup_check = None
active_session_id: str | None = None
DASHBOARD_FILE = Path(__file__).parent / "dashboard" / "profile_dashboard.html"


@app.on_event("startup")
def on_startup():
    global startup_check

    logger.info("Running vision module self check...")
    startup_check = run_self_check()

    if startup_check["ok"]:
        logger.info("Vision module self check passed")
    else:
        logger.warning(f"Vision module self check failed: {startup_check}")


@app.on_event("shutdown")
def on_shutdown():
    release_camera_stream()
    logger.info("Camera stream released")


def get_startup_check():
    global startup_check

    if startup_check is None:
        startup_check = run_self_check()

    return startup_check


def get_runtime_status():
    check = get_startup_check()
    checks = dict(check["checks"])

    if settings.MOCK_SCENARIO == "off":
        try:
            camera_status = get_configured_camera_status()
            checks["camera"] = {
                "ok": True,
                "message": (
                    f"camera opened, index={settings.CAMERA_INDEX}, "
                    f"backend={settings.CAMERA_BACKEND}, "
                    f"mode={camera_status.get('mode')}, "
                    f"frame={camera_status['frame']['width']}x"
                    f"{camera_status['frame']['height']}"
                ),
                "detail": camera_status,
            }
        except Exception as e:
            checks["camera"] = {
                "ok": False,
                "message": str(e),
            }

    camera_ready = checks["camera"]["ok"]
    model_ready = checks["pose"]["ok"] and checks["face"]["ok"]
    age_gender_ready = checks["ageGender"]["modelReady"]
    age_gender_mode = checks["ageGender"]["mode"]

    return {
        "check": check,
        "cameraReady": camera_ready,
        "modelReady": model_ready,
        "ageGenderReady": age_gender_ready,
        "ageGenderMode": age_gender_mode,
    }


def run_capture_and_infer(session_id: str | None = None):
    scenario = settings.MOCK_SCENARIO

    if scenario != "off":
        logger.info(f"Using mock scenario: {scenario}")

    if scenario == "success":
        return VisionProfile(
            age=22,
            gender="unknown",
            height_cm=172.0,
            shoulder_width_cm=43.0,
            body_type="medium",
            upper_color="dark",
            presence=True,
        ), {}

    if scenario == "no_person":
        return VisionProfile(
            age=None,
            gender="unknown",
            height_cm=None,
            shoulder_width_cm=None,
            body_type="unknown",
            upper_color="unknown",
            presence=False,
        ), {}

    if scenario == "camera_unavailable":
        raise RuntimeError("camera unavailable")

    if scenario == "timeout":
        import time

        time.sleep(60)
        return VisionProfile(
            age=None,
            gender="unknown",
            height_cm=None,
            shoulder_width_cm=None,
            body_type="unknown",
            upper_color="unknown",
            presence=False,
        ), {}

    image = capture_image()
    debug_info = save_debug_images(image, session_id=session_id)
    profile = infer_image(image)

    return profile, debug_info


def normalize_timeout_ms(value) -> int:
    if value is None:
        return settings.DEFAULT_TIMEOUT_MS

    try:
        timeout_ms = int(value)
    except (TypeError, ValueError):
        return settings.DEFAULT_TIMEOUT_MS

    return max(1000, min(timeout_ms, 30000))


def build_quality(profile: VisionProfile, debug_info: dict | None = None):
    warnings = []

    if settings.MOCK_SCENARIO != "off":
        warnings.append(f"mock scenario enabled: {settings.MOCK_SCENARIO}")

    if profile.height_cm is None:
        warnings.append("height is unavailable or filtered")

    if profile.body_type == "unknown":
        warnings.append("body type is unavailable")

    if profile.gender == "unknown":
        warnings.append("gender is unknown")

    if profile.age is None:
        warnings.append("age range is unknown")

    confidence = vision_profile_to_protocol(profile)["confidence"]

    if confidence >= 0.75:
        overall = "good"
    elif confidence >= 0.45:
        overall = "fair"
    else:
        overall = "poor"

    quality = {
        "overall": overall,
        "warnings": warnings,
    }

    if debug_info:
        quality["debug"] = debug_info

    return quality


def validate_envelope(message):
    if not isinstance(message, dict):
        return "message must be a JSON object"

    required_fields = ["protocol", "type", "messageId", "timestamp", "payload"]
    missing = [field for field in required_fields if field not in message]

    if missing:
        return f"missing required field(s): {', '.join(missing)}"

    if not isinstance(message.get("payload"), dict):
        return "payload must be an object"

    return None


@app.get("/")
def root():
    return {
        "message": "Vending Vision Module is running",
        "api": {
            "dashboard": "/dashboard",
            "infer": "/infer",
            "capture_infer": "/capture_infer",
            "camera_status": "/camera/status",
            "camera_reopen": "/camera/reopen",
            "camera_probe": "/camera/probe",
            "ws": "/ws",
            "health": "/health",
            "version": "/version",
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD_FILE.exists():
        return HTMLResponse(
            status_code=404,
            content="<h1>dashboard not found</h1>",
        )

    return HTMLResponse(
        content=DASHBOARD_FILE.read_text(encoding="utf-8"),
    )


@app.get("/health")
def health():
    status = get_runtime_status()

    service_status = (
        "ok"
        if status["cameraReady"] and status["modelReady"]
        else "degraded"
    )

    return {
        "status": service_status,
        "module": "vision",
        "protocol": PROTOCOL,
        "version": APP_VERSION,
        "mockScenario": settings.MOCK_SCENARIO,
        "busy": busy,
        "cameraReady": status["cameraReady"],
        "modelReady": status["modelReady"],
        "ageGenderReady": status["ageGenderReady"],
        "ageGenderMode": status["ageGenderMode"],
        "checks": status["check"]["checks"],
    }


@app.get("/version")
def version():
    return {
        "module": settings.APP_NAME,
        "version": APP_VERSION,
        "protocol": PROTOCOL,
        "host": settings.HOST,
        "port": settings.PORT,
        "mock_scenario": settings.MOCK_SCENARIO,
        "default_timeout_ms": settings.DEFAULT_TIMEOUT_MS,
        "debug": {
            "save_debug_images": settings.SAVE_DEBUG_IMAGES,
            "debug_output_dir": settings.DEBUG_OUTPUT_DIR,
            "max_debug_images": settings.MAX_DEBUG_IMAGES,
        },
        "process_trace": {
            "enabled": settings.PROCESS_TRACE_ENABLED,
            "output_dir": settings.PROCESS_TRACE_OUTPUT_DIR,
            "max_events": settings.PROCESS_TRACE_MAX_EVENTS,
        },
        "profile_push": {
            "enabled": settings.PROFILE_PUSH_ENABLED,
            "push_interval_ms": settings.PROFILE_PUSH_INTERVAL_MS,
            "push_cooldown_ms": settings.PROFILE_PUSH_COOLDOWN_MS,
            "sample_count": settings.PROFILE_SAMPLE_COUNT,
            "body_buffer_max_frames": settings.PROFILE_BODY_BUFFER_MAX_FRAMES,
            "body_buffer_ttl_ms": settings.PROFILE_BODY_BUFFER_TTL_MS,
            "track_enabled": settings.PROFILE_TRACK_ENABLED,
            "track_max_missing_frames": settings.PROFILE_TRACK_MAX_MISSING_FRAMES,
            "track_max_center_shift": settings.PROFILE_TRACK_MAX_CENTER_SHIFT,
            "track_min_match_score": settings.PROFILE_TRACK_MIN_MATCH_SCORE,
            "occupancy_gate_enabled": settings.PROFILE_OCCUPANCY_GATE_ENABLED,
            "occupancy_reset_absent_frames": (
                settings.PROFILE_OCCUPANCY_RESET_ABSENT_FRAMES
            ),
            "min_confidence": settings.PROFILE_MIN_CONFIDENCE,
            "min_valid_frames": settings.PROFILE_MIN_VALID_FRAMES,
            "face_vote_enabled": settings.PROFILE_FACE_VOTE_ENABLED,
            "face_vote_sample_count": settings.PROFILE_FACE_VOTE_SAMPLE_COUNT,
            "face_vote_interval_ms": settings.PROFILE_FACE_VOTE_INTERVAL_MS,
            "face_vote_min_sharpness": settings.PROFILE_FACE_VOTE_MIN_SHARPNESS,
            "detection_width": settings.PROFILE_DETECTION_WIDTH,
            "detection_height": settings.PROFILE_DETECTION_HEIGHT,
        },
        "proximity": {
            "enabled": settings.PROXIMITY_ENABLED,
            "monitor_width": settings.PROXIMITY_MONITOR_WIDTH,
            "monitor_height": settings.PROXIMITY_MONITOR_HEIGHT,
            "present_face_ratio": settings.PROXIMITY_PRESENT_FACE_RATIO,
            "close_face_ratio": settings.PROXIMITY_CLOSE_FACE_RATIO,
            "close_consecutive_frames": settings.PROXIMITY_CLOSE_CONSECUTIVE_FRAMES,
            "person_enabled": settings.PROXIMITY_PERSON_ENABLED,
            "present_person_ratio": settings.PROXIMITY_PRESENT_PERSON_RATIO,
            "close_person_ratio": settings.PROXIMITY_CLOSE_PERSON_RATIO,
            "body_enabled": settings.PROXIMITY_BODY_ENABLED,
            "body_min_visibility": settings.PROXIMITY_BODY_MIN_VISIBILITY,
            "body_min_visible_points": settings.PROXIMITY_BODY_MIN_VISIBLE_POINTS,
            "present_body_ratio": settings.PROXIMITY_PRESENT_BODY_RATIO,
            "close_body_ratio": settings.PROXIMITY_CLOSE_BODY_RATIO,
        },
        "primary_target": {
            "mode": "single_user",
            "primary_face_max_head_distance_ratio": (
                settings.PRIMARY_FACE_MAX_HEAD_DISTANCE_RATIO
            ),
        },
        "calibration": {
            "camera_index": settings.CAMERA_INDEX,
            "camera_backend": settings.CAMERA_BACKEND,
            "camera_width": settings.CAMERA_WIDTH,
            "camera_height": settings.CAMERA_HEIGHT,
            "camera_fps": settings.CAMERA_FPS,
            "camera_fourcc": settings.CAMERA_FOURCC,
            "camera_keep_open": settings.CAMERA_KEEP_OPEN,
            "camera_read_retry_count": settings.CAMERA_READ_RETRY_COUNT,
            "camera_reconnect_delay_ms": settings.CAMERA_RECONNECT_DELAY_MS,
            "height_scale": settings.HEIGHT_SCALE,
            "height_offset": settings.HEIGHT_OFFSET,
            "body_type_thin_threshold": settings.BODY_TYPE_THIN_THRESHOLD,
            "body_type_fat_threshold": settings.BODY_TYPE_FAT_THRESHOLD,
            "upper_body_type_thin_threshold": (
                settings.UPPER_BODY_TYPE_THIN_THRESHOLD
            ),
            "upper_body_type_fat_threshold": (
                settings.UPPER_BODY_TYPE_FAT_THRESHOLD
            ),
            "pose_enable_segmentation": settings.POSE_ENABLE_SEGMENTATION,
            "body_mask_enabled": settings.BODY_MASK_ENABLED,
            "body_mask_threshold": settings.BODY_MASK_THRESHOLD,
            "body_mask_min_area_ratio": settings.BODY_MASK_MIN_AREA_RATIO,
            "body_mask_type_thin_threshold": (
                settings.BODY_MASK_TYPE_THIN_THRESHOLD
            ),
            "body_mask_type_fat_threshold": (
                settings.BODY_MASK_TYPE_FAT_THRESHOLD
            ),
        },
        "model_paths": {
            "face_detector_model": settings.FACE_DETECTOR_MODEL,
            "person_detector_model": settings.PERSON_DETECTOR_MODEL,
            "age_model_proto": settings.AGE_MODEL_PROTO,
            "age_model_weights": settings.AGE_MODEL_WEIGHTS,
            "gender_model_proto": settings.GENDER_MODEL_PROTO,
            "gender_model_weights": settings.GENDER_MODEL_WEIGHTS,
        },
        "features": {
            "presence_detection": True,
            "pose_estimation": True,
            "body_estimation": True,
            "face_detection": "yunet_or_haar",
            "person_detection": "opencv_dnn_onnx_optional",
            "age_gender_estimation": "opencv_dnn_or_mock",
            "fastapi_infer_api": True,
            "camera_capture_api": True,
            "websocket_protocol": True,
            "timeout_control": True,
            "logging": True,
            "mock_scenario": True,
            "debug_images": True,
            "debug_cleanup": True,
        },
    }


@app.get("/camera/status")
def camera_status():
    try:
        return get_configured_camera_status()
    except Exception as e:
        logger.exception("HTTP /camera/status failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/camera/reopen")
def camera_reopen():
    try:
        reset_camera_stream()
        return get_configured_camera_status()
    except Exception as e:
        logger.exception("HTTP /camera/reopen failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/camera/probe")
def camera_probe(max_index: int = 8, backend: str | None = None):
    try:
        return {
            "backend": backend or settings.CAMERA_BACKEND,
            "maxIndex": max_index,
            "cameras": probe_cameras(max_index=max_index, backend_name=backend),
        }
    except Exception as e:
        logger.exception("HTTP /camera/probe failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/proximity/check")
def proximity_check():
    try:
        return check_proximity_once()
    except Exception as e:
        logger.exception("HTTP /proximity/check failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/infer", response_model=VisionProfile)
async def infer(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "failed to read image, please upload jpg/png"},
            )

        return infer_image(image)

    except Exception as e:
        logger.exception("HTTP /infer failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/capture_infer", response_model=VisionProfile)
def capture_infer():
    try:
        profile, _ = run_capture_and_infer()
        return profile

    except Exception as e:
        logger.exception("HTTP /capture_infer failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


async def send_error(
    websocket: WebSocket,
    code: str,
    message: str,
    session_id: str | None = None,
    retryable: bool = True,
    detail: dict | None = None,
    message_id: str | None = None,
):
    await websocket.send_json(
        error_envelope(
            code=code,
            message=message,
            session_id=session_id,
            retryable=retryable,
            detail=detail,
            message_id=message_id,
        )
    )


async def profile_push_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    presence_status_enabled: bool = False,
):
    while True:
        try:
            update = await asyncio.to_thread(
                collect_profile_update,
                presence_status_enabled,
            )

            if update is None:
                await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)
                continue

            event_payload = update["payload"]
            message_type = update["message_type"]
            message_prefix = (
                "result"
                if message_type == "vision.profile_result"
                else "status"
            )

            async with send_lock:
                await websocket.send_json(
                    envelope(
                        message_type=message_type,
                        message_id=f"{message_prefix}-{event_payload['eventId']}",
                        payload=event_payload,
                    )
                )

            if message_type == "vision.profile_result":
                logger.info(f"Profile push sent eventId={event_payload['eventId']}")
                await asyncio.sleep(settings.PROFILE_PUSH_COOLDOWN_MS / 1000.0)
            else:
                await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)

        except RuntimeError as e:
            async with send_lock:
                await websocket.send_json(
                    error_envelope(
                        code="camera_unavailable",
                        message=str(e),
                        retryable=True,
                    )
                )
            logger.exception("Profile push camera error")
            await asyncio.sleep(settings.PROFILE_PUSH_COOLDOWN_MS / 1000.0)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            async with send_lock:
                await websocket.send_json(
                    error_envelope(
                        code="internal_error",
                        message=str(e),
                        retryable=True,
                    )
                )
            logger.exception("Profile push loop failed")
            await asyncio.sleep(settings.PROFILE_PUSH_COOLDOWN_MS / 1000.0)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    push_task: asyncio.Task | None = None
    send_lock = asyncio.Lock()

    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        while True:
            try:
                raw = await websocket.receive_text()
                message = json.loads(raw)
            except JSONDecodeError:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="invalid_message",
                        message="message must be UTF-8 JSON text",
                        retryable=False,
                    )
                continue

            validation_error = validate_envelope(message)

            if validation_error:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="invalid_message",
                        message=validation_error,
                        retryable=False,
                        message_id=message.get("messageId")
                        if isinstance(message, dict)
                        else None,
                    )
                continue

            message_type = message["type"]
            message_id = message["messageId"]
            protocol = message["protocol"]
            payload = message["payload"]
            session_id = payload.get("sessionId")

            logger.info(
                f"WS received type={message_type}, messageId={message_id}, sessionId={session_id}"
            )

            if protocol != PROTOCOL:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="unsupported_version",
                        message=f"protocol must be {PROTOCOL}",
                        retryable=False,
                        message_id=message_id,
                    )
                logger.warning(f"Unsupported protocol: {protocol}")
                continue

            if message_type == "vision.hello":
                status = get_runtime_status()
                client_capabilities = set(payload.get("capabilities") or [])
                presence_status_enabled = "presence_status" in client_capabilities

                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.ready",
                            message_id="ready-001",
                            payload={
                                "serverName": "vem-vision-python",
                                "serverVersion": APP_VERSION,
                                "cameraReady": status["cameraReady"],
                                "modelReady": status["modelReady"],
                                "capabilities": ["profile_push", "presence_status"],
                            },
                        )
                    )

                if not status["modelReady"]:
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code="model_not_ready",
                                message="required vision model is not ready",
                                retryable=True,
                            )
                        )
                    continue

                if settings.PROFILE_PUSH_ENABLED and push_task is None:
                    push_task = asyncio.create_task(
                        profile_push_loop(
                            websocket,
                            send_lock,
                            presence_status_enabled=presence_status_enabled,
                        )
                    )

                continue

            if message_type == "vision.ping":
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.pong",
                            message_id="pong-001",
                            payload={},
                        )
                    )
                continue

            if message_type in {"vision.start_profile", "vision.cancel"}:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="invalid_message",
                        message=(
                            f"{message_type} is not supported in push protocol; "
                            "send vision.hello and wait for vision.profile_result"
                        ),
                        retryable=False,
                        message_id=message_id,
                    )
                continue

            async with send_lock:
                await send_error(
                    websocket,
                    code="invalid_message",
                    message=f"unknown message type: {message_type}",
                    retryable=False,
                    message_id=message_id,
                )
            logger.warning(f"Invalid message type={message_type}")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if push_task is not None:
            push_task.cancel()
            try:
                await push_task
            except asyncio.CancelledError:
                pass
