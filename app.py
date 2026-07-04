import asyncio
import json
from json import JSONDecodeError
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from vision.camera_manager import (
    get_all_camera_statuses,
    get_camera_config,
    get_camera_status,
    release_all_cameras,
    reset_camera,
)
from vision.camera_owner import get_front_camera_owner
from vision.config import settings
from vision.logger import logger
from vision.profile_push import collect_profile_update
from vision.protocol import APP_VERSION, PROTOCOL, envelope, error_envelope
from vision.self_check import run_self_check
from vision.session_state import (
    get_vision_session_status,
    mark_vision_session_tryon_started,
    mark_vision_session_tryon_stopped,
)
from vision.try_on_session import (
    get_try_on_status,
    iter_try_on_mjpeg,
    is_try_on_session_active,
    start_try_on_session,
    stop_try_on_session,
)


app = FastAPI(
    title=settings.APP_NAME,
    description="Smart vending machine vision API",
    version=APP_VERSION,
)


startup_check = None
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
    release_all_cameras()
    logger.info("Camera streams released")


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
            camera_statuses = get_all_camera_statuses()
            camera_ok = all(
                status.get("ok")
                for status in camera_statuses.values()
            )
            checks["camera"] = {
                "ok": camera_ok,
                "message": "top/front cameras checked",
                "detail": camera_statuses,
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


def validate_envelope(message):
    if not isinstance(message, dict):
        return "message must be a JSON object"

    required_fields = ["protocol", "type", "messageId", "timestamp", "payload"]
    missing = [field for field in required_fields if field not in message]

    if missing:
        return f"missing required field(s): {', '.join(missing)}"

    for field in ["protocol", "type", "messageId", "timestamp"]:
        if not isinstance(message.get(field), str):
            return f"{field} must be a string"

        if not message.get(field).strip():
            return f"{field} must not be empty"

    if not isinstance(message.get("payload"), dict):
        return "payload must be an object"

    return None


def validate_message_payload(message_type: str, payload: dict):
    if message_type == "vision.hello":
        protocol_version = payload.get("protocolVersion")
        capabilities = payload.get("capabilities")
        client_role = payload.get("clientRole")
        machine_code = payload.get("machineCode")

        if (
            not isinstance(protocol_version, int)
            or isinstance(protocol_version, bool)
            or protocol_version != 1
        ):
            return "payload.protocolVersion must be 1"

        if not isinstance(capabilities, list):
            return "payload.capabilities must be an array"

        if not all(isinstance(item, str) and item.strip() for item in capabilities):
            return "payload.capabilities must contain non-empty strings"

        if client_role is not None and not isinstance(client_role, str):
            return "payload.clientRole must be a string"

        if machine_code is not None and not isinstance(machine_code, str):
            return "payload.machineCode must be a string"

    if message_type == "vision.try_on.start":
        session_id = payload.get("sessionId")
        catalog_key = payload.get("catalogKey")
        variant_id = payload.get("variantId")

        if not isinstance(session_id, str) or not session_id.strip():
            return "payload.sessionId must be a non-empty string"

        if catalog_key is not None and not isinstance(catalog_key, str):
            return "payload.catalogKey must be a string"

        if variant_id is not None and not isinstance(variant_id, str):
            return "payload.variantId must be a string"

    if message_type == "vision.try_on.stop":
        session_id = payload.get("sessionId")
        reason = payload.get("reason")

        if not isinstance(session_id, str) or not session_id.strip():
            return "payload.sessionId must be a non-empty string"

        if reason is not None and not isinstance(reason, str):
            return "payload.reason must be a string"

    return None


@app.get("/")
def root():
    return {
        "message": "Vending Vision Module is running",
        "api": {
            "dashboard": "/dashboard",
            "camera_roles_status": "/camera/roles/status",
            "front_camera_owner": "/camera/front/owner",
            "try_on_preview": "/try-on/{sessionId}.mjpeg",
            "session_status": "/session/status",
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
        "cameras": {
            "top": settings.TOP_CAMERA_CONFIG,
            "front": settings.FRONT_CAMERA_CONFIG,
            "front_owner": get_front_camera_owner(),
            "try_on": get_try_on_status(),
            "vision_session": get_vision_session_status(),
            "profile": {
                "max_wait_ms": settings.FRONT_CAMERA_PROFILE_MAX_WAIT_MS,
                "sample_count": settings.FRONT_CAMERA_PROFILE_SAMPLE_COUNT,
                "sample_interval_ms": (
                    settings.FRONT_CAMERA_PROFILE_SAMPLE_INTERVAL_MS
                ),
            },
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
            "websocket_protocol": True,
            "timeout_control": True,
            "logging": True,
            "mock_scenario": True,
        },
    }

@app.get("/camera/roles/status")
def camera_roles_status():
    return get_all_camera_statuses()


@app.get("/camera/{role}/status")
def camera_role_status(role: str):
    try:
        return get_camera_status(role)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"ok": False, "error": str(e)})
    except Exception as e:
        logger.exception(f"HTTP /camera/{role}/status failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/camera/{role}/reopen")
def camera_role_reopen(role: str):
    try:
        get_camera_config(role)
        reset_camera(role)
        return get_camera_status(role)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"ok": False, "error": str(e)})
    except Exception as e:
        logger.exception(f"HTTP /camera/{role}/reopen failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/camera/front/owner")
def front_camera_owner():
    return get_front_camera_owner()


@app.get("/session/status")
def session_status():
    return get_vision_session_status()


@app.get("/try-on/{session_id}.mjpeg")
def try_on_mjpeg(session_id: str):
    if not is_try_on_session_active(session_id):
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "try-on session is not active"},
        )

    media_type = "multipart/x-mixed-replace; boundary=frame"
    return StreamingResponse(
        iter_try_on_mjpeg(session_id),
        media_type=media_type,
    )


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
    ambient_light_enabled: bool = False,
    person_departed_enabled: bool = False,
):
    while True:
        try:
            update = await asyncio.to_thread(
                collect_profile_update,
                presence_status_enabled,
                ambient_light_enabled,
                person_departed_enabled,
            )

            if update is None:
                await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)
                continue

            event_payload = update["payload"]
            message_type = update["message_type"]
            message_prefix = {
                "vision.profile_result": "result",
                "vision.person_departed": "departure",
            }.get(message_type, "status")

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

            payload_error = validate_message_payload(message_type, payload)

            if payload_error:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="invalid_message",
                        message=payload_error,
                        retryable=False,
                        message_id=message_id,
                    )
                continue

            if message_type == "vision.hello":
                status = get_runtime_status()
                client_capabilities = set(payload.get("capabilities") or [])
                presence_status_enabled = "presence_status" in client_capabilities
                ambient_light_enabled = "ambient_light" in client_capabilities
                person_departed_enabled = "person_departed" in client_capabilities

                server_capabilities = [
                    "profile_push",
                    "presence_status",
                    "person_departed",
                    "ambient_light",
                    "try_on_session",
                ]
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
                                "capabilities": server_capabilities,
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
                            ambient_light_enabled=ambient_light_enabled,
                            person_departed_enabled=person_departed_enabled,
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

            if message_type == "vision.try_on.start":
                try:
                    session = start_try_on_session(
                        payload.get("sessionId"),
                        catalog_key=payload.get("catalogKey"),
                        variant_id=payload.get("variantId"),
                    )
                except ValueError as e:
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message=str(e),
                            retryable=False,
                            message_id=message_id,
                        )
                    continue
                except Exception as e:
                    logger.exception("Try-on start failed")
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code="try_on_unavailable",
                                message=str(e),
                                retryable=True,
                                message_id=message_id,
                            )
                        )
                    continue

                mark_vision_session_tryon_started(session)
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.try_on.started",
                            message_id=f"try-on-started-{session['sessionId']}",
                            payload={
                                "sessionId": session["sessionId"],
                                "previewUrl": session["previewUrl"],
                                "streamType": session["streamType"],
                            },
                        )
                    )
                continue

            if message_type == "vision.try_on.stop":
                try:
                    stopped = stop_try_on_session(
                        payload.get("sessionId"),
                        reason=payload.get("reason"),
                    )
                except ValueError as e:
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message=str(e),
                            retryable=False,
                            message_id=message_id,
                        )
                    continue
                except Exception as e:
                    logger.exception("Try-on stop failed")
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code="try_on_unavailable",
                                message=str(e),
                                retryable=True,
                                message_id=message_id,
                            )
                        )
                    continue

                mark_vision_session_tryon_stopped(stopped)
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.try_on.stopped",
                            message_id=f"try-on-stopped-{stopped['sessionId']}",
                            payload={
                                "sessionId": stopped["sessionId"],
                                "reason": stopped["reason"],
                            },
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
