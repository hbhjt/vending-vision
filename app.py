"""
FastAPI 应用入口模块

提供售货机视觉服务的 HTTP 和 WebSocket API：
- HTTP: 健康检查、版本信息、摄像头状态/快照、指标、调试仪表盘
- WebSocket: 画像推送协议（presence_status / profile_result / person_departed）
- 试衣 MJPEG 流: HTTP multipart 流式传输

核心工作循环：
presence_broadcast_loop() 独立轮询顶部摄像头并推送轻量状态；
profile_collection_worker() 按需运行中部摄像头画像采样，不阻塞来人/离开事件。
"""

from __future__ import annotations

import asyncio
import copy
import json
import ipaddress
import threading
from json import JSONDecodeError
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from vision.camera_manager import (
    get_all_camera_statuses,
    get_camera_config,
    get_camera_status,
    read_camera,
    release_all_cameras,
    reset_camera,
)
from vision.camera_owner import get_front_camera_owner
from vision.config import runtime_path, settings
from vision.logger import logger
from vision.metrics import metrics
from vision.presence_runtime import get_presence_runtime
from vision.profile_push import collect_front_profile_update, collect_profile_update
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

# 启动时的自检结果缓存
startup_check = None
# 调试仪表盘 HTML 文件路径
DASHBOARD_FILE = Path(runtime_path("dashboard/profile_dashboard.html"))


@app.on_event("startup")
def on_startup():
    """服务启动事件：运行自检并记录结果。"""
    global startup_check

    try:
        host = str(settings.HOST).strip().lower()
        if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
            raise RuntimeError("VISION_HOST must be a loopback address in production")
    except ValueError as exc:
        raise RuntimeError("VISION_HOST must be a numeric loopback address") from exc

    logger.info("Running vision module self check...")
    startup_check = run_self_check()

    if startup_check["ok"]:
        logger.info("Vision module self check passed")
    else:
        logger.warning(f"Vision module self check failed: {startup_check}")


@app.on_event("shutdown")
def on_shutdown():
    """服务关闭事件：停止后台工作线程并释放所有摄像头资源。"""
    for task in (_presence_worker_task, _profile_worker_task):
        if task is not None and not task.done():
            task.cancel()

    release_all_cameras()
    logger.info("Camera streams released")


def get_startup_check():
    """获取自检结果（懒执行：首次调用时运行）。"""
    global startup_check

    if startup_check is None:
        startup_check = run_self_check()

    return startup_check


def get_runtime_status():
    """获取运行时状态摘要。

    在 Mock 模式之外，还会实时检查摄像头连接状态。
    返回：cameraReady, modelReady, ageGenderReady, ageGenderMode 等信息。
    """
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

    runtime_check = dict(check)
    runtime_check["checks"] = checks
    return {
        "check": runtime_check,
        "cameraReady": camera_ready,
        "modelReady": model_ready,
        "ageGenderReady": age_gender_ready,
        "ageGenderMode": age_gender_mode,
    }


def validate_envelope(message):
    """验证 WebSocket 消息的外层封包格式。

    检查必填字段（protocol, type, messageId, timestamp, payload）
    及其类型是否正确。
    """
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
    """根据消息类型验证 payload 的字段格式。

    支持的验证：
    - vision.hello: protocolVersion, capabilities, clientRole, machineCode
    - vision.try_on.start: sessionId, catalogKey, variantId
    - vision.try_on.stop: sessionId, reason
    """
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
    """根路径：返回 API 索引和服务基本信息。"""
    return {
        "message": "Vending Vision Module is running",
        "api": {
            "dashboard": "/dashboard",
            "camera_roles_status": "/camera/roles/status",
            "front_camera_owner": "/camera/front/owner",
            "camera_snapshot": "/camera/{role}/snapshot.jpg",
            "proximity_debug": "/proximity/debug",
            "try_on_preview": "/try-on/{sessionId}.mjpeg",
            "session_status": "/session/status",
            "metrics": "/metrics",
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
        "profile_push": {
            "enabled": settings.PROFILE_PUSH_ENABLED,
            "push_interval_ms": settings.PROFILE_PUSH_INTERVAL_MS,
            "push_cooldown_ms": settings.PROFILE_PUSH_COOLDOWN_MS,
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
            "metrics": True,
            "mock_scenario": True,
        },
        "logging": {
            "level": settings.LOG_LEVEL,
            "file": settings.LOG_FILE,
            "max_bytes": settings.LOG_MAX_BYTES,
            "backup_count": settings.LOG_BACKUP_COUNT,
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


@app.get("/camera/{role}/snapshot.jpg")
def camera_role_snapshot(role: str):
    try:
        get_camera_config(role)
        if role == "front" and (
            get_front_camera_owner().get("owner") != "idle"
            or get_try_on_status().get("activeSessionId")
        ):
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": "front camera is reserved"},
            )
        image = read_camera(role, warmup_frames=1)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        if not ok:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": "failed to encode snapshot"},
            )

        return Response(content=encoded.tobytes(), media_type="image/jpeg")
    except ValueError as e:
        return JSONResponse(status_code=404, content={"ok": False, "error": str(e)})
    except Exception as e:
        logger.exception(f"HTTP /camera/{role}/snapshot.jpg failed")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/camera/{role}/reopen")
def camera_role_reopen(role: str):
    try:
        get_camera_config(role)
        if role == "front" and (
            get_front_camera_owner().get("owner") != "idle"
            or get_try_on_status().get("activeSessionId")
        ):
            return JSONResponse(
                status_code=409,
                content={"ok": False, "error": "front camera is reserved"},
            )
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


@app.get("/proximity/debug")
def proximity_debug():
    snapshot = get_presence_runtime().latest()
    if snapshot is None:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "error": "presence monitor has not produced a snapshot"},
        )
    return {"ok": True, "proximity": snapshot.get("proximity"), "snapshot": snapshot}


@app.get("/metrics")
def metrics_snapshot():
    return metrics.snapshot()


@app.get("/try-on/{session_id}.mjpeg")
def try_on_mjpeg(session_id: str, token: Optional[str] = None):
    if not token or not is_try_on_session_active(session_id, stream_token=token):
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "try-on session is not active"},
        )

    media_type = "multipart/x-mixed-replace; boundary=frame"
    return StreamingResponse(
        iter_try_on_mjpeg(session_id, stream_token=str(token)),
        media_type=media_type,
    )


# ---------------------------------------------------------------------------
# 画像广播子系统
# ---------------------------------------------------------------------------

# 已注册的 WebSocket 客户端（key: websocket id, value: client info）
_profile_clients: dict[int, dict] = {}
_profile_clients_lock = asyncio.Lock()
# Top-camera events and front-camera profiling intentionally run as separate
# tasks.  A slow front inference must never delay presence/departure events.
_presence_worker_task: asyncio.Task | None = None
_profile_worker_task: asyncio.Task | None = None
_profile_cancel_event: threading.Event | None = None


async def register_profile_client(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    capabilities: set[str],
    owned_try_on_session_ids: set,
):
    """Register a post-hello client and start the lightweight worker."""
    global _presence_worker_task

    async with _profile_clients_lock:
        existing = _profile_clients.pop(id(websocket), None)
        if existing and existing.get("sender_task"):
            existing["sender_task"].cancel()
        queue = asyncio.Queue(maxsize=max(int(settings.WEBSOCKET_QUEUE_SIZE), 1))
        client = {
            "websocket": websocket,
            "send_lock": send_lock,
            "capabilities": set(capabilities),
            "owned_try_on_session_ids": owned_try_on_session_ids,
            "owner_id": str(id(websocket)),
            "queue": queue,
        }
        client["sender_task"] = asyncio.create_task(profile_client_sender(client))
        _profile_clients[id(websocket)] = {
            **client,
        }
        metrics.set_gauge("profile_broadcast_clients", len(_profile_clients))

        if _presence_worker_task is None or _presence_worker_task.done():
            _presence_worker_task = asyncio.create_task(presence_broadcast_loop())
            logger.info("Presence broadcast worker started")


def cleanup_owned_try_on_sessions(session_ids: set, reason: str, owner_id: str | None = None):
    """清理客户端拥有的试衣会话（断开连接时调用）。"""
    for session_id in list(session_ids):
        try:
            stopped = stop_try_on_session(
                session_id, reason=reason, owner_id=owner_id,
            )
            mark_vision_session_tryon_stopped(stopped)
            session_ids.discard(session_id)
            logger.info(
                f"Stopped try-on session reason={reason}, sessionId={session_id}"
            )
        except Exception:
            logger.exception(
                f"Failed to stop try-on session reason={reason}, "
                f"sessionId={session_id}"
            )


async def unregister_profile_client(websocket: WebSocket):
    global _profile_cancel_event
    async with _profile_clients_lock:
        removed = _profile_clients.pop(id(websocket), None)
        no_clients = not _profile_clients

    if removed is not None:
        sender_task = removed.get("sender_task")
        if sender_task is not None and sender_task is not asyncio.current_task():
            sender_task.cancel()
        cleanup_owned_try_on_sessions(
            removed["owned_try_on_session_ids"],
            reason="websocket_disconnected",
            owner_id=removed.get("owner_id"),
        )
        async with _profile_clients_lock:
            metrics.set_gauge("profile_broadcast_clients", len(_profile_clients))
        if no_clients and _profile_cancel_event is not None:
            _profile_cancel_event.set()
        logger.info("Profile broadcast client unregistered")


async def profile_client_snapshot():
    async with _profile_clients_lock:
        return list(_profile_clients.values())


async def profile_worker_capabilities():
    clients = await profile_client_snapshot()
    capabilities = set()

    for client in clients:
        capabilities.update(client["capabilities"])

    return clients, capabilities


def filter_payload_for_client(payload: dict, capabilities: set[str]):
    if "ambient_light" in capabilities:
        return payload

    filtered = copy.deepcopy(payload)
    filtered.pop("ambientLight", None)
    return filtered


def should_deliver_profile_message(message_type: str, capabilities: set[str]):
    if message_type == "vision.profile_result":
        return "profile_push" in capabilities

    if message_type == "vision.presence_status":
        return "presence_status" in capabilities

    if message_type == "vision.person_departed":
        return "person_departed" in capabilities

    return True


async def profile_client_sender(client: dict):
    """Deliver one client's queue without stalling other subscribers."""
    websocket = client["websocket"]
    try:
        while True:
            message = await client["queue"].get()
            if message is None:
                return
            async with client["send_lock"]:
                await asyncio.wait_for(
                    websocket.send_json(message),
                    timeout=max(settings.WEBSOCKET_SEND_TIMEOUT_MS, 1) / 1000.0,
                )
            metrics.increment(
                "profile_broadcast_sent_total",
                message_type=message.get("type", "unknown"),
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        metrics.increment("profile_broadcast_send_failure_total")
        logger.exception("Profile client sender failed")
        await unregister_profile_client(websocket)


def queue_profile_message(client: dict, message_type: str, payload: dict):
    capabilities = client["capabilities"]

    if not should_deliver_profile_message(message_type, capabilities):
        return

    event_payload = filter_payload_for_client(payload, capabilities)
    message_prefix = {
        "vision.profile_result": "result",
        "vision.person_departed": "departure",
    }.get(message_type, "status")

    message = envelope(
        message_type=message_type,
        message_id=f"{message_prefix}-{event_payload['eventId']}-{uuid4()}",
        payload=event_payload,
    )
    try:
        client["queue"].put_nowait(message)
    except asyncio.QueueFull as exc:
        raise RuntimeError("profile client queue is full") from exc


async def broadcast_profile_update(update: dict):
    """向所有已注册客户端广播画像更新消息。

    发送失败时自动注销该客户端（清理过期连接）。"""
    clients = await profile_client_snapshot()
    stale_clients = []

    for client in clients:
        try:
            queue_profile_message(
                client,
                update["message_type"],
                update["payload"],
            )
        except Exception:
            stale_clients.append(client["websocket"])
            metrics.increment("profile_broadcast_send_failure_total")
            logger.exception("Profile broadcast send failed")

    for websocket in stale_clients:
        await unregister_profile_client(websocket)


async def broadcast_profile_error(code: str, message: str, retryable: bool = True):
    clients = await profile_client_snapshot()
    stale_clients = []

    for client in clients:
        try:
            client["queue"].put_nowait(
                error_envelope(code=code, message=message, retryable=retryable)
            )
        except asyncio.QueueFull:
            stale_clients.append(client["websocket"])
        except Exception:
            stale_clients.append(client["websocket"])
            metrics.increment("profile_broadcast_error_send_failure_total")
            logger.exception("Profile broadcast error send failed")

    for websocket in stale_clients:
        await unregister_profile_client(websocket)


async def profile_collection_worker(candidate, cancel_event: threading.Event):
    """Run slow front-camera sampling without blocking top-camera polling."""
    global _profile_worker_task, _profile_cancel_event
    runtime = get_presence_runtime()
    pushed = False
    try:
        update = await asyncio.to_thread(
            collect_front_profile_update,
            candidate.event_id,
            candidate.proximity,
            candidate.track,
            bool(candidate.proximity.get("close")),
            candidate.ambient_light,
            True,
            cancel_event,
            lambda: (not cancel_event.is_set() and runtime.is_candidate_valid(candidate.generation)),
            lambda: (runtime.latest() or {}).get("occupancy", {}),
            lambda: bool(
                ((runtime.latest() or {}).get("proximity") or {}).get("close")
            ),
        )
        # collect_front_profile_update validates the latest top-camera snapshot
        # immediately before committing a result.  A successful result then
        # locks the occupancy gate, so checking is_candidate_valid() again here
        # would incorrectly discard the just-committed profile.
        if update is not None and not cancel_event.is_set():
            pushed = update.get("message_type") == "vision.profile_result"
            await broadcast_profile_update(update)
    except asyncio.CancelledError:
        cancel_event.set()
        raise
    except Exception as exc:
        logger.exception("Profile collection worker failed")
        await broadcast_profile_error("internal_error", str(exc), retryable=True)
    finally:
        runtime.finish_collection(candidate.generation, pushed=pushed)
        _profile_cancel_event = None
        _profile_worker_task = None


async def presence_broadcast_loop():
    """Poll the top camera continuously and schedule profiling separately."""
    global _profile_worker_task, _profile_cancel_event
    while True:
        try:
            loop_started = asyncio.get_running_loop().time()
            clients, capabilities = await profile_worker_capabilities()

            if not clients:
                logger.info("Presence broadcast worker stopped: no clients")
                if _profile_cancel_event is not None:
                    _profile_cancel_event.set()
                return

            if settings.MOCK_SCENARIO != "off":
                mock_update = await asyncio.to_thread(
                    collect_profile_update,
                    "presence_status" in capabilities,
                    "ambient_light" in capabilities,
                    "person_departed" in capabilities,
                )
                if mock_update is not None:
                    await broadcast_profile_update(mock_update)
                await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)
                continue

            result = await asyncio.to_thread(
                get_presence_runtime().poll,
                "presence_status" in capabilities,
                "ambient_light" in capabilities,
                "person_departed" in capabilities,
            )
            metrics.observe_ms(
                "presence_worker_collect_duration_ms",
                (asyncio.get_running_loop().time() - loop_started) * 1000,
            )

            if _profile_cancel_event is not None and not get_presence_runtime().is_candidate_valid(
                getattr(_profile_worker_task, "candidate_generation", -1)
            ):
                _profile_cancel_event.set()

            if result.update is not None:
                metrics.increment(
                    "presence_worker_update_total",
                    message_type=result.update["message_type"],
                )
                await broadcast_profile_update(result.update)

            if (
                result.candidate is not None
                and "profile_push" in capabilities
                and (_profile_worker_task is None or _profile_worker_task.done())
            ):
                _profile_cancel_event = threading.Event()
                _profile_worker_task = asyncio.create_task(
                    profile_collection_worker(result.candidate, _profile_cancel_event)
                )
                _profile_worker_task.candidate_generation = result.candidate.generation

            await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)

        except RuntimeError as e:
            metrics.increment("presence_worker_error_total", code="camera_unavailable")
            await broadcast_profile_error(
                code="camera_unavailable",
                message=str(e),
                retryable=True,
            )
            logger.exception("Presence broadcast camera error")
            await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            metrics.increment("presence_worker_error_total", code="internal_error")
            await broadcast_profile_error(
                code="internal_error",
                message=str(e),
                retryable=True,
            )
            logger.exception("Presence broadcast loop failed")
            await asyncio.sleep(settings.PROFILE_PUSH_INTERVAL_MS / 1000.0)


async def profile_broadcast_loop():
    """Backward-compatible name for callers of the old worker entry point."""
    await presence_broadcast_loop()


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
            message_id=f"error-{message_id or 'server'}-{uuid4()}",
        )
    )


def websocket_origin_allowed(websocket: WebSocket) -> bool:
    """Accept native clients without Origin and allow configured local WebViews."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    origin_host = str(settings.HOST)
    if ":" in origin_host and not origin_host.startswith("["):
        origin_host = f"[{origin_host}]"
    default_origins = {
        f"http://{origin_host}:{settings.PORT}",
        f"http://localhost:{settings.PORT}",
    }
    allowed = set(settings.ALLOWED_ORIGINS) or default_origins
    return origin in allowed


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 主端点。

    处理以下消息类型：
    - vision.hello: 握手 + 能力协商 + 注册画像广播客户端
    - vision.ping: 心跳（回复 vision.pong）
    - vision.try_on.start: 启动试衣会话
    - vision.try_on.stop: 停止试衣会话
    - vision.start_profile / vision.cancel: 不支持（主动推送协议中不需要）

    断开连接时自动清理试衣会话和广播注册。
    """
    send_lock = asyncio.Lock()
    owned_try_on_session_ids = set()
    handshake_complete = False

    await websocket.accept()
    if not websocket_origin_allowed(websocket):
        await websocket.close(code=1008)
        logger.warning("WebSocket rejected due to untrusted Origin")
        return
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

            if message_type != "vision.hello" and not handshake_complete:
                async with send_lock:
                    await send_error(
                        websocket,
                        code="invalid_message",
                        message="vision.hello is required before business messages",
                        retryable=False,
                        message_id=message_id,
                    )
                continue

            if message_type == "vision.hello":
                status = get_runtime_status()
                client_capabilities = set(payload.get("capabilities") or [])

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
                            message_id=f"ready-{uuid4()}",
                            payload={
                                "serverName": "vem-vision-python",
                                "serverVersion": APP_VERSION,
                                "cameraReady": status["cameraReady"],
                                "modelReady": status["modelReady"],
                                "capabilities": server_capabilities,
                            },
                        )
                    )

                # The VEM daemon treats vision as a non-sale-critical capability.
                # Complete the protocol handshake even when cameras/models are
                # degraded so the runtime can expose diagnostics and retry after
                # the site is calibrated. Profile workers remain guarded by the
                # runtime checks and will publish a retryable error when needed.
                if not status["modelReady"] or not status["cameraReady"]:
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code=(
                                    "model_not_ready"
                                    if not status["modelReady"]
                                    else "camera_not_ready"
                                ),
                                message=(
                                    "required vision model is not ready"
                                    if not status["modelReady"]
                                    else "configured camera is not ready"
                                ),
                                retryable=True,
                                message_id=f"error-degraded-{uuid4()}",
                            )
                        )

                if settings.PROFILE_PUSH_ENABLED:
                    await register_profile_client(
                        websocket,
                        send_lock,
                        client_capabilities,
                        owned_try_on_session_ids,
                    )

                handshake_complete = True

                continue

            if message_type == "vision.ping":
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.pong",
                            message_id=f"pong-{message_id}-{uuid4()}",
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
                        owner_id=str(id(websocket)),
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
                except (PermissionError, RuntimeError) as e:
                    logger.exception("Try-on start failed")
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code="try_on_unavailable",
                                message=str(e),
                                retryable=True,
                                message_id=f"error-{message_id}-{uuid4()}",
                            )
                        )
                    continue

                if _profile_cancel_event is not None:
                    _profile_cancel_event.set()
                mark_vision_session_tryon_started(session)
                owned_try_on_session_ids.add(session["sessionId"])
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.try_on.started",
                            message_id=f"try-on-started-{session['sessionId']}-{uuid4()}",
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
                        owner_id=str(id(websocket)),
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
                except (PermissionError, RuntimeError) as e:
                    logger.exception("Try-on stop failed")
                    async with send_lock:
                        await websocket.send_json(
                            error_envelope(
                                code="try_on_unavailable",
                                message=str(e),
                                retryable=True,
                                message_id=f"error-{message_id}-{uuid4()}",
                            )
                        )
                    continue

                mark_vision_session_tryon_stopped(stopped)
                owned_try_on_session_ids.discard(stopped["sessionId"])
                async with send_lock:
                    await websocket.send_json(
                        envelope(
                            message_type="vision.try_on.stopped",
                            message_id=f"try-on-stopped-{stopped['sessionId']}-{uuid4()}",
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
        await unregister_profile_client(websocket)
        cleanup_owned_try_on_sessions(
            owned_try_on_session_ids,
            reason="websocket_disconnected",
            owner_id=str(id(websocket)),
        )
