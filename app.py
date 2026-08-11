"""
FastAPI 应用入口模块

提供售货机视觉服务的 HTTP 和 WebSocket API：
- HTTP: 健康检查、版本信息、摄像头状态/快照、指标、调试仪表盘、V2 试衣结果读取
- WebSocket: 画像推送协议（presence_status / profile_result / person_departed）与 V2 试衣尝试事件

核心工作循环：
presence_broadcast_loop() 独立轮询顶部摄像头并推送轻量状态；
profile_collection_worker() 按需运行中部摄像头画像采样，不阻塞来人/离开事件。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import ipaddress
import os
import re
import secrets
import threading
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from vision.camera_manager import (
    abort_all_camera_requests,
    abort_camera_request,
    get_all_camera_statuses,
    get_camera_config,
    get_camera_status,
    read_camera,
    read_camera_with_source,
    read_camera_with_source_async,
    release_all_cameras,
    reset_camera,
)
from vision.camera_binding import (
    CAMERA_MAINTENANCE_CONTRACT_VERSION,
    get_camera_maintenance,
)
from vision.camera_owner import get_front_camera_owner
from vision.camera_owner import (
    acquire_front_camera,
    release_front_camera,
    release_front_camera_io_lock,
    try_acquire_front_camera_io_lock,
)
from vision.config import runtime_path, settings
from vision.logger import logger
from vision.metrics import metrics
from vision.presence_runtime import get_presence_runtime
from vision.profile_push import collect_front_profile_update, collect_profile_update
from vision.protocol import APP_VERSION, PROTOCOL, envelope, error_envelope
from vision.self_check import check_camera, run_self_check
from vision.session_state import get_vision_session_status
from vision.fast_tryon import FastTryOnRuntime, GarmentFetchError, PoseUnavailableError
from vision.fast_attempt_registry import AttemptReceipt, FastAttemptRegistry, TerminalTransition
from vision.attempt_worker import FastRenderBroker, render_attempt_frame
from vision.acquisition_preview import AcquisitionPreviewStore
from vision.acquisition_observer import AcquisitionObservationWorker
from vision.v2_contract_bundle import (
    V2ContractBundleUnavailable,
    load_v2_contract_identity,
    parse_v2_client_message,
    parse_v2_server_message,
)
from vision.ai_model_pack import official_ai_readiness


app = FastAPI(
    title=settings.APP_NAME,
    description="Smart vending machine vision API",
    version=APP_VERSION,
)

_FAST_RESULT_TTL_SECONDS = 5 * 60
_FAST_RESULT_MAX_BYTES = 16 * 1024 * 1024
_FAST_RESULT_MAX_COUNT = 1000
_FAST_RESULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_FAST_ATTEMPT_TIMEOUT_SECONDS = 15
_FAST_ATTEMPT_MAX_TASKS = 2
_FAST_TERMINAL_SEND_TIMEOUT_SECONDS = 0.25
_fast_runtime = FastTryOnRuntime()
_fast_attempt_registry = FastAttemptRegistry(
    terminal_ttl_seconds=_FAST_RESULT_TTL_SECONDS,
    result_max_count=_FAST_RESULT_MAX_COUNT,
    result_max_bytes=_FAST_RESULT_MAX_TOTAL_BYTES,
    result_single_max_bytes=_FAST_RESULT_MAX_BYTES,
)
_fast_attempt_task_slots = asyncio.Semaphore(_FAST_ATTEMPT_MAX_TASKS)
_fast_render_broker = FastRenderBroker()
_acquisition_previews = AcquisitionPreviewStore()
_acquisition_observer: AcquisitionObservationWorker | None = None
_ACQUISITION_STABLE_FRAMES = 3
_ACQUISITION_POLL_SECONDS = 0.05

# 启动时的自检结果缓存
startup_check = None
# 调试仪表盘 HTML 文件路径
DASHBOARD_FILE = Path(runtime_path("dashboard/profile_dashboard.html"))
@app.on_event("startup")
async def on_startup():
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
    try:
        await _fast_render_broker.start()
    except Exception:
        # Fast is an enhancement capability.  A failed broker readiness probe
        # keeps the service alive but is reflected truthfully in V2 readiness.
        logger.exception("Fast render broker failed startup readiness")
    try:
        await _get_acquisition_observer().start()
    except Exception:
        # Acquisition stays fail-closed until the production observation
        # boundary can be prewarmed; ordinary Vision capability remains alive.
        logger.exception("Acquisition observer failed startup readiness")


@app.on_event("shutdown")
async def on_shutdown():
    """服务关闭事件：停止后台工作线程并释放所有摄像头资源。"""
    for task in (_presence_worker_task, _profile_worker_task):
        if task is not None and not task.done():
            task.cancel()

    _fast_render_broker.quiesce()
    preview_shutdown_error = None
    try:
        await _acquisition_previews.close()
    except Exception as exc:
        preview_shutdown_error = exc
        logger.warning("Acquisition preview shutdown close failed: %s", exc)
    await _fast_attempt_registry.shutdown()
    observer = _acquisition_observer
    if observer is not None:
        await observer.shutdown()
    render_shutdown_error = None
    try:
        await _fast_render_broker.shutdown()
    except Exception as exc:
        render_shutdown_error = exc
    aborted = await abort_all_camera_requests(reason="vision_shutdown")
    released = release_all_cameras()
    if (
        render_shutdown_error is not None
        or preview_shutdown_error is not None
        or not all(aborted.values())
        or not all(released.values())
    ):
        raise RuntimeError(
            "runtime broker shutdown incomplete: "
            f"render={render_shutdown_error}, preview={preview_shutdown_error}, "
            f"aborted={aborted}, released={released}"
        )
    logger.info("Camera streams released")


def _discard_completed_fast_attempt(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Fast attempt task ended with an unhandled error")


async def _await_cleanup_uncancelled(awaitable):
    """Finish one cleanup await even when the transport scope is repeatedly cancelled."""
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _cancel_disconnect_owner_and_publish(receipt: AttemptReceipt) -> None:
    transition = await _fast_attempt_registry.cancel_owner_and_join(
        receipt,
        _generated_v2_envelope(
            "vision.try_on.attempt.canceled",
            {"attemptId": receipt.attempt_id, "reason": "disconnect"},
        ),
    )
    await _publish_fast_transition(transition)


async def _run_owned_attempt_step(receipt: AttemptReceipt, operation, *, timeout: float):
    """Race an owned process operation with cancellation and join it in all cases."""
    if not await _fast_attempt_registry.is_current(receipt):
        raise GarmentFetchError("attempt_canceled")
    worker = asyncio.create_task(operation)
    cancel_waiter = asyncio.create_task(
        (await _fast_attempt_registry.cancel_event_for(receipt)).wait()
    )
    try:
        done, _ = await asyncio.wait(
            {worker, cancel_waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if worker in done:
            return worker.result()
        if cancel_waiter in done:
            raise GarmentFetchError("attempt_canceled")
        raise asyncio.TimeoutError()
    except (asyncio.TimeoutError, asyncio.CancelledError, GarmentFetchError):
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        raise
    finally:
        cancel_waiter.cancel()
        await asyncio.gather(cancel_waiter, return_exceptions=True)
        if not await _fast_attempt_registry.is_current(receipt):
            raise GarmentFetchError("attempt_canceled")


async def _acquire_front_io_until(deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        if try_acquire_front_camera_io_lock():
            return
        await asyncio.sleep(0.002)
    raise asyncio.TimeoutError()


async def _read_attempt_front_frame(
    receipt: AttemptReceipt, *, timeout: float, lease_token: str | None = None
):
    """Read one V2 acquisition frame through the production front-camera lane."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    owner_acquired = False
    io_acquired = False
    managed_lease = lease_token is None
    lease_token = lease_token or f"try-on:{receipt.attempt_id}:{receipt.generation}:{receipt.owner_token}"
    try:
        await _acquire_front_io_until(deadline)
        io_acquired = True
        if not await _fast_attempt_registry.is_current(receipt):
            raise GarmentFetchError("attempt_canceled")
        if managed_lease:
            owner = acquire_front_camera(
                "try_on_attempt",
                reason=f"try_on_acquisition:{receipt.attempt_id}",
                lease_token=lease_token,
            )
            if not owner.get("ok"):
                raise GarmentFetchError(owner.get("error") or "front_camera_busy")
            owner_acquired = True

        if getattr(read_camera_with_source, "__module__", "") != "vision.camera_manager":
            return read_camera_with_source("front", 1)

        # All production consumers enter through camera_manager.  Physical
        # DirectShow pipe waits run on one dedicated request thread, while the
        # event loop remains available to process replacement and disconnect.
        read_task = asyncio.create_task(
            read_camera_with_source_async(
                "front", 1, timeout=max(0.001, deadline - loop.time())
            )
        )
        cancel_waiter = asyncio.create_task(
            (await _fast_attempt_registry.cancel_event_for(receipt)).wait()
        )
        try:
            done, _ = await asyncio.wait(
                {read_task, cancel_waiter},
                timeout=max(0.001, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if read_task in done:
                return read_task.result()

            abort_reason = (
                "try_on_attempt_canceled" if cancel_waiter in done else "try_on_attempt_timeout"
            )
            abort_task = asyncio.create_task(
                abort_camera_request("front", reason=abort_reason)
            )
            read_task.cancel()
            dead = await abort_task
            await asyncio.gather(read_task, return_exceptions=True)
            if not dead:
                dead = await abort_camera_request(
                    "front", reason=f"{abort_reason}_confirm"
                )
            if not dead:
                raise RuntimeError("front camera broker remained alive after abort")
            if cancel_waiter in done:
                raise GarmentFetchError("attempt_canceled")
            raise asyncio.TimeoutError()
        except asyncio.CancelledError:
            abort_task = asyncio.create_task(
                abort_camera_request("front", reason="try_on_attempt_cancelled")
            )
            read_task.cancel()
            dead = await _await_cleanup_uncancelled(abort_task)
            await _await_cleanup_uncancelled(
                asyncio.gather(read_task, return_exceptions=True)
            )
            if not dead:
                dead = await _await_cleanup_uncancelled(
                    abort_camera_request("front", reason="try_on_attempt_cancelled_confirm")
                )
            if not dead:
                raise RuntimeError("front camera broker remained alive after cancel")
            raise
        finally:
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)
    finally:
        if io_acquired:
            release_front_camera_io_lock()
        if owner_acquired:
            release_front_camera(
                "try_on_attempt",
                reason=f"try_on_acquisition_done:{receipt.attempt_id}",
                lease_token=lease_token,
            )


async def _publish_fast_transition(transition: TerminalTransition | None) -> None:
    """Deliver a registry-won terminal to every still-live subscriber."""
    if transition is None:
        return
    async def send_one(subscriber) -> None:
        try:
            await _send_json_bounded(
                subscriber.websocket,
                subscriber.send_lock,
                transition.message,
                timeout=_FAST_TERMINAL_SEND_TIMEOUT_SECONDS,
            )
        except Exception:
            await _fast_attempt_registry.detach_subscriber(subscriber.websocket)

    await asyncio.gather(
        *(send_one(subscriber) for subscriber in transition.subscribers),
        return_exceptions=True,
    )


async def _send_json_bounded(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    message: dict,
    *,
    timeout: float | None = None,
) -> None:
    """Bound acquiring the per-connection send lock and the actual send."""
    send_timeout = (
        timeout
        if timeout is not None
        else max(settings.WEBSOCKET_SEND_TIMEOUT_MS, 1) / 1000.0
    )

    async def locked_send() -> None:
        async with send_lock:
            await websocket.send_json(message)

    await asyncio.wait_for(locked_send(), timeout=max(send_timeout, 0.001))


def get_startup_check():
    """获取自检结果（懒执行：首次调用时运行）。"""
    global startup_check

    if startup_check is None:
        startup_check = run_self_check()

    return startup_check


def _fast_result_reference(attempt_id: str, token: str) -> str:
    host = str(settings.HOST)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{settings.PORT}/v2/try-on/results/{attempt_id}?token={token}"


def _acquisition_preview_reference(token: str) -> str:
    host = str(settings.HOST)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{settings.PORT}/v2/try-on/acquisition/preview.mjpeg?token={token}"


def _get_acquisition_observer() -> AcquisitionObservationWorker:
    global _acquisition_observer
    if _acquisition_observer is None:
        _acquisition_observer = AcquisitionObservationWorker()
    return _acquisition_observer


def _acquisition_observer_ready() -> bool:
    observer = _acquisition_observer
    return bool(
        observer is not None
        and getattr(observer, "ready", False)
        and getattr(observer, "fatal_error", None) is None
    )


def _acquiring_message(attempt_id: str, token: str, occupancy: str, aligned: bool, stable: bool) -> dict:
    if occupancy == "none":
        guidance, manual = "no_person", False
    elif occupancy == "multiple":
        guidance, manual = "multiple_people", False
    elif not aligned:
        guidance, manual = "align", False
    elif stable:
        guidance, manual = "ready", False
    else:
        guidance, manual = "hold_still", True
    return _generated_v2_envelope("vision.try_on.attempt.acquiring", {
        "attemptId": attempt_id,
        "preview": {"reference": _acquisition_preview_reference(token), "streamType": "mjpeg"},
        "occupancy": occupancy,
        "guidance": guidance,
        "manualCaptureAllowed": manual,
    })


def _prepare_fast_result(attempt_id: str, image: bytes) -> tuple[dict, dict]:
    if len(image) > _FAST_RESULT_MAX_BYTES:
        raise RuntimeError("fast_result_too_large")
    png_signature = b"\x89PNG\r\n\x1a\n"
    if len(image) < 33 or image[:8] != png_signature or image[12:16] != b"IHDR":
        raise RuntimeError("fast_result_invalid_png")
    width = int.from_bytes(image[16:20], "big")
    height = int.from_bytes(image[20:24], "big")
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise RuntimeError("fast_result_dimensions_too_large")
    if width * height > 8192 * 8192:
        raise RuntimeError("fast_result_dimensions_too_large")
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3:
        raise RuntimeError("fast_result_invalid_png")
    if decoded.shape[1] != width or decoded.shape[0] != height:
        raise RuntimeError("fast_result_invalid_png")
    token = secrets.token_urlsafe(32)
    stored = {
        "token": token,
        "bytes": image,
        "reference": _fast_result_reference(attempt_id, token),
        "digest": f"sha256:{hashlib.sha256(image).hexdigest()}",
        "contentType": "image/png",
        "byteSize": len(image),
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
    }
    public = {key: value for key, value in stored.items() if key not in {"token", "bytes"}}
    return stored, public


def _generated_v2_envelope(message_type: str, payload: dict) -> dict:
    parsed = parse_v2_server_message(
        envelope(message_type=message_type, message_id=str(uuid4()), payload=payload)
    )
    if parsed.type != message_type:
        raise ValueError("invalid_v2_boundary_message")
    return parsed.model_dump(mode="json")


@app.api_route("/v2/try-on/results/{attempt_id}", methods=["GET", "HEAD"])
async def read_fast_result(request: Request, attempt_id: str, token: Optional[str] = None):
    """Serve only an unguessable, disposable local PNG result read grant."""
    raw_query = request.scope.get("query_string", b"")
    if not isinstance(raw_query, bytes) or not re.fullmatch(
        rb"token=[A-Za-z0-9_-]{1,128}", raw_query
    ):
        raise HTTPException(status_code=404, detail="result not found")
    try:
        canonical_token = raw_query[6:].decode("ascii")
    except UnicodeDecodeError:
        raise HTTPException(status_code=404, detail="result not found")
    result = await _fast_attempt_registry.get_result(attempt_id, canonical_token)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")
    image = result["bytes"]
    return Response(
        content=b"" if request.method == "HEAD" else image,
        media_type="image/png",
        headers={"Content-Length": str(len(image)), "Cache-Control": "no-store"},
    )


@app.api_route("/v2/try-on/acquisition/preview.mjpeg", methods=["GET", "HEAD"])
async def read_acquisition_preview(request: Request, token: Optional[str] = None):
    """Display-only attempt preview; input frames never originate from HTTP bytes."""
    raw_query = request.scope.get("query_string", b"")
    if not isinstance(raw_query, bytes) or not re.fullmatch(rb"token=[A-Za-z0-9_-]{1,128}", raw_query):
        raise HTTPException(status_code=404, detail="preview not found")
    canonical_token = raw_query[6:].decode("ascii")
    snapshot = await _acquisition_previews.get(canonical_token)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="preview not found")
    media_type = "multipart/x-mixed-replace; boundary=frame"
    if request.method == "HEAD":
        return Response(content=b"", media_type=media_type, headers={"Cache-Control": "no-store"})

    try:
        lease = await _acquisition_previews.acquire(canonical_token)
    except RuntimeError as exc:
        if str(exc) == "acquisition_preview_reader_limit":
            raise HTTPException(status_code=429, detail="preview busy") from exc
        raise
    if lease is None:
        raise HTTPException(status_code=404, detail="preview not found")

    async def frames():
        try:
            await _acquisition_previews.register_task(lease.lease_id, asyncio.current_task())
            current = lease.snapshot
            while current is not None:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(current.jpeg)).encode("ascii")
                    + b"\r\n\r\n" + current.jpeg + b"\r\n"
                )
                current = await _acquisition_previews.wait_for_change(canonical_token, current.jpeg)
        finally:
            # ASGI body close/disconnect runs the generator finalizer, returning
            # the bounded lease even when the client drops mid-frame.
            await _acquisition_previews.release(lease.lease_id)

    return StreamingResponse(frames(), media_type=media_type, headers={"Cache-Control": "no-store"})


def get_runtime_status():
    """获取运行时状态摘要。

    摄像头状态来自启动时的稳定角色枚举，不在健康请求中打开设备；
    真实取帧由维护 test/confirm 和业务读取路径负责。
    返回：cameraReady, modelReady, ageGenderReady, ageGenderMode 等信息。
    """
    check = get_startup_check()
    checks = dict(check["checks"])
    checks["camera"] = check_camera()

    camera_ready = checks["camera"]["ok"]
    model_ready = (
        checks["modelManifest"]["ok"]
        and checks["pose"]["ok"]
        and checks["face"]["ok"]
        and checks["person"]["modelReady"]
        and checks["ageGender"]["modelReady"]
    )
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
        "fastRenderReady": _fast_render_broker.ready,
        "fastPoseReady": _fast_render_broker.pose_ready,
        "acquisitionObserverReady": _acquisition_observer_ready(),
    }


def build_v2_ready_message(hello: dict, status: dict) -> tuple[dict, set[str]]:
    """Return a generated-boundary V2 ready envelope without degrading core Vision."""
    try:
        parsed_hello = parse_v2_client_message(hello)
        if parsed_hello.type != "vision.hello":
            raise ValueError("invalid_v2_boundary_message")
        payload = parsed_hello.payload.model_dump()
        client_capabilities = payload["capabilities"]
    except V2ContractBundleUnavailable:
        # Without the generated parser there is no safe hello fallback.
        raise
    except ValueError:
        raise

    try:
        identity = load_v2_contract_identity()
    except V2ContractBundleUnavailable:
        identity = None

    if identity is None:
        diagnostic = "contract_bundle_unavailable"
        schema_version = "unavailable"
        bundle_version = "unavailable"
        contract_digest = "0" * 64
    elif (
        payload["schemaVersion"] != identity.schema_version
        or payload["bundleVersion"] != identity.bundle_version
    ):
        diagnostic = "contract_version_mismatch"
        schema_version = identity.schema_version
        bundle_version = identity.bundle_version
        contract_digest = identity.contract_digest
    elif payload["contractDigest"] != identity.contract_digest:
        diagnostic = "contract_digest_mismatch"
        schema_version = identity.schema_version
        bundle_version = identity.bundle_version
        contract_digest = identity.contract_digest
    elif not status["cameraReady"]:
        diagnostic = "camera_unavailable"
        schema_version = identity.schema_version
        bundle_version = identity.bundle_version
        contract_digest = identity.contract_digest
    elif (
        not status.get("fastRenderReady", True)
        or not status.get("fastPoseReady", True)
        or not status.get("acquisitionObserverReady", True)
    ):
        # The frozen V2 diagnostic vocabulary has one local Vision capability
        # unavailable value.  Do not extend that cross-repository contract in
        # this worker-only slice.
        diagnostic = "camera_unavailable"
        schema_version = identity.schema_version
        bundle_version = identity.bundle_version
        contract_digest = identity.contract_digest
    else:
        diagnostic = "ready"
        schema_version = identity.schema_version
        bundle_version = identity.bundle_version
        contract_digest = identity.contract_digest

    ready = envelope(
        message_type="vision.ready",
        message_id=str(uuid4()),
        payload={
            "serverName": "vem-vision-python",
            "serverVersion": APP_VERSION,
            "schemaVersion": schema_version,
            "bundleVersion": bundle_version,
            "contractDigest": contract_digest,
            "cameraReady": status["cameraReady"],
            "fastReady": diagnostic == "ready",
            # The separate pack is optional for core/Fast.  This lightweight
            # verifier never loads model weights or performs inference.
            "aiReady": diagnostic == "ready" and official_ai_readiness(os.environ.get("VEM_AI_MODEL_PACK")),
            "visionBusinessReady": diagnostic == "ready",
            "businessReadinessDiagnostic": diagnostic,
            "capabilities": [
                "profile_push",
                "presence_status",
                "person_departed",
                "ambient_light",
                "try_on_fast",
                *( ["try_on_ai"] if diagnostic == "ready" and official_ai_readiness(os.environ.get("VEM_AI_MODEL_PACK")) else [] ),
            ],
        },
    )
    # This is the server-to-machine strict generated boundary validation even
    # when the identity manifest is unavailable.  The latter is a degraded
    # business capability only if the generated parser itself remains usable.
    parsed_ready = parse_v2_server_message(ready)
    return parsed_ready.model_dump(), set(client_capabilities)


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

    if len(message["messageId"]) > 128:
        return "messageId must not exceed 128 characters"

    try:
        timestamp = datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            return "timestamp must include a timezone"
    except ValueError:
        return "timestamp must be an ISO datetime"

    if not isinstance(message.get("payload"), dict):
        return "payload must be an object"

    return None


def validate_message_payload(message_type: str, payload: dict):
    """根据消息类型验证 payload 的字段格式。

    支持的验证：
    - vision.ping: empty payload
    """
    supported_types = {"vision.ping"}
    if message_type not in supported_types:
        return f"unsupported client message type: {message_type}"

    return None


@app.get("/")
def root():
    """根路径：返回 API 索引和服务基本信息。"""
    return {
        "message": "Vending Vision Module is running",
        "api": {
            "dashboard": "/dashboard",
            "camera_roles_status": "/camera/roles/status",
            "camera_maintenance": "/maintenance/cameras",
            "front_camera_owner": "/camera/front/owner",
            "proximity_debug": "/proximity/debug",
            "session_status": "/session/status",
            "metrics": "/metrics",
            "ws": "/ws",
            "diagnostic_ws": "/debug/ws",
            "health": "/health",
            "version": "/version",
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    if not settings.DEVELOPMENT_DASHBOARD_ENABLED:
        return HTMLResponse(status_code=404, content="<h1>development dashboard disabled</h1>")
    if not DASHBOARD_FILE.exists():
        return HTMLResponse(
            status_code=404,
            content="<h1>dashboard not found</h1>",
        )

    return HTMLResponse(
        content=DASHBOARD_FILE.read_text(encoding="utf-8"),
    )


@app.get("/debug/contract-bundle")
def debug_contract_bundle():
    """Expose only the generated V2 identity needed by the diagnostic client."""
    try:
        identity = load_v2_contract_identity()
    except V2ContractBundleUnavailable as exc:
        raise HTTPException(status_code=503, detail="contract_bundle_unavailable") from exc
    return {
        "protocol": identity.protocol,
        "schemaVersion": identity.schema_version,
        "bundleVersion": identity.bundle_version,
        "contractDigest": identity.contract_digest,
    }


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
            "top": {key: value for key, value in settings.TOP_CAMERA_CONFIG.items() if key != "index"},
            "front": {key: value for key, value in settings.FRONT_CAMERA_CONFIG.items() if key != "index"},
            "front_owner": get_front_camera_owner(),
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


def _maintenance_error(exc: Exception, status_code: int = 409):
    return JSONResponse(
        status_code=status_code,
        content={"contractVersion": CAMERA_MAINTENANCE_CONTRACT_VERSION,
                 "error": {"code": type(exc).__name__, "message": str(exc)}},
    )


@app.get("/maintenance/cameras")
def camera_maintenance_contract():
    """Versioned loopback contract; device identities stay opaque to VEM."""
    return get_camera_maintenance().contract()


@app.post("/maintenance/cameras/refresh")
def camera_maintenance_refresh():
    get_camera_maintenance().refresh()
    return get_camera_maintenance().contract()


@app.get("/maintenance/cameras/{candidate_id}/preview.jpg")
def camera_maintenance_preview(candidate_id: str):
    try:
        return Response(
            content=get_camera_maintenance().preview(candidate_id),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except (ValueError, RuntimeError) as exc:
        return _maintenance_error(exc)


@app.post("/maintenance/cameras/{role}/test")
def camera_maintenance_test(role: str, payload: dict = Body(...)):
    try:
        if set(payload) != {"candidateId"}:
            raise ValueError("test request must contain only candidateId")
        candidate_id = payload.get("candidateId")
        if not isinstance(candidate_id, str):
            raise ValueError("candidateId is required")
        return get_camera_maintenance().test(role, candidate_id)
    except (ValueError, RuntimeError) as exc:
        return _maintenance_error(exc)


@app.post("/maintenance/cameras/{role}/confirm")
def camera_maintenance_confirm(role: str, payload: dict = Body(...)):
    try:
        required = {"candidateId", "testEvidenceId", "operatorVisualConfirmation", "expectedGeneration"}
        if set(payload) != required:
            raise ValueError("confirm request must contain candidateId, testEvidenceId, operatorVisualConfirmation and expectedGeneration")
        candidate_id = payload.get("candidateId")
        if not isinstance(candidate_id, str):
            raise ValueError("candidateId is required")
        test_evidence_id = payload.get("testEvidenceId")
        expected_generation = payload.get("expectedGeneration")
        visual = payload.get("operatorVisualConfirmation")
        if not isinstance(test_evidence_id, str) or not test_evidence_id:
            raise ValueError("confirm requires testEvidenceId")
        if visual is not True:
            raise ValueError("confirm requires explicit operatorVisualConfirmation")
        if not isinstance(expected_generation, str) or not expected_generation:
            raise ValueError("confirm requires expectedGeneration")
        return get_camera_maintenance().confirm(
            role, candidate_id, test_evidence_id=test_evidence_id,
            operator_visual_confirmation=visual, expected_generation=expected_generation,
        )
    except (ValueError, RuntimeError) as exc:
        return _maintenance_error(exc)


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
    # This legacy diagnostic is intentionally absent from managed production.
    # Managed production camera bytes use the plain loopback v2 maintenance preview route.
    if not settings.DEVELOPMENT_DASHBOARD_ENABLED:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
    try:
        get_camera_config(role)
        if role == "front" and get_front_camera_owner().get("owner") != "idle":
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
    if not settings.DEVELOPMENT_DASHBOARD_ENABLED:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
    try:
        get_camera_config(role)
        if role == "front" and get_front_camera_owner().get("owner") != "idle":
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


async def unregister_profile_client(websocket: WebSocket):
    global _profile_cancel_event
    async with _profile_clients_lock:
        removed = _profile_clients.pop(id(websocket), None)
        no_clients = not _profile_clients

    if removed is not None:
        sender_task = removed.get("sender_task")
        if sender_task is not None and sender_task is not asyncio.current_task():
            sender_task.cancel()
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
    message = envelope(
        message_type=message_type,
        message_id=str(uuid4()),
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


async def _cancel_active_attempt(reason: str) -> None:
    """Top-camera departure fences the current try-on without stopping top work."""
    attempt_id = await _fast_attempt_registry.active_attempt_id()
    if attempt_id is None:
        return
    await _publish_fast_transition(await _cancel_v2_attempt(
        attempt_id=attempt_id,
        terminal=_generated_v2_envelope(
            "vision.try_on.attempt.canceled",
            {"attemptId": attempt_id, "reason": reason},
        ),
    ))


async def _cancel_v2_attempt(*, attempt_id: str, terminal: dict) -> TerminalTransition | None:
    """Make cancellation immediately relinquish acquisition-only capabilities."""
    transition = await _fast_attempt_registry.cancel_current(
        attempt_id=attempt_id, terminal=terminal
    )
    if transition is None:
        return None
    cleanup_errors = []
    try:
        await _acquisition_previews.close(attempt_id)
    except Exception as exc:
        cleanup_errors.append(f"preview_close:{type(exc).__name__}:{exc}")
    owner = get_front_camera_owner()
    token = owner.get("leaseToken")
    if owner.get("owner") == "try_on_attempt" and isinstance(token, str) and token.startswith(
        f"try-on:{attempt_id}:"
    ):
        release = release_front_camera(
            "try_on_attempt", reason=f"try_on_canceled:{attempt_id}", lease_token=token
        )
        if not release.get("ok"):
            cleanup_errors.append(f"front_release:{release.get('error')}")
    if cleanup_errors:
        logger.warning(
            "V2 attempt cancellation cleanup completed with errors attemptId=%s errors=%s",
            attempt_id,
            cleanup_errors,
        )
    return transition


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
                    if mock_update.get("message_type") == "vision.person_departed":
                        await _cancel_active_attempt("departure")
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
                if result.update["message_type"] == "vision.person_departed":
                    await _cancel_active_attempt("departure")
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


async def _release_acquisition_resources(
    receipt: AttemptReceipt, lease_token: str
) -> list[str]:
    """Release all acquisition resources and report, never throw, cleanup errors."""
    errors: list[str] = []
    try:
        await _acquisition_previews.close(receipt.attempt_id)
    except Exception as exc:
        errors.append(f"preview_close:{type(exc).__name__}:{exc}")
    try:
        released = release_front_camera(
            "try_on_attempt",
            reason=f"try_on_acquisition_done:{receipt.attempt_id}",
            lease_token=lease_token,
        )
        if not released.get("ok"):
            errors.append(f"front_release:{released.get('error')}")
    except Exception as exc:
        errors.append(f"front_release:{type(exc).__name__}:{exc}")
    if errors:
        logger.warning(
            "V2 acquisition cleanup completed with errors attemptId=%s errors=%s",
            receipt.attempt_id,
            errors,
        )
    return errors


async def run_v2_fast_attempt(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    message: dict,
    fast_ready: bool,
    owned_fast_attempt_receipts: set[AttemptReceipt],
    connection_closed: asyncio.Event,
) -> None:
    """Run one bounded Fast attempt without holding a WS or store lock on I/O."""
    try:
        parsed = parse_v2_client_message(message)
        if parsed.type != "vision.try_on.attempt.start":
            raise ValueError("invalid_v2_boundary_message")
        payload = parsed.payload.model_dump()
    except (V2ContractBundleUnavailable, ValueError):
        async with send_lock:
            await send_error(
                websocket,
                code="invalid_message",
                message="invalid_v2_boundary_message",
                retryable=False,
                message_id=message.get("messageId"),
            )
        return

    # Handshake readiness is a negotiation fact, not a lifetime lease on the
    # render process.  Every attempt must also observe the live broker before
    # constructing admission replay.
    fast_ready = bool(
        fast_ready
        and _fast_render_broker.ready
        and _fast_render_broker.pose_ready
        and _acquisition_observer_ready()
    )
    attempt_id = payload["attemptId"]
    # An AI selection can never silently route through Fast.  Until the
    # official attempt child is ready it receives only the AI-specific
    # terminal, leaving Fast and ordinary Vision independent.
    if payload["mode"] != "fast":
        await _send_json_bounded(
            websocket,
            send_lock,
            _generated_v2_envelope(
                "vision.try_on.attempt.failed",
                {"attemptId": attempt_id, "reason": "ai_unavailable"},
            ),
        )
        return
    unavailable_terminal = _generated_v2_envelope(
        "vision.try_on.attempt.failed",
        {"attemptId": attempt_id, "reason": "fast_unavailable"},
    )
    canceled_terminal = _generated_v2_envelope(
        "vision.try_on.attempt.canceled",
        {"attemptId": attempt_id, "reason": "replaced"},
    )
    accepted = (
        _generated_v2_envelope(
            "vision.try_on.attempt.accepted", {"attemptId": attempt_id, "mode": "fast"}
        )
        if fast_ready
        else None
    )
    preparation = await _fast_attempt_registry.prepare_admission(
        attempt_id=attempt_id,
        websocket=websocket,
        send_lock=send_lock,
        task=asyncio.current_task(),
        canceled_terminal=canceled_terminal,
        owner_receipts=owned_fast_attempt_receipts,
    )
    for transition in preparation.transitions:
        await _publish_fast_transition(transition)
    admission = await _fast_attempt_registry.commit_prepared_admission(
        preparation,
        accepted=accepted,
        generating=None,
        unavailable_terminal=unavailable_terminal,
        readiness=lambda: bool(
            fast_ready
            and _fast_render_broker.ready
            and _fast_render_broker.pose_ready
            and _acquisition_observer_ready()
        ),
    )
    if not admission.is_owner:
        for replay_message in admission.replay:
            await _send_json_bounded(websocket, send_lock, replay_message)
        return
    receipt = admission.receipt
    assert receipt is not None
    if connection_closed.is_set():
        await _publish_fast_transition(
            await _fast_attempt_registry.cancel_owner_and_join(receipt)
        )
        return
    stored_result = None
    lease_token = f"try-on:{receipt.attempt_id}:{receipt.generation}:{receipt.owner_token}"
    lease_acquired = False
    try:
        for replay_message in admission.replay:
            await _send_json_bounded(websocket, send_lock, replay_message)
        await _acquire_front_io_until(asyncio.get_running_loop().time() + _FAST_ATTEMPT_TIMEOUT_SECONDS)
        try:
            owner = acquire_front_camera(
                "try_on_attempt", reason=f"try_on_acquisition:{attempt_id}", lease_token=lease_token
            )
        finally:
            release_front_camera_io_lock()
        if not owner.get("ok"):
            raise GarmentFetchError(owner.get("error") or "front_camera_busy")
        lease_acquired = True

        stable_frames = 0
        last_guidance = None
        captured_frame = None
        source_frame = None
        deadline = asyncio.get_running_loop().time() + _FAST_ATTEMPT_TIMEOUT_SECONDS
        preview_token = None
        while asyncio.get_running_loop().time() < deadline:
            remaining = max(0.001, deadline - asyncio.get_running_loop().time())
            frame, source = await _read_attempt_front_frame(
                receipt, timeout=remaining, lease_token=lease_token
            )
            observation = await _get_acquisition_observer().observe(frame, timeout=remaining)
            jpeg = observation.jpeg
            if preview_token is None:
                preview_token = await _acquisition_previews.open(attempt_id, jpeg)
            else:
                await _acquisition_previews.update(attempt_id, preview_token, jpeg)
            occupancy, aligned = observation.occupancy, observation.aligned
            stable_frames = stable_frames + 1 if occupancy == "single" and aligned else 0
            stable = stable_frames >= _ACQUISITION_STABLE_FRAMES
            acquiring = _acquiring_message(attempt_id, preview_token, occupancy, aligned, stable)
            if acquiring["payload"]["guidance"] != last_guidance:
                await _publish_fast_transition(
                    await _fast_attempt_registry.publish_nonterminal(receipt, acquiring)
                )
                last_guidance = acquiring["payload"]["guidance"]
            manual = await _fast_attempt_registry.take_manual_capture_request(receipt)
            if occupancy == "single" and aligned and (stable or manual):
                # The source frame remains Vision memory, never the MJPEG representation.
                captured_frame, source_frame = frame.copy(), source
                break
            await asyncio.sleep(_ACQUISITION_POLL_SECONDS)
        if captured_frame is None:
            raise asyncio.TimeoutError()

        # Capture transitions are ordered: capability close + lease release,
        # then public generating.  Profile ownership can resume before render.
        cleanup_errors = await _release_acquisition_resources(receipt, lease_token)
        lease_acquired = False
        if cleanup_errors:
            raise RuntimeError("acquisition_cleanup_failed")
        generating = _generated_v2_envelope(
            "vision.try_on.attempt.generating", {"attemptId": attempt_id, "stage": "preparing"}
        )
        await _publish_fast_transition(
            await _fast_attempt_registry.publish_nonterminal(receipt, generating)
        )
        garment_source = await asyncio.wait_for(
            _fast_runtime.fetch_garment(
                payload["garment"], await _fast_attempt_registry.cancel_event_for(receipt)
            ), timeout=_FAST_ATTEMPT_TIMEOUT_SECONDS,
        )
        result_image = await _run_owned_attempt_step(
            receipt,
            render_attempt_frame(
                captured_frame,
                garment_source.png_bytes,
                digest=garment_source.digest,
                template=garment_source.template,
                timeout=_FAST_ATTEMPT_TIMEOUT_SECONDS,
                broker=_fast_render_broker,
            ),
            timeout=_FAST_ATTEMPT_TIMEOUT_SECONDS,
        )
        stored_result, result = _prepare_fast_result(attempt_id, result_image)
        logger.info(
            "Fast attempt completed attemptId=%s frameSource=%s",
            attempt_id,
            source_frame.get("source") if isinstance(source_frame, dict) else "unknown",
        )
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.completed", {"attemptId": attempt_id, "result": result}
        )
    except PoseUnavailableError as error:
        # The contract's existing fast_failed terminal is the stable local
        # equivalent of pose_unavailable; never publish a result without
        # valid official shoulder/hip geometry.
        logger.warning("Fast pose unavailable attemptId=%s reason=%s", attempt_id, error)
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.failed",
            {"attemptId": attempt_id, "reason": "fast_failed"},
        )
    except GarmentFetchError as error:
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.canceled"
            if str(error) == "attempt_canceled"
            else "vision.try_on.attempt.failed",
            {"attemptId": attempt_id, "reason": "replaced"}
            if str(error) == "attempt_canceled"
            else {"attemptId": attempt_id, "reason": "garment_rejected"},
        )
    except asyncio.TimeoutError:
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.canceled", {"attemptId": attempt_id, "reason": "timeout"}
        )
    except Exception:
        logger.exception("Fast attempt failed attemptId=%s", attempt_id)
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.failed",
            {"attemptId": attempt_id, "reason": "fast_failed"},
        )
    except asyncio.CancelledError:
        terminal = _generated_v2_envelope(
            "vision.try_on.attempt.canceled", {"attemptId": attempt_id, "reason": "replaced"}
        )
    finally:
        if lease_acquired:
            await _release_acquisition_resources(receipt, lease_token)
        # The registry's cleanup barrier retains admission ownership until the
        # one worker slot is physically idle, even when a model call was slow.
        await _get_acquisition_observer().wait_idle()
    # Preparation is deliberately private until the completed contract is
    # encoded and validated.  Any failure above commits the one failed
    # terminal without passing staged token, reference, or bytes across the
    # registry's atomic boundary.
    if terminal.get("type") != "vision.try_on.attempt.completed":
        stored_result = None
    transition = await _fast_attempt_registry.commit_terminal_transition(
        receipt, terminal, stored_result
    )
    if transition is not None:
        await _publish_fast_transition(transition)


async def reject_v2_fast_attempt_for_backpressure(
    websocket: WebSocket, send_lock: asyncio.Lock, message: dict
) -> None:
    """Reject only an overflowed *new* attempt while preserving same-ID joins."""
    try:
        parsed = parse_v2_client_message(message)
        if parsed.type != "vision.try_on.attempt.start":
            raise ValueError("invalid_v2_boundary_message")
        attempt_id = parsed.payload.attemptId
    except (V2ContractBundleUnavailable, ValueError):
        async with send_lock:
            await send_error(
                websocket,
                code="invalid_message",
                message="invalid_v2_boundary_message",
                retryable=False,
                message_id=message.get("messageId"),
            )
        return
    terminal = _generated_v2_envelope(
        "vision.try_on.attempt.failed",
        {"attemptId": attempt_id, "reason": "fast_unavailable"},
    )
    admission = await _fast_attempt_registry.join_pending_or_reject(
        attempt_id=attempt_id,
        websocket=websocket,
        send_lock=send_lock,
        terminal=terminal,
    )
    for replay in admission.replay:
        await _send_json_bounded(websocket, send_lock, replay)


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


async def websocket_session(websocket: WebSocket, allowed_client_roles: set[str]):
    """WebSocket 主端点。

    处理以下消息类型：
    - vision.hello: 握手 + 能力协商 + 注册画像广播客户端
    - vision.ping: 心跳（回复 vision.pong）
    - vision.start_profile / vision.cancel: 不支持（主动推送协议中不需要）

    断开连接时自动清理广播注册。
    """
    send_lock = asyncio.Lock()
    owned_fast_attempt_receipts: set[AttemptReceipt] = set()
    fast_attempt_tasks: set[asyncio.Task] = set()
    connection_closed = asyncio.Event()
    handshake_complete = False
    fast_attempt_ready = False

    await websocket.accept()
    if not websocket_origin_allowed(websocket):
        await websocket.close(code=1008)
        logger.warning("WebSocket rejected due to untrusted Origin")
        return
    logger.info("WebSocket connected roles=%s", sorted(allowed_client_roles))

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
                try:
                    ready, client_capabilities = build_v2_ready_message(
                        message,
                        get_runtime_status(),
                    )
                except V2ContractBundleUnavailable:
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message="contract_bundle_unavailable",
                            retryable=False,
                            message_id=message_id,
                        )
                    await websocket.close(code=1008)
                    return
                except ValueError:
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message="invalid_v2_boundary_message",
                            retryable=False,
                            message_id=message_id,
                        )
                    continue

                if message["payload"].get("clientRole") not in allowed_client_roles:
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message=(
                                "payload.clientRole must be one of: "
                                + ", ".join(sorted(allowed_client_roles))
                            ),
                            retryable=False,
                            message_id=message_id,
                        )
                    continue

                async with send_lock:
                    await websocket.send_json(ready)

                fast_attempt_ready = bool(ready["payload"]["fastReady"])

                # Contract incompatibility is an enhancement-only readiness
                # fact. Presence/profile registration remains independent.
                if settings.PROFILE_PUSH_ENABLED:
                    await register_profile_client(
                        websocket,
                        send_lock,
                        client_capabilities,
                    )

                handshake_complete = True
                continue

            if message_type in {
                "vision.try_on.attempt.start",
                "vision.try_on.attempt.capture",
                "vision.try_on.attempt.cancel",
            }:
                try:
                    # Every V2 frame arriving at Vision crosses only the
                    # generated client entrypoint.  It is intentionally not a
                    # permissive envelope parser shared with server events.
                    message = parse_v2_client_message(message).model_dump(mode="json")
                    payload = message["payload"]
                except (V2ContractBundleUnavailable, ValueError):
                    async with send_lock:
                        await send_error(
                            websocket,
                            code="invalid_message",
                            message="invalid_v2_boundary_message",
                            retryable=False,
                            message_id=message_id,
                        )
                    continue

            is_v2_attempt_message = message_type in {
                "vision.try_on.attempt.start",
                "vision.try_on.attempt.capture",
                "vision.try_on.attempt.cancel",
            }
            is_v2_fast_attempt = (
                message_type == "vision.try_on.attempt.start"
                and isinstance(payload, dict)
                and "attemptId" in payload
            )
            payload_error = (
                None
                if is_v2_attempt_message
                else validate_message_payload(message_type, payload)
            )

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

            if is_v2_fast_attempt:
                if _fast_attempt_task_slots.locked():
                    await reject_v2_fast_attempt_for_backpressure(websocket, send_lock, message)
                    continue
                await _fast_attempt_task_slots.acquire()
                task = asyncio.create_task(
                    run_v2_fast_attempt(
                        websocket,
                        send_lock,
                        message,
                        fast_attempt_ready,
                        owned_fast_attempt_receipts,
                        connection_closed,
                    )
                )
                task.add_done_callback(lambda _: _fast_attempt_task_slots.release())
                fast_attempt_tasks.add(task)
                task.add_done_callback(fast_attempt_tasks.discard)
                task.add_done_callback(_discard_completed_fast_attempt)
                continue

            if message_type == "vision.try_on.attempt.capture":
                # Intent is deliberately non-terminal and accepts no frame bytes.
                await _fast_attempt_registry.request_manual_capture(payload["attemptId"])
                continue

            if message_type == "vision.try_on.attempt.cancel":
                terminal = _generated_v2_envelope(
                    "vision.try_on.attempt.canceled",
                    {"attemptId": payload["attemptId"], "reason": payload["reason"]},
                )
                await _publish_fast_transition(await _cancel_v2_attempt(
                    attempt_id=payload["attemptId"], terminal=terminal
                ))
                continue

            if message_type == "vision.ping":
                await _send_json_bounded(
                    websocket,
                    send_lock,
                    envelope(
                        message_type="vision.pong",
                        message_id=f"pong-{message_id}-{uuid4()}",
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
        connection_closed.set()
        await _await_cleanup_uncancelled(_fast_attempt_registry.detach_subscriber(websocket))
        for receipt in list(owned_fast_attempt_receipts):
            await _await_cleanup_uncancelled(
                _cancel_disconnect_owner_and_publish(receipt)
            )
        if fast_attempt_tasks:
            await _await_cleanup_uncancelled(
                asyncio.gather(*list(fast_attempt_tasks), return_exceptions=True)
            )
        await _await_cleanup_uncancelled(unregister_profile_client(websocket))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """VEM machine protocol surface; accepts only machine-role clients."""
    try:
        await websocket_session(websocket, {"machine"})
    except asyncio.CancelledError:
        # ASGI servers may cancel the scope immediately after a peer closes.
        # The attempt itself has already installed its cleanup barrier; do not
        # surface that transport teardown as an unhandled application task.
        logger.info("WebSocket scope cancelled after cleanup handoff")
        return


@app.websocket("/debug/ws")
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """Vendor diagnostic surface using the generated machine-role V2 hello."""
    await websocket_session(websocket, {"machine"})
