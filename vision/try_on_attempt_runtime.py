"""Deep runtime for the one V2 Virtual Try-On attempt lifecycle.

The application supplies its concrete camera, registry, worker and transport
adapters as one private port.  This module owns the externally observable
attempt identity and state order; callers never orchestrate capture, render or
terminal fencing themselves.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from vision.acquisition_session import AcquisitionSession


class TryOnAttemptPorts(Protocol):
    """Explicit host seam required by the attempt runtime.

    The concrete app adapter deliberately exposes only these named members;
    transport module globals outside this interface are not runtime inputs.
    """

    V2ContractBundleUnavailable: type[Exception]
    GarmentFetchError: type[Exception]
    PoseUnavailableError: type[Exception]
    AttemptWorkerError: type[Exception]
    _ACQUISITION_TIMEOUT_SECONDS: float
    _ACQUISITION_HOLD_SECONDS: float
    _ACQUISITION_POLL_SECONDS: float
    _TRY_ON_ATTEMPT_TIMEOUT_SECONDS: float
    _try_on_attempt_registry: Any
    _try_on_render_broker: Any
    _acquisition_previews: Any
    _captured_frames: Any
    _try_on_adjustment_store: Any
    _try_on_runtime: Any
    logger: Any
    parse_v2_client_message: Any
    send_error: Any
    _acquisition_observer_ready: Any
    _generated_v2_envelope: Any
    _publish_try_on_transition: Any
    _send_json_bounded: Any
    _acquire_front_io_until: Any
    acquire_front_camera: Any
    release_front_camera_io_lock: Any
    _acquiring_message: Any
    _read_attempt_front_frame: Any
    _get_acquisition_observer: Any
    _captured_frame_reference: Any
    _release_acquisition_resources: Any
    _run_owned_attempt_step: Any
    render_attempt_frame: Any
    _prepare_try_on_result: Any
    get_front_camera_owner: Any
    release_front_camera: Any
    _await_cleanup_uncancelled: Any


class TryOnAttemptRuntime:
    """Run one V2 request through admission, capture, generation and terminal."""

    def __init__(self, ports: TryOnAttemptPorts) -> None:
        self._ports = ports

    async def run(self, websocket, send_lock, message, try_on_ready, owned_receipts, connection_closed) -> None:
        p = self._ports
        try:
            parsed = p.parse_v2_client_message(message)
            if parsed.type != "vision.try_on.attempt.start":
                raise ValueError("invalid_v2_boundary_message")
            payload = parsed.payload.model_dump()
        except (p.V2ContractBundleUnavailable, ValueError):
            async with send_lock:
                await p.send_error(websocket, code="invalid_message", message="invalid_v2_boundary_message", retryable=False, message_id=message.get("messageId"))
            return

        try_on_ready = bool(try_on_ready and p._try_on_render_broker.ready and p._try_on_render_broker.pose_ready and p._acquisition_observer_ready())
        attempt_id = payload["attemptId"]
        unavailable = p._generated_v2_envelope("vision.try_on.attempt.failed", {"attemptId": attempt_id, "reason": "try_on_unavailable"})
        replaced = p._generated_v2_envelope("vision.try_on.attempt.canceled", {"attemptId": attempt_id, "reason": "replaced"})
        accepted = p._generated_v2_envelope("vision.try_on.attempt.accepted", {"attemptId": attempt_id}) if try_on_ready else None
        preparation = await p._try_on_attempt_registry.prepare_admission(
            attempt_id=attempt_id, websocket=websocket, send_lock=send_lock,
            task=asyncio.current_task(), canceled_terminal=replaced,
            owner_receipts=owned_receipts,
        )
        for transition in preparation.transitions:
            # Replacement makes its canceled terminal externally observable
            # before the old task has joined.  Revoke its media now, rather
            # than letting that brief handoff retain a usable captured token.
            if transition.message.get("type") == "vision.try_on.attempt.canceled":
                await self._revoke_media(
                    transition.message["payload"]["attemptId"]
                )
            await p._publish_try_on_transition(transition)
        admission = await p._try_on_attempt_registry.commit_prepared_admission(
            preparation, accepted=accepted, generating=None, unavailable_terminal=unavailable,
            readiness=lambda: bool(try_on_ready and p._try_on_render_broker.ready and p._try_on_render_broker.pose_ready and p._acquisition_observer_ready()),
        )
        if not admission.is_owner:
            for replay in admission.replay:
                await p._send_json_bounded(websocket, send_lock, replay)
            return
        receipt = admission.receipt
        assert receipt is not None
        if connection_closed.is_set():
            await p._publish_try_on_transition(await p._try_on_attempt_registry.cancel_owner_and_join(receipt))
            return

        stored_result = None
        lease_token = f"try-on:{receipt.attempt_id}:{receipt.generation}:{receipt.owner_token}"
        lease_acquired = False
        acquisition_deadline = (
            asyncio.get_running_loop().time() + p._ACQUISITION_TIMEOUT_SECONDS
        )
        try:
            for replay in admission.replay:
                await p._send_json_bounded(websocket, send_lock, replay)
            await p._acquire_front_io_until(acquisition_deadline)
            try:
                owner = p.acquire_front_camera("try_on_attempt", reason=f"try_on_acquisition:{attempt_id}", lease_token=lease_token)
            finally:
                p.release_front_camera_io_lock()
            if not owner.get("ok"):
                raise p.GarmentFetchError(owner.get("error") or "front_camera_busy")
            lease_acquired = True

            async def publish(preview_token, occupancy, _guidance, aligned, remaining):
                acquiring = p._acquiring_message(attempt_id, preview_token, occupancy, aligned, remaining)
                await p._publish_try_on_transition(await p._try_on_attempt_registry.publish_nonterminal(receipt, acquiring))

            acquisition = AcquisitionSession(
                attempt_id=attempt_id,
                timeout_seconds=max(
                    0.0, acquisition_deadline - asyncio.get_running_loop().time()
                ),
                stable_seconds=p._ACQUISITION_HOLD_SECONDS, preview_interval_seconds=p._ACQUISITION_POLL_SECONDS,
                read_frame=lambda timeout: p._read_attempt_front_frame(receipt, timeout=timeout, lease_token=lease_token),
                observe=lambda frame, timeout: p._get_acquisition_observer().observe(frame, timeout=timeout),
                preview_open=p._acquisition_previews.open, preview_update=p._acquisition_previews.update,
                publish=publish,
            )
            captured = await acquisition.acquire(
                manual_requested=lambda: p._try_on_attempt_registry.manual_capture_requested(receipt),
                consume_manual=lambda: p._try_on_attempt_registry.consume_manual_capture_request(receipt),
            )
            captured_token = await p._captured_frames.admit(attempt_id, captured)
            captured_event = p._generated_v2_envelope("vision.try_on.attempt.captured", {"attemptId": attempt_id, "captured": captured.public(p._captured_frame_reference(captured_token))})
            await p._publish_try_on_transition(await p._try_on_attempt_registry.publish_nonterminal(receipt, captured_event))
            cleanup_errors = await p._release_acquisition_resources(receipt, lease_token)
            lease_acquired = False
            if cleanup_errors:
                raise RuntimeError("acquisition_cleanup_failed")
            generating = p._generated_v2_envelope("vision.try_on.attempt.generating", {"attemptId": attempt_id, "stage": "preparing"})
            await p._publish_try_on_transition(await p._try_on_attempt_registry.publish_nonterminal(receipt, generating))
            garment = await asyncio.wait_for(p._try_on_runtime.fetch_garment(payload["garment"], await p._try_on_attempt_registry.cancel_event_for(receipt)), timeout=p._TRY_ON_ATTEMPT_TIMEOUT_SECONDS)
            result_image = await p._run_owned_attempt_step(
                receipt, p.render_attempt_frame(captured.frame, garment.png_bytes, digest=garment.digest, template=garment.template, timeout=p._TRY_ON_ATTEMPT_TIMEOUT_SECONDS, broker=p._try_on_render_broker), timeout=p._TRY_ON_ATTEMPT_TIMEOUT_SECONDS,
            )
            stored_result, result = p._prepare_try_on_result(attempt_id, result_image)
            p._try_on_adjustment_store.admit(attempt_id, captured.frame, garment.png_bytes, garment.digest, garment.template)
            terminal = p._generated_v2_envelope("vision.try_on.attempt.completed", {"attemptId": attempt_id, "result": result})
        except p.PoseUnavailableError:
            terminal = p._generated_v2_envelope("vision.try_on.attempt.failed", {"attemptId": attempt_id, "reason": "try_on_failed"})
        except p.GarmentFetchError as error:
            terminal = p._generated_v2_envelope("vision.try_on.attempt.canceled" if str(error) == "attempt_canceled" else "vision.try_on.attempt.failed", {"attemptId": attempt_id, "reason": "replaced"} if str(error) == "attempt_canceled" else {"attemptId": attempt_id, "reason": "garment_rejected"})
        except asyncio.TimeoutError:
            terminal = p._generated_v2_envelope("vision.try_on.attempt.canceled", {"attemptId": attempt_id, "reason": "timeout"})
        except asyncio.CancelledError:
            terminal = p._generated_v2_envelope("vision.try_on.attempt.canceled", {"attemptId": attempt_id, "reason": "replaced"})
        except Exception:
            p.logger.exception("Try-On attempt failed attemptId=%s", attempt_id)
            terminal = p._generated_v2_envelope("vision.try_on.attempt.failed", {"attemptId": attempt_id, "reason": "try_on_failed"})
        finally:
            if lease_acquired:
                await p._release_acquisition_resources(receipt, lease_token)
            await p._get_acquisition_observer().wait_idle()
        if terminal.get("type") != "vision.try_on.attempt.completed":
            stored_result = None
            await self._revoke_media(attempt_id)
        transition = await p._try_on_attempt_registry.commit_terminal_transition(receipt, terminal, stored_result)
        if transition is None:
            # A cancel/replacement may have won while an uncooperative worker
            # was returning.  Its staged source and captured capability are
            # never valid without the matching terminal ownership.
            await self._revoke_media(attempt_id)
        if transition is not None:
            await p._publish_try_on_transition(transition)

    async def read_captured(self, token: str):
        """Resolve one immutable captured capability for the HTTP adapter."""
        return await self._ports._captured_frames.get(token)

    async def _revoke_media(self, attempt_id: str) -> list[str]:
        """Revoke all non-terminal customer capabilities for one attempt."""
        p = self._ports
        errors: list[str] = []
        await p._captured_frames.discard(attempt_id)
        p._try_on_adjustment_store.discard(attempt_id)
        try:
            await p._acquisition_previews.close(attempt_id)
        except Exception as exc:
            errors.append(f"preview_close:{type(exc).__name__}:{exc}")
        return errors

    async def cancel(self, attempt_id: str, reason: str):
        """Fence an active attempt and revoke its media before publication."""
        p = self._ports
        terminal = p._generated_v2_envelope(
            "vision.try_on.attempt.canceled",
            {"attemptId": attempt_id, "reason": reason},
        )
        transition = await p._try_on_attempt_registry.cancel_current(
            attempt_id=attempt_id, terminal=terminal
        )
        if transition is None:
            if reason == "route_leave":
                await p._try_on_attempt_registry.discard_terminal(attempt_id)
                await self._revoke_media(attempt_id)
            return None
        cleanup_errors = await self._revoke_media(attempt_id)
        owner = p.get_front_camera_owner()
        token = owner.get("leaseToken")
        if (
            owner.get("owner") == "try_on_attempt"
            and isinstance(token, str)
            and token.startswith(f"try-on:{attempt_id}:")
        ):
            released = p.release_front_camera(
                "try_on_attempt",
                reason=f"try_on_canceled:{attempt_id}",
                lease_token=token,
            )
            if not released.get("ok"):
                cleanup_errors.append(f"front_release:{released.get('error')}")
        if cleanup_errors:
            p.logger.warning(
                "V2 attempt cancellation cleanup completed with errors "
                "attemptId=%s errors=%s",
                attempt_id,
                cleanup_errors,
            )
        await p._publish_try_on_transition(transition)
        return transition

    async def cancel_active(self, reason: str) -> None:
        """Cancel the currently active attempt, if one exists."""
        attempt_id = await self._ports._try_on_attempt_registry.active_attempt_id()
        if attempt_id is not None:
            await self.cancel(attempt_id, reason)

    async def adjust(self, websocket, send_lock, message: dict) -> None:
        """Re-render one completed attempt through its retained source."""
        p = self._ports
        payload = message["payload"]
        attempt_id = payload["attemptId"]
        scale = payload["garmentScale"]
        snapshot = p._try_on_adjustment_store.get(attempt_id)
        if snapshot is None:
            async with send_lock:
                await p.send_error(
                    websocket,
                    code="adjustment_unavailable",
                    message="the Try-On adjustment source is no longer retained",
                    retryable=False,
                    message_id=message.get("messageId"),
                )
            return
        try:
            result_image = await p.render_attempt_frame(
                snapshot.frame,
                snapshot.garment_png,
                digest=snapshot.garment_digest,
                template=snapshot.template,
                timeout=p._TRY_ON_ATTEMPT_TIMEOUT_SECONDS,
                broker=p._try_on_render_broker,
                garment_scale=scale,
            )
            stored_result, _ = p._prepare_try_on_result(attempt_id, result_image)
            replaced = await p._try_on_attempt_registry.replace_completed_result(
                attempt_id, stored_result
            )
            if replaced is None:
                raise RuntimeError("try_on_adjustment_target_unavailable")
            adjusted = p._generated_v2_envelope(
                "vision.try_on.result.adjusted",
                {"attemptId": attempt_id, "result": replaced},
            )
            async with send_lock:
                await websocket.send_json(adjusted)
        except (
            p.PoseUnavailableError,
            p.AttemptWorkerError,
            RuntimeError,
            TimeoutError,
        ):
            p.logger.exception(
                "Try-On adjustment failed attemptId=%s scale=%s",
                attempt_id,
                scale,
            )
            async with send_lock:
                await p.send_error(
                    websocket,
                    code="internal_error",
                    message="the Try-On result could not be adjusted",
                    retryable=False,
                    message_id=message.get("messageId"),
                )

    async def disconnect(self, websocket, owner_receipts, attempt_tasks) -> None:
        """Finish all attempt cleanup owned by one disconnected transport."""
        p = self._ports
        await p._await_cleanup_uncancelled(
            p._try_on_attempt_registry.detach_subscriber(websocket)
        )
        for receipt in list(owner_receipts):
            transition = await p._try_on_attempt_registry.cancel_owner_and_join(
                receipt,
                p._generated_v2_envelope(
                    "vision.try_on.attempt.canceled",
                    {"attemptId": receipt.attempt_id, "reason": "disconnect"},
                ),
            )
            await p._publish_try_on_transition(transition)
        if attempt_tasks:
            await p._await_cleanup_uncancelled(
                asyncio.gather(*list(attempt_tasks), return_exceptions=True)
            )
        completed_attempt_ids = await p._await_cleanup_uncancelled(
            p._try_on_attempt_registry.revoke_completed_owner_results(websocket)
        )
        for attempt_id in completed_attempt_ids:
            await p._await_cleanup_uncancelled(self._revoke_media(attempt_id))
