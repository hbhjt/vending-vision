"""Deep runtime for the one V2 Virtual Try-On attempt lifecycle.

The application supplies its concrete camera, registry, worker and transport
adapters as one private port.  This module owns the externally observable
attempt identity and state order; callers never orchestrate capture, render or
terminal fencing themselves.
"""

from __future__ import annotations

import asyncio
from typing import Any

from vision.acquisition_session import AcquisitionSession


class TryOnAttemptRuntime:
    """Run one V2 request through admission, capture, generation and terminal."""

    def __init__(self, port: Any) -> None:
        self._port = port

    async def run(self, websocket, send_lock, message, try_on_ready, owned_receipts, connection_closed) -> None:
        p = self._port
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
                await p._revoke_try_on_attempt_media(
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
        try:
            for replay in admission.replay:
                await p._send_json_bounded(websocket, send_lock, replay)
            await p._acquire_front_io_until(asyncio.get_running_loop().time() + p._ACQUISITION_TIMEOUT_SECONDS)
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
                attempt_id=attempt_id, timeout_seconds=p._ACQUISITION_TIMEOUT_SECONDS,
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
            await p._revoke_try_on_attempt_media(attempt_id)
        transition = await p._try_on_attempt_registry.commit_terminal_transition(receipt, terminal, stored_result)
        if transition is None:
            # A cancel/replacement may have won while an uncooperative worker
            # was returning.  Its staged source and captured capability are
            # never valid without the matching terminal ownership.
            await p._revoke_try_on_attempt_media(attempt_id)
        if transition is not None:
            await p._publish_try_on_transition(transition)
