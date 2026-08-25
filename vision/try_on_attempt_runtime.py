"""Deep runtime for the single V2 Virtual Try-On attempt lifecycle."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Protocol, cast

import numpy as np

from vision.acquisition_session import (
    AcquisitionCameraPort,
    AcquisitionObserverPort,
    AcquisitionPreviewPort,
    AcquisitionSession,
    CapturedFrame,
)
from vision.attempt_worker import AttemptWorkerError
from vision.garment_composer import (
    GarmentFetchError,
    PoseUnavailableError,
    TransparentGarmentSource,
)
from vision.try_on_adjustment_store import TryOnAdjustmentSnapshot
from vision.try_on_attempt_registry import (
    AttemptReceipt,
    TerminalTransition,
    TryOnAttemptRegistry,
)
from vision.v2_contract_bundle import V2ContractBundleUnavailable


class AttemptRegistryPort(Protocol):
    @property
    def current(self) -> TryOnAttemptRegistry: ...


class AttemptMediaPort(Protocol):
    @property
    def preview(self) -> AcquisitionPreviewPort: ...

    async def store_captured(
        self, attempt_id: str, captured: CapturedFrame
    ) -> dict[str, object]: ...

    async def read_captured(self, token: str) -> CapturedFrame | None: ...

    async def revoke(self, attempt_id: str) -> None: ...

    def prepare_result(
        self, attempt_id: str, image: bytes
    ) -> tuple[dict, dict[str, object]]: ...

    def retain_adjustment(
        self, attempt_id: str, frame: np.ndarray, garment: TransparentGarmentSource
    ) -> None: ...

    def adjustment(self, attempt_id: str) -> TryOnAdjustmentSnapshot | None: ...

    def preview_public(self, token: str) -> dict[str, object]: ...


class AttemptRenderPort(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def pose_ready(self) -> bool: ...

    async def fetch_garment(
        self, descriptor: dict[str, object], cancel_event: asyncio.Event
    ) -> TransparentGarmentSource: ...

    async def render(
        self,
        frame: np.ndarray,
        garment: TransparentGarmentSource,
        *,
        timeout: float,
        garment_scale: float = 1.0,
    ) -> bytes: ...


class AttemptCameraPort(Protocol):
    def ready(self) -> bool: ...

    def bind(self, receipt: AttemptReceipt) -> AcquisitionCameraPort: ...

    def observer(self) -> AcquisitionObserverPort: ...

    async def wait_released(self, attempt_id: str) -> bool: ...


class AttemptTransportPort(Protocol):
    def parse_start(self, message: dict) -> dict[str, object]: ...

    def envelope(
        self, message_type: str, payload: dict[str, object]
    ) -> dict: ...

    async def publish(self, transition: TerminalTransition | None) -> None: ...

    async def send(
        self, websocket: object, send_lock: asyncio.Lock, message: dict
    ) -> None: ...

    async def send_error(
        self,
        websocket: object,
        send_lock: asyncio.Lock,
        *,
        code: str,
        message: str,
        retryable: bool,
        message_id: str | None,
    ) -> None: ...


class AttemptRuntimeConfig(Protocol):
    @property
    def acquisition_timeout_seconds(self) -> float: ...

    @property
    def acquisition_hold_seconds(self) -> float: ...

    @property
    def acquisition_poll_seconds(self) -> float: ...

    @property
    def render_timeout_seconds(self) -> float: ...


@dataclass(frozen=True)
class TryOnAttemptPorts:
    registry: AttemptRegistryPort
    media: AttemptMediaPort
    render: AttemptRenderPort
    camera: AttemptCameraPort
    transport: AttemptTransportPort
    config: AttemptRuntimeConfig


class TryOnAttemptRuntime:
    """Own admission, capture, generation, cleanup and terminal replay."""

    def __init__(self, ports: TryOnAttemptPorts) -> None:
        self._ports = ports
        self._logger = logging.getLogger("vending_vision.try_on_attempt")

    async def run(
        self,
        websocket: object,
        send_lock: asyncio.Lock,
        message: dict,
        try_on_ready: bool,
        owned_receipts: set[AttemptReceipt],
        connection_closed: asyncio.Event,
    ) -> None:
        p = self._ports
        registry = p.registry.current
        try:
            payload = p.transport.parse_start(message)
        except (V2ContractBundleUnavailable, ValueError):
            await p.transport.send_error(
                websocket,
                send_lock,
                code="invalid_message",
                message="invalid_v2_boundary_message",
                retryable=False,
                message_id=cast(str | None, message.get("messageId")),
            )
            return

        try_on_ready = bool(
            try_on_ready
            and p.render.ready
            and p.render.pose_ready
            and p.camera.ready()
        )
        attempt_id = cast(str, payload["attemptId"])
        unavailable = p.transport.envelope(
            "vision.try_on.attempt.failed",
            {"attemptId": attempt_id, "reason": "try_on_unavailable"},
        )
        replaced = p.transport.envelope(
            "vision.try_on.attempt.canceled",
            {"attemptId": attempt_id, "reason": "replaced"},
        )
        accepted = (
            p.transport.envelope(
                "vision.try_on.attempt.accepted", {"attemptId": attempt_id}
            )
            if try_on_ready
            else None
        )
        preparation = await registry.prepare_admission(
            attempt_id=attempt_id,
            websocket=websocket,
            send_lock=send_lock,
            task=cast(asyncio.Task, asyncio.current_task()),
            canceled_terminal=replaced,
            owner_receipts=owned_receipts,
        )
        for transition in preparation.transitions:
            if transition.message.get("type") == "vision.try_on.attempt.canceled":
                await p.media.revoke(
                    cast(str, transition.message["payload"]["attemptId"])
                )
            await p.transport.publish(transition)
        admission = await registry.commit_prepared_admission(
            preparation,
            accepted=accepted,
            generating=None,
            unavailable_terminal=unavailable,
            readiness=lambda: bool(
                try_on_ready
                and p.render.ready
                and p.render.pose_ready
                and p.camera.ready()
            ),
        )
        if not admission.is_owner:
            for replay in admission.replay:
                await p.transport.send(websocket, send_lock, replay)
            return
        receipt = admission.receipt
        assert receipt is not None
        if connection_closed.is_set():
            await p.transport.publish(await registry.cancel_owner_and_join(receipt))
            return

        acquisition_deadline = (
            asyncio.get_running_loop().time()
            + p.config.acquisition_timeout_seconds
        )
        stored_result = None
        try:
            for replay in admission.replay:
                await p.transport.send(websocket, send_lock, replay)

            async def publish(
                preview_token: str,
                occupancy: str,
                guidance: str,
                aligned: bool,
                remaining: int | None,
            ) -> None:
                acquiring = p.transport.envelope(
                    "vision.try_on.attempt.acquiring",
                    self._acquiring_payload(
                        attempt_id,
                        preview_token,
                        occupancy,
                        guidance,
                        aligned,
                        remaining,
                    ),
                )
                await p.transport.publish(
                    await registry.publish_nonterminal(receipt, acquiring)
                )

            acquisition = AcquisitionSession(
                attempt_id=attempt_id,
                deadline=acquisition_deadline,
                timeout_seconds=max(
                    0.0,
                    acquisition_deadline - asyncio.get_running_loop().time(),
                ),
                stable_seconds=p.config.acquisition_hold_seconds,
                preview_interval_seconds=p.config.acquisition_poll_seconds,
                camera=p.camera.bind(receipt),
                observer=p.camera.observer(),
                preview=p.media.preview,
                publish=publish,
            )
            captured = await acquisition.acquire(
                manual_requested=lambda: registry.manual_capture_requested(receipt),
                consume_manual=lambda: registry.consume_manual_capture_request(
                    receipt
                ),
            )
            captured_public = await p.media.store_captured(attempt_id, captured)
            captured_event = p.transport.envelope(
                "vision.try_on.attempt.captured",
                {"attemptId": attempt_id, "captured": captured_public},
            )
            await p.transport.publish(
                await registry.publish_nonterminal(receipt, captured_event)
            )
            generating = p.transport.envelope(
                "vision.try_on.attempt.generating",
                {"attemptId": attempt_id, "stage": "preparing"},
            )
            await p.transport.publish(
                await registry.publish_nonterminal(receipt, generating)
            )
            garment = await asyncio.wait_for(
                p.render.fetch_garment(
                    cast(dict[str, object], payload["garment"]),
                    await registry.cancel_event_for(receipt),
                ),
                timeout=p.config.render_timeout_seconds,
            )
            result_image = await self._run_owned_render(
                receipt,
                p.render.render(
                    captured.frame,
                    garment,
                    timeout=p.config.render_timeout_seconds,
                ),
                timeout=p.config.render_timeout_seconds,
            )
            stored_result, result = p.media.prepare_result(
                attempt_id, result_image
            )
            p.media.retain_adjustment(attempt_id, captured.frame, garment)
            terminal = p.transport.envelope(
                "vision.try_on.attempt.completed",
                {"attemptId": attempt_id, "result": result},
            )
        except PoseUnavailableError:
            terminal = p.transport.envelope(
                "vision.try_on.attempt.failed",
                {"attemptId": attempt_id, "reason": "try_on_failed"},
            )
        except GarmentFetchError as error:
            canceled = str(error) == "attempt_canceled"
            terminal = p.transport.envelope(
                "vision.try_on.attempt.canceled"
                if canceled
                else "vision.try_on.attempt.failed",
                {
                    "attemptId": attempt_id,
                    "reason": "replaced" if canceled else "garment_rejected",
                },
            )
        except asyncio.TimeoutError:
            terminal = p.transport.envelope(
                "vision.try_on.attempt.canceled",
                {"attemptId": attempt_id, "reason": "timeout"},
            )
        except asyncio.CancelledError:
            terminal = p.transport.envelope(
                "vision.try_on.attempt.canceled",
                {"attemptId": attempt_id, "reason": "replaced"},
            )
        except Exception:
            self._logger.exception("Try-On attempt failed attemptId=%s", attempt_id)
            terminal = p.transport.envelope(
                "vision.try_on.attempt.failed",
                {"attemptId": attempt_id, "reason": "try_on_failed"},
            )

        if terminal.get("type") != "vision.try_on.attempt.completed":
            stored_result = None
            await p.media.revoke(attempt_id)
        final_transition = await registry.commit_terminal_transition(
            receipt, terminal, stored_result
        )
        if final_transition is None:
            await p.media.revoke(attempt_id)
        else:
            await p.transport.publish(final_transition)

    def _acquiring_payload(
        self,
        attempt_id: str,
        preview_token: str,
        occupancy: str,
        guidance: str,
        aligned: bool,
        remaining: int | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "attemptId": attempt_id,
            "preview": self._ports.media.preview_public(preview_token),
            "occupancy": occupancy,
            "guidance": guidance,
            "manualCaptureAllowed": occupancy == "single" and aligned,
        }
        if guidance == "counting_down":
            payload["holdRemainingMs"] = min(int(remaining or 0), 10_000)
        return payload

    async def _run_owned_render(
        self,
        receipt: AttemptReceipt,
        operation: Awaitable[bytes],
        *,
        timeout: float,
    ) -> bytes:
        registry = self._ports.registry.current
        if not await registry.is_current(receipt):
            raise GarmentFetchError("attempt_canceled")
        async def wait_operation() -> object:
            return await operation

        worker: asyncio.Task[object] = asyncio.create_task(wait_operation())
        cancel_event = await registry.cancel_event_for(receipt)
        cancel_waiter: asyncio.Task[object] = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {worker, cancel_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker in done and not cancel_event.is_set():
                return cast(bytes, worker.result())
            if cancel_event.is_set() or cancel_waiter in done:
                raise GarmentFetchError("attempt_canceled")
            raise asyncio.TimeoutError()
        except (asyncio.TimeoutError, asyncio.CancelledError, GarmentFetchError):
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            raise
        finally:
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)
            if not await registry.is_current(receipt):
                raise GarmentFetchError("attempt_canceled")

    async def read_captured(self, token: str) -> CapturedFrame | None:
        return await self._ports.media.read_captured(token)

    async def cancel(self, attempt_id: str, reason: str):
        p = self._ports
        registry = p.registry.current
        terminal = p.transport.envelope(
            "vision.try_on.attempt.canceled",
            {"attemptId": attempt_id, "reason": reason},
        )
        transition = await registry.cancel_current(
            attempt_id=attempt_id, terminal=terminal
        )
        if transition is None:
            if reason == "route_leave":
                await registry.revoke_completed_result(attempt_id)
                await p.media.revoke(attempt_id)
            return None
        await p.media.revoke(attempt_id)
        await p.camera.wait_released(attempt_id)
        await p.transport.publish(transition)
        return transition

    async def adjust(
        self, websocket: object, send_lock: asyncio.Lock, message: dict
    ) -> None:
        p = self._ports
        payload = cast(dict[str, object], message["payload"])
        attempt_id = cast(str, payload["attemptId"])
        scale = float(cast(float, payload["garmentScale"]))
        snapshot = p.media.adjustment(attempt_id)
        if snapshot is None:
            await p.transport.send_error(
                websocket,
                send_lock,
                code="adjustment_unavailable",
                message="the Try-On adjustment source is no longer retained",
                retryable=False,
                message_id=cast(str | None, message.get("messageId")),
            )
            return
        try:
            garment = TransparentGarmentSource(
                png_bytes=snapshot.garment_png,
                digest=snapshot.garment_digest,
                template=snapshot.template,
            )
            result_image = await p.render.render(
                cast(np.ndarray, snapshot.frame),
                garment,
                timeout=p.config.render_timeout_seconds,
                garment_scale=scale,
            )
            stored_result, _ = p.media.prepare_result(attempt_id, result_image)
            replaced = await p.registry.current.replace_completed_result(
                attempt_id, stored_result
            )
            if replaced is None:
                raise RuntimeError("try_on_adjustment_target_unavailable")
            adjusted = p.transport.envelope(
                "vision.try_on.result.adjusted",
                {"attemptId": attempt_id, "result": replaced},
            )
            await p.transport.send(websocket, send_lock, adjusted)
        except (PoseUnavailableError, AttemptWorkerError, RuntimeError, TimeoutError):
            self._logger.exception(
                "Try-On adjustment failed attemptId=%s scale=%s", attempt_id, scale
            )
            await p.transport.send_error(
                websocket,
                send_lock,
                code="internal_error",
                message="the Try-On result could not be adjusted",
                retryable=False,
                message_id=cast(str | None, message.get("messageId")),
            )

    async def disconnect(
        self,
        websocket: object,
        owner_receipts: set[AttemptReceipt],
        attempt_tasks: set[asyncio.Task],
    ) -> None:
        p = self._ports
        registry = p.registry.current
        await self._await_uncancelled(registry.detach_subscriber(websocket))
        for receipt in list(owner_receipts):
            transition = await registry.cancel_owner_and_join(
                receipt,
                p.transport.envelope(
                    "vision.try_on.attempt.canceled",
                    {"attemptId": receipt.attempt_id, "reason": "disconnect"},
                ),
            )
            await p.transport.publish(transition)
        if attempt_tasks:
            await self._await_uncancelled(
                asyncio.gather(*list(attempt_tasks), return_exceptions=True)
            )
        completed_attempt_ids = await self._await_uncancelled(
            registry.revoke_completed_owner_results(websocket)
        )
        for attempt_id in completed_attempt_ids:
            await self._await_uncancelled(p.media.revoke(attempt_id))

    @staticmethod
    async def _await_uncancelled(awaitable):
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()
