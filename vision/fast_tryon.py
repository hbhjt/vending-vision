"""Bounded, local-only Fast virtual try-on primitive.

The runtime receives only a daemon-issued loopback PNG descriptor.  It never
contacts platform services or follows redirects, and derives placement from the
verified image bounds plus the declared template rather than product anchors.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
import numpy as np


class GarmentFetchError(ValueError):
    pass


class PoseUnavailableError(RuntimeError):
    """The official pose model could not produce safe try-on geometry."""


@dataclass(frozen=True)
class PreparedGarment:
    rgba: np.ndarray
    digest: str
    template: str
    alpha_mask: np.ndarray
    opaque_bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class ValidatedGarmentSource:
    png_bytes: bytes
    digest: str
    template: str


@dataclass(frozen=True)
class PoseGeometry:
    """Frame-space geometry derived from the official pose landmarks."""

    shoulder_center: np.ndarray
    shoulder_span: float
    across_unit: np.ndarray
    torso_down_unit: np.ndarray
    torso_length: float
    landmarks: dict[str, np.ndarray]


_POSE_LANDMARKS = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


class FastTryOnRuntime:
    def __init__(
        self,
        max_garment_bytes: int = 8 * 1024 * 1024,
        *,
        pose_estimator=None,
    ):
        self.max_garment_bytes = max_garment_bytes
        # The render worker owns this instance for its whole lifetime.  No
        # estimator is imported or constructed on the parent request path.
        self.pose_estimator = pose_estimator

    async def fetch_garment(
        self, descriptor: dict, cancel_event=None
    ) -> ValidatedGarmentSource:
        reference = descriptor.get("reference")
        if not isinstance(reference, str):
            raise GarmentFetchError("reference")
        url = urlparse(reference)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not parse_qs(url.query).get("token")
        ):
            raise GarmentFetchError("loopback")
        if cancel_event is not None and cancel_event.is_set():
            raise GarmentFetchError("attempt_canceled")
        timeout = httpx.Timeout(connect=1.0, read=0.25, write=1.0, pool=1.0)
        try:
            async with asyncio.timeout(5.0):
                # trust_env is intentionally disabled: a local daemon read
                # grant must never be diverted to a configured proxy.
                async with httpx.AsyncClient(
                    trust_env=False,
                    follow_redirects=False,
                    timeout=timeout,
                ) as client:
                    stream = client.stream("GET", reference)
                    response = await self._await_or_cancel(stream.__aenter__(), cancel_event)
                    try:
                        if 300 <= response.status_code < 400:
                            raise GarmentFetchError("redirect")
                        if (
                            response.status_code != 200
                            or response.headers.get("content-type", "").split(";", 1)[0]
                            != "image/png"
                        ):
                            raise GarmentFetchError("content_type")
                        chunks: list[bytes] = []
                        total = 0
                        iterator = response.aiter_bytes(64 * 1024).__aiter__()
                        while True:
                            if cancel_event is not None and cancel_event.is_set():
                                raise GarmentFetchError("attempt_canceled")
                            try:
                                chunk = await self._await_or_cancel(anext(iterator), cancel_event)
                            except StopAsyncIteration:
                                break
                            if not chunk:
                                continue
                            chunks.append(chunk)
                            total += len(chunk)
                            if total > self.max_garment_bytes:
                                raise GarmentFetchError("byte_size")
                        payload = b"".join(chunks)
                    finally:
                        await stream.__aexit__(None, None, None)
        except TimeoutError as exc:
            raise GarmentFetchError("deadline") from exc
        except httpx.HTTPError as exc:
            raise GarmentFetchError("transport") from exc
        if len(payload) > self.max_garment_bytes or len(payload) != descriptor.get("byteSize"):
            raise GarmentFetchError("byte_size")
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != descriptor.get("digest"):
            raise GarmentFetchError("digest")
        self._predecode_png(payload)
        if descriptor.get("contentType") != "image/png" or descriptor.get(
            "template"
        ) not in {"tshirt_short_sleeve", "tshirt_long_sleeve"}:
            raise GarmentFetchError("descriptor")
        return ValidatedGarmentSource(
            png_bytes=payload,
            digest=digest,
            template=descriptor["template"],
        )

    def prepare_garment(self, source: ValidatedGarmentSource) -> PreparedGarment:
        if len(source.png_bytes) > self.max_garment_bytes:
            raise GarmentFetchError("byte_size")
        digest = "sha256:" + hashlib.sha256(source.png_bytes).hexdigest()
        if digest != source.digest:
            raise GarmentFetchError("digest")
        self._predecode_png(source.png_bytes)
        rgba = cv2.imdecode(
            np.frombuffer(source.png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise GarmentFetchError("png_decode")
        if source.template not in {"tshirt_short_sleeve", "tshirt_long_sleeve"}:
            raise GarmentFetchError("descriptor")
        alpha_mask = np.where(rgba[:, :, 3] >= 12, 255, 0).astype(np.uint8)
        # Keep source cut-outs (neckline and transparent hem) while closing
        # only single-pixel encode noise at the garment boundary.
        alpha_mask = cv2.morphologyEx(
            alpha_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )
        rgba = rgba.copy()
        rgba[:, :, 3] = np.where(alpha_mask > 0, rgba[:, :, 3], 0)
        points = cv2.findNonZero(alpha_mask)
        if points is None:
            raise GarmentFetchError("transparent_png")
        x, y, width, height = cv2.boundingRect(points)
        # This preparation is attempt-local: it is deterministically derived
        # from source PNG + template + verified digest and is discarded after
        # rendering. No garment anchors or product tuning cross the boundary.
        return PreparedGarment(
            rgba=rgba,
            digest=digest,
            template=source.template,
            alpha_mask=alpha_mask,
            opaque_bounds=(x, y, width, height),
        )

    @staticmethod
    async def _await_or_cancel(awaitable, cancel_event):
        """Race every socket wait, including response headers, with cancellation."""
        operation = asyncio.create_task(awaitable)
        if cancel_event is None:
            return await operation
        cancel_waiter = asyncio.create_task(cancel_event.wait())
        done, pending = await asyncio.wait(
            {operation, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        for pending_task in pending:
            pending_task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if cancel_waiter in done:
            # Cancelling the in-flight transport call before the client context
            # exits closes a blocked connect/header/body read immediately.
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise GarmentFetchError("attempt_canceled")
        return operation.result()

    def _predecode_png(self, payload: bytes) -> None:
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise GarmentFetchError("png_magic")
        if len(payload) < 33:
            raise GarmentFetchError("png_malformed")
        ihdr_length = struct.unpack(">I", payload[8:12])[0]
        if ihdr_length != 13 or payload[12:16] != b"IHDR":
            raise GarmentFetchError("png_ihdr")
        width, height = struct.unpack(">II", payload[16:24])
        if (
            width == 0
            or height == 0
            or width > 4096
            or height > 4096
            or width * height > 4096 * 4096
        ):
            raise GarmentFetchError("png_dimensions")
        offset = 8
        seen_iend = False
        chunks = 0
        while offset + 12 <= len(payload):
            length = struct.unpack(">I", payload[offset : offset + 4])[0]
            chunk_type = payload[offset + 4 : offset + 8]
            offset += 8
            if length > self.max_garment_bytes or offset + length + 4 > len(payload):
                raise GarmentFetchError("png_chunk")
            offset += length + 4
            chunks += 1
            if chunks > 4096:
                raise GarmentFetchError("png_chunk")
            if chunk_type == b"IEND":
                seen_iend = True
                break
        if not seen_iend or offset != len(payload):
            raise GarmentFetchError("png_malformed")

    def _landmark_points(self, pose_results, width: int, height: int) -> dict[str, np.ndarray]:
        if pose_results is None:
            raise PoseUnavailableError("pose_unavailable")
        collection = getattr(pose_results, "pose_landmarks", pose_results)
        raw = getattr(collection, "landmark", collection)
        if raw is None:
            raise PoseUnavailableError("pose_unavailable")
        points: dict[str, np.ndarray] = {}
        for name, index in _POSE_LANDMARKS.items():
            try:
                landmark = raw[index] if not isinstance(raw, dict) else raw.get(index, raw.get(name))
            except (IndexError, KeyError, TypeError):
                continue
            if landmark is None:
                continue
            visibility = float(getattr(landmark, "visibility", 1.0))
            x = float(getattr(landmark, "x", float("nan")))
            y = float(getattr(landmark, "y", float("nan")))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if not math.isfinite(visibility) or visibility < 0.25:
                continue
            # MediaPipe can report a point a little outside the image, but a
            # grossly invalid normalized geometry must never place a garment.
            if x < -0.10 or x > 1.10 or y < -0.10 or y > 1.10:
                continue
            points[name] = np.array([x * width, y * height], dtype=np.float32)
        return points

    def pose_geometry(self, pose_results, width: int, height: int) -> PoseGeometry:
        points = self._landmark_points(pose_results, width, height)
        if "left_shoulder" not in points or "right_shoulder" not in points:
            raise PoseUnavailableError("pose_unavailable")
        # MediaPipe's LEFT/RIGHT landmarks are anatomical sides.  A person
        # facing an unmirrored camera therefore commonly has LEFT at a larger
        # x than RIGHT.  Garment pixels, however, are source *screen* sides:
        # sort before deriving the projective across axis so a source-left
        # pixel always reaches screen-left regardless of camera mirroring or
        # anatomical labels.
        anatomical_left_shoulder = points["left_shoulder"]
        anatomical_right_shoulder = points["right_shoulder"]
        if anatomical_left_shoulder[0] <= anatomical_right_shoulder[0]:
            screen_left_shoulder = anatomical_left_shoulder
            screen_right_shoulder = anatomical_right_shoulder
        else:
            screen_left_shoulder = anatomical_right_shoulder
            screen_right_shoulder = anatomical_left_shoulder
        shoulder_axis = screen_right_shoulder - screen_left_shoulder
        shoulder_span = float(np.linalg.norm(shoulder_axis))
        shoulder_center = (screen_left_shoulder + screen_right_shoulder) * 0.5

        # Infer hips when the lower body is cut off at the bottom of a close-up
        # frame: derive a downward torso direction from the shoulder axis and
        # place inferred hips one shoulder span below each shoulder.
        estimated_down = np.array(
            [
                -shoulder_axis[1] / max(shoulder_span, 1e-6),
                shoulder_axis[0] / max(shoulder_span, 1e-6),
            ],
            dtype=np.float32,
        )
        if estimated_down[1] < 0:
            estimated_down = -estimated_down
        if "left_hip" not in points:
            points["left_hip"] = anatomical_left_shoulder + estimated_down * (
                shoulder_span * 1.25
            )
        if "right_hip" not in points:
            points["right_hip"] = anatomical_right_shoulder + estimated_down * (
                shoulder_span * 1.25
            )

        hip_center = (points["left_hip"] + points["right_hip"]) * 0.5
        torso_axis = hip_center - shoulder_center
        torso_length = float(np.linalg.norm(torso_axis))
        frame_diagonal = float(math.hypot(width, height))
        if (
            not math.isfinite(shoulder_span)
            or not math.isfinite(torso_length)
            or shoulder_span < max(12.0, width * 0.055)
            or torso_length < max(18.0, height * 0.08)
            or torso_length > frame_diagonal * 1.25
            or float(np.dot(torso_axis, np.array([0.0, 1.0]))) <= 0
        ):
            raise PoseUnavailableError("pose_unavailable")
        return PoseGeometry(
            shoulder_center=shoulder_center,
            shoulder_span=shoulder_span,
            across_unit=shoulder_axis / shoulder_span,
            torso_down_unit=torso_axis / torso_length,
            torso_length=torso_length,
            landmarks=points,
        )

    @staticmethod
    def _protected_regions(geometry: PoseGeometry, width: int, height: int) -> np.ndarray:
        """Return head/face and visible-arm regions that must remain original."""
        protected = np.zeros((height, width), dtype=np.uint8)
        points = geometry.landmarks
        span = geometry.shoulder_span
        down = geometry.torso_down_unit
        across = geometry.across_unit
        head_center = points.get("nose", geometry.shoulder_center - down * span * 0.50)
        face_points = [points[name] for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") if name in points]
        if len(face_points) >= 3:
            hull = cv2.convexHull(np.asarray(face_points, dtype=np.float32).astype(np.int32))
            cv2.fillConvexPoly(protected, hull, 255)
        # A generous ellipse covers hair, cheeks and the untracked face edge.
        cv2.ellipse(
            protected,
            tuple(np.rint(head_center).astype(int)),
            (max(5, int(span * 0.25)), max(8, int(span * 0.34))),
            -math.degrees(math.atan2(across[1], across[0])) + 90.0,
            0,
            360,
            255,
            -1,
        )
        for side in ("left", "right"):
            shoulder = points.get(f"{side}_shoulder")
            elbow = points.get(f"{side}_elbow")
            wrist = points.get(f"{side}_wrist")
            if shoulder is None or elbow is None:
                continue
            arm = [shoulder, elbow]
            if wrist is not None:
                arm.append(wrist)
            arm_axis = arm[-1] - arm[0]
            normal = np.array([-arm_axis[1], arm_axis[0]], dtype=np.float32)
            normal /= max(float(np.linalg.norm(normal)), 1e-6)
            radius = span * 0.17
            polygon = np.asarray([arm[0] + normal * radius, arm[0] - normal * radius, arm[-1] - normal * radius * 0.65, arm[-1] + normal * radius * 0.65], dtype=np.float32).astype(np.int32)
            cv2.fillConvexPoly(protected, polygon, 255)
        return protected

    def _resolve_pose(self, frame: np.ndarray, pose_results=None) -> PoseGeometry:
        if pose_results is None:
            if self.pose_estimator is None:
                raise PoseUnavailableError("pose_unavailable")
            try:
                pose_results = self.pose_estimator.detect(frame)
            except Exception as exc:
                raise PoseUnavailableError("pose_unavailable") from exc
        return self.pose_geometry(pose_results, frame.shape[1], frame.shape[0])

    def render(
        self,
        frame: np.ndarray,
        garment: PreparedGarment | ValidatedGarmentSource,
        *,
        pose_results=None,
        garment_scale: float = 1.0,
    ) -> bytes:
        if isinstance(garment, ValidatedGarmentSource):
            garment = self.prepare_garment(garment)
        garment_scale = float(garment_scale)
        if not math.isfinite(garment_scale):
            raise PoseUnavailableError("pose_unavailable")
        garment_scale = min(1.6, max(0.8, garment_scale))
        height, width = frame.shape[:2]
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise PoseUnavailableError("pose_unavailable")
        geometry = self._resolve_pose(frame, pose_results)
        x0, y0, source_width, source_height = garment.opaque_bounds
        source = garment.rgba[y0 : y0 + source_height, x0 : x0 + source_width]
        # Map the prepared shirt rectangle onto the shoulder/hip frame-space
        # quadrilateral in source screen-left -> screen-right order. Every
        # placement value follows the current customer's center, shoulder
        # width and torso axis; no screen bands or fixed percentage placement
        # remain.
        long_sleeve = garment.template == "tshirt_long_sleeve"
        target_width = geometry.shoulder_span * (1.34 if long_sleeve else 1.26)
        target_height = geometry.torso_length * (1.38 if long_sleeve else 1.08)
        top_center = geometry.shoulder_center + geometry.torso_down_unit * geometry.torso_length * 0.025
        bottom_center = top_center + geometry.torso_down_unit * target_height
        half_width = geometry.across_unit * (target_width * 0.5)
        destination = np.asarray(
            [top_center - half_width, top_center + half_width, bottom_center + half_width, bottom_center - half_width],
            dtype=np.float32,
        )
        if garment_scale != 1.0:
            # A customer adjustment scales the whole placed shirt uniformly
            # around its fixed geometric center; pose and center never move.
            destination_center = destination.mean(axis=0)
            destination = (
                destination_center + (destination - destination_center) * garment_scale
            )
        source_corners = np.asarray([[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source_corners, destination)
        overlay = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = frame.copy()
        protected = self._protected_regions(geometry, width, height)
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0
        alpha[protected > 0] = 0.0
        alpha = alpha[:, :, None]
        result = (overlay[:, :, :3].astype(np.float32) * alpha + result.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("fast_result_encode")
        return encoded.tobytes()
