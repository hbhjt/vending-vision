"""Bounded, local-only garment composition primitive.

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
    sleeve_semantics: str
    quality: "GarmentQualityFacts"


@dataclass(frozen=True)
class TransparentGarmentSource:
    png_bytes: bytes
    digest: str
    template: str


@dataclass(frozen=True)
class GarmentQualityFacts:
    digest: str
    opaque_pixel_count: int
    opaque_ratio: float
    source_aspect_ratio: float
    sleeve_semantics: str


@dataclass(frozen=True)
class _SleeveContourFacts:
    """Source-only evidence used to validate a declared sleeve template."""

    has_bilateral_short_sleeves: bool
    has_wrist_length_sleeves: bool


@dataclass(frozen=True)
class CompositionGeometryFacts:
    garment_digest: str
    source_aspect_ratio: float
    placed_aspect_ratio: float
    center: tuple[float, float]
    width: float
    height: float
    rotation_degrees: float


@dataclass(frozen=True)
class CompositionResult:
    png: bytes
    geometry: CompositionGeometryFacts


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


def _source_row_edges(rows: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return source-mask left/right opaque edges for each supplied row."""
    left = np.argmax(rows, axis=1)
    right = width - 1 - np.argmax(rows[:, ::-1], axis=1)
    return left, right


def _classify_source_sleeve_contour(source_bbox: np.ndarray) -> _SleeveContourFacts:
    """Classify source-only short and wrist-length sleeve silhouette evidence."""
    height, width = source_bbox.shape
    row_widths = np.count_nonzero(source_bbox, axis=1)

    # A short-sleeve declaration needs sustained, roughly symmetric expansion
    # on both sides of a lower-torso reference; a wide rectangle alone is not
    # sufficient evidence.
    torso_rows = source_bbox[height * 3 // 4 : height * 19 // 20]
    torso_lefts, torso_rights = _source_row_edges(torso_rows, width)
    torso_left = float(np.median(torso_lefts))
    torso_right = float(np.median(torso_rights))
    torso_width = torso_right - torso_left + 1.0
    shoulder_rows = source_bbox[height * 3 // 20 : height * 3 // 5]
    shoulder_lefts, shoulder_rights = _source_row_edges(shoulder_rows, width)
    left_expansion = torso_left - shoulder_lefts
    right_expansion = shoulder_rights - torso_right
    minimum_expansion = max(3.0, torso_width * 0.08)
    bilateral_sleeve_rows = (
        (left_expansion >= minimum_expansion)
        & (right_expansion >= minimum_expansion)
        & (
            np.minimum(left_expansion, right_expansion)
            / np.maximum(1.0, np.maximum(left_expansion, right_expansion))
            >= 0.55
        )
    )
    has_bilateral_short_sleeves = np.count_nonzero(bilateral_sleeve_rows) >= max(
        5, len(shoulder_rows) // 5
    )

    # Wrist-length sleeves are a distinct lower-body expansion relative to
    # the terminal torso width.
    terminal_width = float(np.quantile(row_widths[height * 3 // 4 :], 0.15))
    terminal_rows = source_bbox[row_widths <= terminal_width * 1.05]
    terminal_lefts, terminal_rights = _source_row_edges(terminal_rows, width)
    terminal_left = float(np.median(terminal_lefts))
    terminal_right = float(np.median(terminal_rights))
    lower_start, lower_stop = height * 3 // 5, height * 19 // 20
    lower_rows = source_bbox[lower_start:lower_stop]
    lower_widths = row_widths[lower_start:lower_stop]
    lower_left, lower_right = _source_row_edges(lower_rows, width)
    outward = max(2.0, width * 0.05)
    expanded_bilateral_rows = (
        (lower_widths >= terminal_width * 1.25)
        & (lower_left <= terminal_left - outward)
        & (lower_right >= terminal_right + outward)
    )
    has_wrist_length_sleeves = np.count_nonzero(expanded_bilateral_rows) >= max(
        3, len(lower_rows) // 4
    )
    return _SleeveContourFacts(
        has_bilateral_short_sleeves=has_bilateral_short_sleeves,
        has_wrist_length_sleeves=has_wrist_length_sleeves,
    )


class GarmentComposer:
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
    ) -> TransparentGarmentSource:
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
        # The daemon's local media read can briefly block behind a catalog
        # adoption or media-cache transaction. A sub-second read deadline
        # therefore turned an occasional busy daemon into a spurious
        # garment_rejected terminal; the attempt-scoped 15s generation budget
        # remains the authoritative outer deadline.
        timeout = httpx.Timeout(connect=2.0, read=2.0, write=1.0, pool=1.0)
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
        return TransparentGarmentSource(
            png_bytes=payload,
            digest=digest,
            template=descriptor["template"],
        )

    def prepare_garment(self, source: TransparentGarmentSource) -> PreparedGarment:
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
        if not np.any(rgba[:, :, 3] < 12):
            raise GarmentFetchError("transparent_boundary")
        if not np.any(alpha_mask):
            raise GarmentFetchError("transparent_png")
        # Component validity is a source-quality fact. It must precede the
        # boundary-noise close, which can otherwise fuse distinct silhouettes.
        raw_component_count, _labels = cv2.connectedComponents(alpha_mask)
        if raw_component_count != 2:
            raise GarmentFetchError("garment_quality")
        source_alpha_mask = alpha_mask
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
        opaque_pixel_count = int(np.count_nonzero(alpha_mask))
        opaque_ratio = opaque_pixel_count / float(alpha_mask.size)
        if opaque_ratio < 0.04 or width < 8 or height < 8:
            raise GarmentFetchError("garment_quality")
        component_count, _labels = cv2.connectedComponents(alpha_mask)
        if component_count != 2:
            raise GarmentFetchError("garment_quality")
        if x == 0 or y == 0 or x + width == rgba.shape[1] or y + height == rgba.shape[0]:
            raise GarmentFetchError("garment_cropped")
        sleeve_semantics = "long" if source.template == "tshirt_long_sleeve" else "short"
        source_bbox = source_alpha_mask[y : y + height, x : x + width] > 0
        sleeve_contour = _classify_source_sleeve_contour(source_bbox)
        if sleeve_semantics == "long" and not sleeve_contour.has_wrist_length_sleeves:
            raise GarmentFetchError("template_mismatch")
        if sleeve_semantics == "short" and sleeve_contour.has_wrist_length_sleeves:
            raise GarmentFetchError("template_mismatch")
        if sleeve_semantics == "short" and not sleeve_contour.has_bilateral_short_sleeves:
            raise GarmentFetchError("garment_quality")
        # This preparation is attempt-local: it is deterministically derived
        # from source PNG + template + verified digest and is discarded after
        # rendering. No garment anchors or product tuning cross the boundary.
        return PreparedGarment(
            rgba=rgba,
            digest=digest,
            template=source.template,
            alpha_mask=alpha_mask,
            opaque_bounds=(x, y, width, height),
            sleeve_semantics=sleeve_semantics,
            quality=GarmentQualityFacts(
                digest=digest,
                opaque_pixel_count=opaque_pixel_count,
                opaque_ratio=opaque_ratio,
                source_aspect_ratio=width / float(height),
                sleeve_semantics=sleeve_semantics,
            ),
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
    def _foreground_occlusion(geometry: PoseGeometry, width: int, height: int, sleeve_semantics: str) -> np.ndarray:
        """Return only the person pixels that correctly belong in front."""
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
            if sleeve_semantics == "short" and wrist is not None:
                # A short sleeve owns shoulder-to-elbow. Only the exposed
                # forearm and hand return to the camera foreground.
                start = shoulder + (elbow - shoulder) * 0.72
                cv2.line(protected, tuple(np.rint(start).astype(int)), tuple(np.rint(wrist).astype(int)), 255, max(8, round(span * 0.12)), cv2.LINE_AA)
            elif sleeve_semantics == "long" and wrist is not None:
                across_distance = abs(float(np.dot(wrist - geometry.shoulder_center, geometry.across_unit)))
                if across_distance <= span * 0.65:
                    cv2.line(protected, tuple(np.rint(elbow).astype(int)), tuple(np.rint(wrist).astype(int)), 255, max(9, round(span * 0.14)), cv2.LINE_AA)
            if wrist is not None:
                palm = wrist + (wrist - elbow) * 0.08
                cv2.circle(protected, tuple(np.rint(palm).astype(int)), max(8, round(span * 0.09)), 255, -1, cv2.LINE_AA)
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

    def compose(
        self,
        captured_frame: np.ndarray,
        transparent_garment_source: PreparedGarment | TransparentGarmentSource,
        scale: float,
    ) -> CompositionResult:
        """Compose one captured frame and source into PNG plus stable facts."""
        garment = transparent_garment_source
        if isinstance(garment, TransparentGarmentSource):
            garment = self.prepare_garment(garment)
        garment_scale = float(scale)
        if not math.isfinite(garment_scale) or garment_scale < 0.8 or garment_scale > 1.6:
            raise PoseUnavailableError("pose_unavailable")
        frame = captured_frame
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise PoseUnavailableError("pose_unavailable")
        height, width = frame.shape[:2]
        geometry = self._resolve_pose(frame)
        rotation_degrees = float(math.degrees(math.atan2(geometry.across_unit[1], geometry.across_unit[0])))
        if abs(rotation_degrees) > 30.0:
            raise PoseUnavailableError("pose_unavailable")
        x0, y0, source_width, source_height = garment.opaque_bounds
        source = garment.rgba[y0 : y0 + source_height, x0 : x0 + source_width]
        desired_width = geometry.shoulder_span * (1.90 if garment.sleeve_semantics == "long" else 1.26)
        desired_height = geometry.torso_length * (1.38 if garment.sleeve_semantics == "long" else 1.08)
        base_uniform_scale = min(desired_width / source_width, desired_height / source_height)
        # The placement centre belongs to the captured person, not to a
        # particular adjustment. Customer scaling expands from this locked
        # point and therefore cannot make the garment drift down the torso.
        center = geometry.shoulder_center + geometry.torso_down_unit * (
            source_height * base_uniform_scale * 0.5 + geometry.torso_length * 0.025
        )
        uniform_scale = base_uniform_scale * garment_scale
        placed_width, placed_height = source_width * uniform_scale, source_height * uniform_scale
        source_center = np.array([source_width * 0.5, source_height * 0.5], dtype=np.float32)
        # The shoulder axis decides rotation.  Hips determine only torso
        # placement and scale; using the raw shoulder-to-hip vector here would
        # shear the PNG whenever those two observed axes are not perpendicular.
        rotation_down = np.array(
            [-geometry.across_unit[1], geometry.across_unit[0]], dtype=np.float32
        )
        if float(np.dot(rotation_down, geometry.torso_down_unit)) < 0:
            rotation_down = -rotation_down
        basis = np.column_stack((geometry.across_unit, rotation_down)) * uniform_scale
        offset = center - basis @ source_center
        transform = np.column_stack((basis, offset)).astype(np.float32)
        overlay = cv2.warpAffine(source, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        result = frame.copy()
        protected = self._foreground_occlusion(geometry, width, height, garment.sleeve_semantics)
        alpha = overlay[:, :, 3].astype(np.float32) / 255.0
        alpha[protected > 0] = 0.0
        alpha = alpha[:, :, None]
        result = (overlay[:, :, :3].astype(np.float32) * alpha + result.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("garment_result_encode")
        return CompositionResult(
            png=encoded.tobytes(),
            geometry=CompositionGeometryFacts(
                garment_digest=garment.digest,
                source_aspect_ratio=source_width / float(source_height),
                placed_aspect_ratio=placed_width / placed_height,
                center=(float(center[0]), float(center[1])),
                width=float(placed_width),
                height=float(placed_height),
                rotation_degrees=rotation_degrees,
            ),
        )
