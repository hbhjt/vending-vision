"""Bounded, local-only Fast virtual try-on primitive.

The runtime receives only a daemon-issued loopback PNG descriptor.  It never
contacts platform services or follows redirects, and derives placement from the
verified image bounds plus the declared template rather than product anchors.
"""

from __future__ import annotations

import hashlib
import http.client
import struct
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


class GarmentFetchError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedGarment:
    rgba: np.ndarray
    digest: str
    template: str
    alpha_mask: np.ndarray
    opaque_bounds: tuple[int, int, int, int]


class FastTryOnRuntime:
    def __init__(self, max_garment_bytes: int = 8 * 1024 * 1024):
        self.max_garment_bytes = max_garment_bytes

    def fetch_garment(self, descriptor: dict, cancel_event=None) -> PreparedGarment:
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
        deadline = cv2.getTickCount() / cv2.getTickFrequency() + 5.0
        connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=1)
        try:
            target = url.path or "/"
            if url.query:
                target = f"{target}?{url.query}"
            if cancel_event is not None and cancel_event.is_set():
                raise GarmentFetchError("attempt_replaced")
            connection.request("GET", target)
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise GarmentFetchError("redirect")
            if response.status != 200 or response.getheader("Content-Type", "").split(";", 1)[0] != "image/png":
                raise GarmentFetchError("content_type")
            chunks: list[bytes] = []
            total = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    connection.close()
                    raise GarmentFetchError("attempt_replaced")
                if cv2.getTickCount() / cv2.getTickFrequency() > deadline:
                    connection.close()
                    raise GarmentFetchError("deadline")
                chunk = response.read(min(64 * 1024, self.max_garment_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_garment_bytes:
                    raise GarmentFetchError("byte_size")
            payload = b"".join(chunks)
        finally:
            connection.close()
        if len(payload) > self.max_garment_bytes or len(payload) != descriptor.get("byteSize"):
            raise GarmentFetchError("byte_size")
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != descriptor.get("digest"):
            raise GarmentFetchError("digest")
        self._predecode_png(payload)
        rgba = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise GarmentFetchError("png_decode")
        if descriptor.get("contentType") != "image/png" or descriptor.get("template") not in {"tshirt_short_sleeve", "tshirt_long_sleeve"}:
            raise GarmentFetchError("descriptor")
        alpha_mask = np.where(rgba[:, :, 3] >= 12, 255, 0).astype(np.uint8)
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
            template=descriptor["template"],
            alpha_mask=alpha_mask,
            opaque_bounds=(x, y, width, height),
        )

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

    def render(self, frame: np.ndarray, garment: PreparedGarment) -> bytes:
        height, width = frame.shape[:2]
        x0, y0, source_width, source_height = garment.opaque_bounds
        source = garment.rgba[y0 : y0 + source_height, x0 : x0 + source_width]
        # Fast placement is deterministic and Vision-owned.  It approximates
        # the upper-body pose on ordinary front frames by using shoulder/hip
        # bands in image coordinates, keeping the head band protected and
        # centering the garment on the torso instead of exposing anchors or
        # tuning fields through product APIs.
        head_protect_bottom = int(height * 0.23)
        shoulder_y = int(height * 0.30)
        hip_y = int(height * 0.72)
        torso_height = max(1, hip_y - shoulder_y)
        target_width = max(1, int(width * (0.56 if garment.template == "tshirt_short_sleeve" else 0.60)))
        target_height = max(1, int(target_width * source_height / source_width))
        target_height = min(target_height, max(1, int(torso_height * 1.08)))
        overlay = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
        x = (width - target_width) // 2
        y = max(head_protect_bottom, shoulder_y - int(target_height * 0.06))
        y = min(y, height - target_height)
        result = frame.copy()
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        region = result[y : y + target_height, x : x + target_width].astype(np.float32)
        result[y : y + target_height, x : x + target_width] = (
            overlay[:, :, :3].astype(np.float32) * alpha + region * (1.0 - alpha)
        ).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", result)
        if not ok:
            raise RuntimeError("fast_result_encode")
        return encoded.tobytes()
