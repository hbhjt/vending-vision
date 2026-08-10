"""Bounded, local-only Fast virtual try-on primitive.

The runtime receives only a daemon-issued loopback PNG descriptor.  It never
contacts platform services or follows redirects, and derives placement from the
verified image bounds plus the declared template rather than product anchors.
"""

from __future__ import annotations

import hashlib
import http.client
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

    def fetch_garment(self, descriptor: dict) -> PreparedGarment:
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
        connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=5)
        try:
            target = url.path or "/"
            if url.query:
                target = f"{target}?{url.query}"
            connection.request("GET", target)
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise GarmentFetchError("redirect")
            if response.status != 200 or response.getheader("Content-Type", "").split(";", 1)[0] != "image/png":
                raise GarmentFetchError("content_type")
            payload = response.read(self.max_garment_bytes + 1)
        finally:
            connection.close()
        if len(payload) > self.max_garment_bytes or len(payload) != descriptor.get("byteSize"):
            raise GarmentFetchError("byte_size")
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest != descriptor.get("digest"):
            raise GarmentFetchError("digest")
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise GarmentFetchError("png_magic")
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

    def render(self, frame: np.ndarray, garment: PreparedGarment) -> bytes:
        height, width = frame.shape[:2]
        x0, y0, source_width, source_height = garment.opaque_bounds
        source = garment.rgba[y0 : y0 + source_height, x0 : x0 + source_width]
        target_width = max(1, int(width * (0.54 if garment.template == "tshirt_short_sleeve" else 0.58)))
        target_height = max(1, int(target_width * source_height / source_width))
        target_height = min(target_height, max(1, int(height * 0.64)))
        overlay = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)
        x = (width - target_width) // 2
        y = max(0, int(height * 0.20))
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
