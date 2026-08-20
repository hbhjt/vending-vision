import asyncio
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest


def png_bytes(color=(20, 120, 220, 255)):
    image = np.zeros((48, 36, 4), dtype=np.uint8)
    image[3:45, 10:25] = color  # torso
    image[8:28, 3:10] = color  # short left sleeve
    image[8:28, 25:33] = color  # short right sleeve
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class GarmentHandler(BaseHTTPRequestHandler):
    payload = png_bytes()
    status = 200
    content_type = "image/png"
    redirect = False

    def do_GET(self):
        if self.redirect:
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/garment")
            self.end_headers()
            return
        self.send_response(self.status)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_):
        return


class SlowDripHandler(BaseHTTPRequestHandler):
    streaming = threading.Event()
    closed = threading.Event()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        try:
            # The cancellation assertion must begin only after this real HTTP
            # response has become a slow body stream, not merely after a task
            # has been scheduled to connect to the local server.
            self.wfile.write(b"x")
            self.wfile.flush()
            self.streaming.set()
            while True:
                self.wfile.write(b"x")
                self.wfile.flush()
                threading.Event().wait(0.25)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.closed.set()

    def log_message(self, *_):
        return


@pytest.fixture
def garment_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), GarmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/garment?token=opaque"
    server.shutdown()
    server.server_close()
    thread.join()


def test_garment_source_downloads_only_declared_loopback_png_and_composes_a_decodable_result(
    garment_server,
):
    from vision.garment_composer import GarmentComposer

    garment = GarmentHandler.payload
    class DeterministicPoseEstimator:
        """Explicit test-estimator boundary; production obtains MediaPipe in the worker."""

        def detect(self, _frame):
            points = [type("Point", (), {"x": 0.5, "y": 0.5, "visibility": 0.95})() for _ in range(33)]
            for index, x, y in ((11, 0.35, 0.32), (12, 0.65, 0.32), (23, 0.38, 0.68), (24, 0.62, 0.68)):
                points[index] = type("Point", (), {"x": x, "y": y, "visibility": 0.95})()
            return type("Pose", (), {"pose_landmarks": type("Landmarks", (), {"landmark": points})()})()

    runtime = GarmentComposer(
        max_garment_bytes=1024 * 1024, pose_estimator=DeterministicPoseEstimator()
    )
    prepared = asyncio.run(runtime.fetch_garment(
        {
            "reference": garment_server,
            "digest": "sha256:" + hashlib.sha256(garment).hexdigest(),
            "contentType": "image/png",
            "byteSize": len(garment),
            "template": "tshirt_short_sleeve",
        }
    ))
    frame = np.full((160, 120, 3), (235, 220, 205), dtype=np.uint8)
    result = runtime.compose(frame, prepared, 1.0).png

    decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == frame.shape[:2]
    assert not np.array_equal(decoded, frame)
    assert np.array_equal(decoded[: int(frame.shape[0] * 0.20)], frame[: int(frame.shape[0] * 0.20)])
    torso = decoded[int(frame.shape[0] * 0.34) : int(frame.shape[0] * 0.68)]
    assert np.count_nonzero(torso[:, :, 0] != frame[int(frame.shape[0] * 0.34) : int(frame.shape[0] * 0.68), :, 0]) > 100


def test_garment_source_rejects_redirect_and_digest_mismatch(garment_server):
    from vision.garment_composer import GarmentComposer, GarmentFetchError

    runtime = GarmentComposer(max_garment_bytes=1024 * 1024)
    descriptor = {
        "reference": garment_server,
        "digest": "sha256:" + "0" * 64,
        "contentType": "image/png",
        "byteSize": len(GarmentHandler.payload),
        "template": "tshirt_short_sleeve",
    }
    with pytest.raises(GarmentFetchError, match="digest"):
        asyncio.run(runtime.fetch_garment(descriptor))

    GarmentHandler.redirect = True
    try:
        with pytest.raises(GarmentFetchError, match="redirect"):
            asyncio.run(runtime.fetch_garment({**descriptor, "digest": "sha256:" + hashlib.sha256(GarmentHandler.payload).hexdigest()}))
    finally:
        GarmentHandler.redirect = False


def test_garment_source_rejects_png_bomb_dimensions_before_decode(garment_server):
    from vision.garment_composer import GarmentComposer, GarmentFetchError

    oversized_ihdr = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (4097).to_bytes(4, "big")
        + (32).to_bytes(4, "big")
        + bytes([8, 6, 0, 0, 0])
        + b"\x00\x00\x00\x00"
        + (0).to_bytes(4, "big")
        + b"IEND"
        + b"\x00\x00\x00\x00"
    )
    GarmentHandler.payload = oversized_ihdr
    try:
        runtime = GarmentComposer(max_garment_bytes=1024 * 1024)
        with pytest.raises(GarmentFetchError, match="png_dimensions"):
            asyncio.run(runtime.fetch_garment(
                {
                    "reference": garment_server,
                    "digest": "sha256:" + hashlib.sha256(oversized_ihdr).hexdigest(),
                    "contentType": "image/png",
                    "byteSize": len(oversized_ihdr),
                    "template": "tshirt_short_sleeve",
                }
            ))
    finally:
        GarmentHandler.payload = png_bytes()


def test_garment_source_cancels_a_slow_drip_and_closes_its_stream():
    """A replacement never leaves a blocking response reader behind."""
    SlowDripHandler.streaming.clear()
    SlowDripHandler.closed.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowDripHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        runtime = GarmentComposer(max_garment_bytes=1024 * 1024)
        canceled = asyncio.Event()
        task = asyncio.create_task(
            runtime.fetch_garment(
                {
                    "reference": f"http://127.0.0.1:{server.server_port}/garment?token=opaque",
                    "digest": "sha256:" + "0" * 64,
                    "contentType": "image/png",
                    "byteSize": 1024,
                    "template": "tshirt_short_sleeve",
                },
                canceled,
            )
        )
        # This only bounds a hung test fixture.  It does not change Fast's
        # five-second fetch deadline or its cancellation contract.
        assert await asyncio.to_thread(SlowDripHandler.streaming.wait, 5.0)
        canceled.set()
        with pytest.raises(GarmentFetchError, match="attempt_canceled"):
            await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
        assert await asyncio.to_thread(SlowDripHandler.closed.wait, 5.0)

    from vision.garment_composer import GarmentComposer, GarmentFetchError

    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
