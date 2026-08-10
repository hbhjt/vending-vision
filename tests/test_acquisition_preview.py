import asyncio
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn

import app as vision_app
from vision.acquisition_preview import AcquisitionPreviewStore


def test_preview_reader_leases_are_bounded_and_released_after_a_dropped_body():
    """Only a fixed number of GET bodies may wait for an attempt snapshot."""
    async def scenario():
        store = AcquisitionPreviewStore(max_readers=2)
        token = await store.open("attempt", b"jpeg")
        first, second = await store.acquire(token), await store.acquire(token)
        assert first is not None and second is not None
        assert await store.reader_count() == 2
        with pytest.raises(RuntimeError, match="reader_limit"):
            await store.acquire(token)
        await store.release(first.lease_id)
        third = await store.acquire(token)
        assert third is not None
        await store.close("attempt")
        await store.release(second.lease_id)
        await store.release(third.lease_id)
        assert await store.reader_count() == 0

    asyncio.run(scenario())


def test_preview_close_waits_until_public_reader_leases_are_observably_zero():
    """Close is the cleanup barrier: callers must not see live stream leases after it returns."""
    async def scenario():
        store = AcquisitionPreviewStore(max_readers=2)
        token = await store.open("attempt-close", b"jpeg")
        lease = await store.acquire(token)
        assert lease is not None
        assert await store.reader_count() == 1

        closer = asyncio.create_task(store.close("attempt-close"))
        await asyncio.sleep(0.02)
        assert not closer.done(), "close returned while a public stream lease was still active"

        await store.release(lease.lease_id)
        await asyncio.wait_for(closer, timeout=0.5)
        assert await store.reader_count() == 0

    asyncio.run(scenario())


def _unused_loopback_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_public_preview_stream_bounds_real_http_readers_and_releases_on_close():
    """The public MJPEG route admits only two live GET bodies; rejects never queue."""
    original_store = vision_app._acquisition_previews
    store = AcquisitionPreviewStore(max_readers=2)
    vision_app._acquisition_previews = store
    token = asyncio.run(store.open("attempt-public", b"canonical-jpeg-body"))
    port = _unused_loopback_port()
    server = uvicorn.Server(
        uvicorn.Config(
            vision_app.app,
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.005)
        assert server.started

        url = f"http://127.0.0.1:{port}/v2/try-on/acquisition/preview.mjpeg?token={token}"
        with httpx.Client(timeout=2.0) as client:
            head = client.head(url)
            assert head.status_code == 200
            assert asyncio.run(store.reader_count()) == 0

            first_cm = client.stream("GET", url)
            first = first_cm.__enter__()
            first_iter = first.iter_raw()
            first_chunk = next(first_iter)
            assert b"canonical-jpeg-body" in first_chunk

            second_cm = client.stream("GET", url)
            second = second_cm.__enter__()
            second_iter = second.iter_raw()
            assert b"canonical-jpeg-body" in next(second_iter)
            assert asyncio.run(store.reader_count()) == 2

            started = time.monotonic()
            rejected = client.get(url)
            assert rejected.status_code == 429
            assert time.monotonic() - started < 0.5
            assert client.get(f"{url}&extra=1").status_code == 404

            def reject_once(_):
                response = client.get(url)
                return response.status_code

            probe_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=64) as pool:
                statuses = list(pool.map(reject_once, range(2000)))
            assert set(statuses) == {429}
            assert time.monotonic() - probe_started < 5
            assert asyncio.run(store.reader_count()) == 2

            first_cm.__exit__(None, None, None)
            deadline = time.monotonic() + 1
            while asyncio.run(store.reader_count()) != 1 and time.monotonic() < deadline:
                time.sleep(0.002)
            assert asyncio.run(store.reader_count()) == 1

            third_cm = client.stream("GET", url)
            third = third_cm.__enter__()
            third_iter = third.iter_raw()
            assert b"canonical-jpeg-body" in next(third_iter)
            assert asyncio.run(store.reader_count()) == 2

            asyncio.run(store.close("attempt-public"))
            assert next(second_iter, b"") == b""
            assert next(third_iter, b"") == b""
            second_cm.__exit__(None, None, None)
            third_cm.__exit__(None, None, None)

        deadline = time.monotonic() + 1
        while asyncio.run(store.reader_count()) != 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert asyncio.run(store.reader_count()) == 0
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        vision_app._acquisition_previews = original_store
