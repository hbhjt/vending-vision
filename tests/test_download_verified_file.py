from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.download_verified_file import DownloadError, download_verified_file


class Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class SlowTrickleResponse(Response):
    def __init__(self, payload: bytes, url: str, clock: Clock):
        super().__init__(payload, url)
        self._clock = clock

    def read(self, _size: int = -1) -> bytes:
        self._clock.advance(0.4)
        return super().read(1)


class RedirectAdvancingResponse(Response):
    def __init__(self, payload: bytes, url: str, clock: Clock):
        super().__init__(payload, url)
        self._clock = clock

    def geturl(self) -> str:
        self._clock.advance(2.0)
        return super().geturl()


def test_total_deadline_aborts_slow_trickle_and_closes_response(tmp_path):
    url = "https://example.invalid/slow.bin"
    payload = b"slow"
    destination = tmp_path / "slow.bin"
    clock = Clock()
    response = SlowTrickleResponse(payload, url, clock)

    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_opener_uses_remaining_socket_timeout_and_post_open_deadline(tmp_path):
    url = "https://example.invalid/open.bin"
    payload = b"open"
    destination = tmp_path / "open.bin"
    clock = Clock()
    response = Response(payload, url)
    observed_timeouts: list[float] = []

    def opener(_request, timeout):
        observed_timeouts.append(timeout)
        clock.advance(5.1)
        return response

    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=opener,
            total_timeout_seconds=5.0,
            monotonic=clock,
        )

    assert observed_timeouts == [pytest.approx(5.0)]
    assert response.closed
    assert not destination.exists()


def test_opener_socket_timeout_is_capped_below_long_total_deadline(tmp_path):
    url = "https://example.invalid/socket-cap.bin"
    payload = b"socket cap"
    observed_timeouts: list[float] = []

    def opener(_request, timeout):
        observed_timeouts.append(timeout)
        return Response(payload, url)

    destination = tmp_path / "socket-cap.bin"
    download_verified_file(
        url,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        destination,
        opener=opener,
        total_timeout_seconds=500.0,
        monotonic=Clock(),
    )

    assert observed_timeouts == [120.0]
    assert destination.read_bytes() == payload


def test_deadline_is_checked_after_final_redirect_identity(tmp_path):
    url = "https://example.invalid/redirect.bin"
    payload = b"redirect"
    destination = tmp_path / "redirect.bin"
    clock = Clock()
    response = RedirectAdvancingResponse(payload, url, clock)

    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()


def test_deadline_is_checked_after_fsync_before_digest_and_publish(
    tmp_path, monkeypatch
):
    url = "https://example.invalid/fsync.bin"
    payload = b"fsync"
    destination = tmp_path / "fsync.bin"
    clock = Clock()
    response = Response(payload, url)
    real_fsync = os.fsync

    def deadline_crossing_fsync(descriptor):
        real_fsync(descriptor)
        clock.advance(2.0)

    monkeypatch.setattr(
        "scripts.download_verified_file.os.fsync", deadline_crossing_fsync
    )
    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()


def test_deadline_crossing_after_exact_digest_does_not_publish(tmp_path, monkeypatch):
    url = "https://example.invalid/exact.bin"
    payload = b"exact bytes and digest"
    destination = tmp_path / "exact.bin"
    clock = Clock()
    response = Response(payload, url)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    real_sha256 = hashlib.sha256

    class DeadlineCrossingDigest:
        def __init__(self):
            self._digest = real_sha256()

        def update(self, chunk: bytes) -> None:
            self._digest.update(chunk)

        def hexdigest(self) -> str:
            value = self._digest.hexdigest()
            clock.advance(2.0)
            return value

    monkeypatch.setattr(
        "scripts.download_verified_file.hashlib.sha256", DeadlineCrossingDigest
    )

    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            url,
            expected_sha256,
            len(payload),
            destination,
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), 3600.1, True])
def test_total_deadline_is_finite_positive_and_capped(tmp_path, timeout):
    with pytest.raises(DownloadError, match="download_timeout"):
        download_verified_file(
            "https://example.invalid/file.bin",
            hashlib.sha256(b"x").hexdigest(),
            1,
            tmp_path / "file.bin",
            opener=lambda _request, timeout: Response(
                b"x", "https://example.invalid/file.bin"
            ),
            total_timeout_seconds=timeout,
        )


def test_publish_race_preserves_preexisting_destination_and_removes_temp(
    tmp_path, monkeypatch
):
    url = "https://example.invalid/race.bin"
    payload = b"verified bytes"
    destination = tmp_path / "race.bin"

    def racing_link(_source, target):
        Path(target).write_bytes(b"preexisting winner")
        raise FileExistsError

    monkeypatch.setattr("scripts.download_verified_file.os.link", racing_link)
    with pytest.raises(DownloadError, match="download_destination"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
            total_timeout_seconds=10.0,
            monotonic=Clock(),
        )

    assert destination.read_bytes() == b"preexisting winner"
    assert list(tmp_path.iterdir()) == [destination]


def test_bounded_file_downloader_publishes_only_exact_https_identity(tmp_path):
    url = "https://example.invalid/candidate.zip"
    payload = b"immutable candidate bytes"
    destination = tmp_path / "candidate.zip"

    download_verified_file(
        url,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        destination,
        opener=lambda _request, timeout: Response(payload, url),
    )

    assert destination.read_bytes() == payload


def test_bounded_file_downloader_aborts_expected_plus_one_without_destination(tmp_path):
    url = "https://example.invalid/model.zip"
    payload = b"expected-plus-one"
    destination = tmp_path / "model.zip"

    with pytest.raises(DownloadError, match="download_size"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload) - 1,
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["http", "redirect", "digest"])
def test_bounded_file_downloader_rejects_mutable_or_wrong_identity(tmp_path, mutation):
    expected_url = "https://example.invalid/proof.json"
    payload = b"proof"
    url = expected_url if mutation != "http" else "http://example.invalid/proof.json"
    response_url = (
        "https://cdn.example.invalid/proof.json" if mutation == "redirect" else url
    )
    digest = hashlib.sha256(payload).hexdigest() if mutation != "digest" else "0" * 64
    destination = tmp_path / "proof.json"

    with pytest.raises(DownloadError):
        download_verified_file(
            url,
            digest,
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, response_url),
        )

    assert not destination.exists()


def test_file_downloader_cli_imports_with_stdlib_only(tmp_path):
    environment = tmp_path / "clean-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = Path(__file__).parents[1] / "scripts" / "download_verified_file.py"

    completed = subprocess.run(
        [str(python), str(script), "--help"], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
