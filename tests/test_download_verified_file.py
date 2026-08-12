from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import stat
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


def test_post_link_temp_unlink_failure_rolls_back_only_created_destination(
    tmp_path, monkeypatch
):
    url = "https://example.invalid/unlink-failure.bin"
    payload = b"verified post-link bytes"
    destination = tmp_path / "unlink-failure.bin"
    real_unlink = os.unlink
    failed_once = False

    def transient_temp_unlink(path, *args, **kwargs):
        nonlocal failed_once
        if not failed_once and Path(path).name.startswith(f".{destination.name}-"):
            failed_once = True
            raise OSError("injected post-link temp unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr("scripts.download_verified_file.os.unlink", transient_temp_unlink)
    with pytest.raises(OSError, match="post-link temp unlink"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert failed_once
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_post_link_parent_fsync_failure_rolls_back_destination(tmp_path, monkeypatch):
    url = "https://example.invalid/parent-fsync.bin"
    payload = b"parent fsync"
    destination = tmp_path / "parent-fsync.bin"
    real_fsync = os.fsync
    calls = 0

    def fail_first_parent_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr("scripts.download_verified_file.os.fsync", fail_first_parent_fsync)
    with pytest.raises(OSError, match="parent fsync failure"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert calls >= 3  # file, failing first parent fsync, rollback parent fsync
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_post_link_failure_preserves_an_identity_replacement(tmp_path, monkeypatch):
    url = "https://example.invalid/replaced.bin"
    payload = b"created identity"
    destination = tmp_path / "replaced.bin"
    other = b"other actor replacement"
    real_unlink = os.unlink
    failed_once = False

    def replace_then_fail_temp_unlink(path, *args, **kwargs):
        nonlocal failed_once
        if not failed_once and Path(path).name.startswith(f".{destination.name}-"):
            failed_once = True
            replacement = tmp_path / "other.tmp"
            replacement.write_bytes(other)
            os.replace(replacement, destination)
            raise OSError("injected replacement window")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(
        "scripts.download_verified_file.os.unlink", replace_then_fail_temp_unlink
    )
    with pytest.raises(OSError, match="replacement window"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert destination.read_bytes() == other
    assert [path.name for path in tmp_path.iterdir()] == [destination.name]


def test_final_identity_change_after_durable_cleanup_preserves_replacement(
    tmp_path, monkeypatch
):
    url = "https://example.invalid/final-replaced.bin"
    payload = b"created final identity"
    destination = tmp_path / "final-replaced.bin"
    other = b"other final identity"
    real_fsync = os.fsync
    fsync_calls = 0

    def replace_after_second_parent_fsync(descriptor):
        nonlocal fsync_calls
        result = real_fsync(descriptor)
        facts = os.fstat(descriptor)
        if stat.S_ISDIR(facts.st_mode):
            fsync_calls += 1
            if fsync_calls == 2:
                replacement = tmp_path / "other-final.tmp"
                replacement.write_bytes(other)
                os.replace(replacement, destination)
        return result

    monkeypatch.setattr(
        "scripts.download_verified_file.os.fsync", replace_after_second_parent_fsync
    )
    with pytest.raises(DownloadError, match="destination_identity"):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert destination.read_bytes() == other
    assert [path.name for path in tmp_path.iterdir()] == [destination.name]


def test_normal_publish_fsyncs_file_and_parent_around_temp_cleanup(
    tmp_path, monkeypatch
):
    url = "https://example.invalid/ordered.bin"
    payload = b"ordered durable publish"
    destination = tmp_path / "ordered.bin"
    events: list[str] = []
    real_fsync = os.fsync
    real_link = os.link
    real_unlink = os.unlink

    def record_fsync(descriptor):
        facts = os.fstat(descriptor)
        events.append(
            "parent-fsync" if stat.S_ISDIR(facts.st_mode) else "file-fsync"
        )
        return real_fsync(descriptor)

    def record_link(source, target):
        events.append("link")
        return real_link(source, target)

    def record_unlink(path, *args, **kwargs):
        if Path(path).name.startswith(f".{destination.name}-"):
            events.append("temp-unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr("scripts.download_verified_file.os.fsync", record_fsync)
    monkeypatch.setattr("scripts.download_verified_file.os.link", record_link)
    monkeypatch.setattr("scripts.download_verified_file.os.unlink", record_unlink)
    download_verified_file(
        url,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        destination,
        opener=lambda _request, timeout: Response(payload, url),
    )

    assert events == [
        "file-fsync",
        "link",
        "parent-fsync",
        "temp-unlink",
        "parent-fsync",
    ]
    assert destination.read_bytes() == payload


def test_rollback_cleanup_failure_never_returns_success(tmp_path, monkeypatch):
    url = "https://example.invalid/rollback-failure.bin"
    payload = b"rollback failure"
    destination = tmp_path / "rollback-failure.bin"
    real_fsync = os.fsync
    real_unlink = os.unlink
    fsync_calls = 0
    temp_failed = False

    def fail_rollback_parent_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("publish parent fsync failure")
        if fsync_calls == 3:
            raise OSError("rollback parent fsync failure")
        return real_fsync(descriptor)

    def fail_first_temp_cleanup(path, *args, **kwargs):
        nonlocal temp_failed
        if not temp_failed and Path(path).name.startswith(f".{destination.name}-"):
            temp_failed = True
            raise OSError("temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr("scripts.download_verified_file.os.fsync", fail_rollback_parent_fsync)
    monkeypatch.setattr("scripts.download_verified_file.os.unlink", fail_first_temp_cleanup)
    with pytest.raises(OSError):
        download_verified_file(
            url,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            destination,
            opener=lambda _request, timeout: Response(payload, url),
        )

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


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
