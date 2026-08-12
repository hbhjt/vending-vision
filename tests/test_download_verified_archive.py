import hashlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import zipfile

import pytest

from scripts.download_verified_archive import ArchiveError, download_verified_archive


class _Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _SlowResponse(_Response):
    def __init__(self, payload, url, clock):
        super().__init__(payload, url)
        self._clock = clock

    def read(self, _size=-1):
        self._clock.advance(0.4)
        return super().read(1)


class _ExitAdvancingResponse(_Response):
    def __init__(self, payload, url, clock):
        super().__init__(payload, url)
        self._clock = clock

    def __exit__(self, *_args):
        super().__exit__(*_args)
        self._clock.advance(2.0)


def test_archive_total_deadline_aborts_slow_trickle_and_cleans_workdir(tmp_path):
    url = "https://example.invalid/slow.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _SlowResponse(payload, url, clock)

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_archive_opener_uses_remaining_timeout_and_checks_post_open(tmp_path):
    url = "https://example.invalid/open.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _Response(payload, url)
    timeouts = []

    def opener(_request, timeout):
        timeouts.append(timeout)
        clock.advance(5.1)
        return response

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=opener,
            total_timeout_seconds=5.0,
            monotonic=clock,
        )

    assert timeouts == [pytest.approx(5.0)]
    assert response.closed
    assert not destination.exists()


def test_archive_deadline_after_exact_digest_does_not_publish(tmp_path, monkeypatch):
    url = "https://example.invalid/exact.zip"
    payload = _zip("demo.whl")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _ExitAdvancingResponse(payload, url, clock)

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            expected_sha256,
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), 3600.1, True])
def test_archive_total_timeout_is_finite_positive_and_capped(tmp_path, timeout):
    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            "https://example.invalid/file.zip",
            "0" * 64,
            tmp_path / "wheelhouse",
            expected_bytes=1,
            opener=lambda _request, timeout: _Response(
                b"x", "https://example.invalid/file.zip"
            ),
            total_timeout_seconds=timeout,
        )


def _zip(member):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, b"wheel")
    return output.getvalue()


def test_verified_archive_download_extracts_only_digest_bound_safe_members(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("demo.whl")

    def opener(_request, timeout):
        assert timeout == 120.0
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    download_verified_archive(
        url, hashlib.sha256(payload).hexdigest(), destination,
        expected_bytes=len(payload), opener=opener,
    )

    assert (destination / "demo.whl").read_bytes() == b"wheel"


def test_verified_archive_rejects_traversal_without_publishing_destination(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("../escape.whl")

    def opener(_request, timeout):
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    with pytest.raises(ArchiveError, match="archive_unsafe_path"):
        download_verified_archive(
            url, hashlib.sha256(payload).hexdigest(), destination,
            expected_bytes=len(payload), opener=opener,
        )

    assert not destination.exists()
    assert not (tmp_path / "escape.whl").exists()


@pytest.mark.parametrize("case", ["symlink", "collision", "oversize", "special"])
def test_verified_archive_rejects_symlink_special_collision_and_size(tmp_path, case):
    if case == "special":
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("fifo")
            member.type = tarfile.FIFOTYPE
            archive.addfile(member)
        payload = output.getvalue()
    else:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            if case == "symlink":
                member = zipfile.ZipInfo("link")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "target")
            elif case == "collision":
                archive.writestr("Demo.whl", b"one")
                archive.writestr("demo.whl", b"two")
            else:
                archive.writestr("large.whl", b"12345")
        payload = output.getvalue()
    url = "https://example.invalid/core-wheelhouse.archive"

    def opener(_request, timeout):
        return _Response(payload, url)

    with pytest.raises(ArchiveError, match="archive_"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload),
            opener=opener,
            max_extracted_bytes=4 if case == "oversize" else 1024,
        )
    assert not (tmp_path / "wheelhouse").exists()


def test_archive_downloader_imports_in_clean_stdlib_only_python(tmp_path):
    clean_venv = tmp_path / "clean-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(clean_venv)],
        check=True,
    )
    clean_python = clean_venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = Path(__file__).parents[1] / "scripts" / "download_verified_archive.py"

    completed = subprocess.run(
        [str(clean_python), str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_archive_downloader_aborts_oversized_stream_before_writing_chunk(tmp_path, monkeypatch):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = b"expected-plus-one"
    writes = []
    real_open = Path.open

    class Sink(io.BytesIO):
        def write(self, value):
            writes.append(len(value))
            return super().write(value)

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name == "payload.archive" and mode == "xb":
            return Sink()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ArchiveError, match="archive_download_size"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload) - 1,
            max_download_bytes=len(payload) + 10,
            opener=lambda _request, timeout: _Response(payload, url),
        )

    assert writes == []
    assert not (tmp_path / "wheelhouse").exists()


def test_verified_archive_rejects_member_count_before_extracting(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("one.whl", b"one")
        archive.writestr("two.whl", b"two")
    payload = output.getvalue()

    with pytest.raises(ArchiveError, match="archive_member_count"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload),
            max_members=1,
            opener=lambda _request, timeout: _Response(payload, url),
        )
    assert not (tmp_path / "wheelhouse").exists()
