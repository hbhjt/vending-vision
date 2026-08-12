from __future__ import annotations

import hashlib
import io
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
