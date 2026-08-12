"""Bounded stdlib-only downloader for an immutable HTTPS file identity."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 120.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 1800.0
MAX_TOTAL_TIMEOUT_SECONDS = 3600.0


class DownloadError(RuntimeError):
    pass


def download_verified_file(
    url: str,
    sha256: str,
    expected_bytes: int,
    destination: Path,
    *,
    opener=urlopen,
    max_download_bytes: int = MAX_DOWNLOAD_BYTES,
    total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
    monotonic=time.monotonic,
) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.fragment
        or re.fullmatch(r"[a-f0-9]{64}", sha256) is None
    ):
        raise DownloadError("download_identity")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(max_download_bytes) is not int
        or max_download_bytes <= 0
        or expected_bytes > max_download_bytes
    ):
        raise DownloadError("download_size")
    if (
        type(total_timeout_seconds) not in {int, float}
        or not math.isfinite(total_timeout_seconds)
        or total_timeout_seconds <= 0
        or total_timeout_seconds > MAX_TOTAL_TIMEOUT_SECONDS
    ):
        raise DownloadError("download_timeout")
    deadline = monotonic() + total_timeout_seconds

    def remaining() -> float:
        value = deadline - monotonic()
        if not math.isfinite(value) or value <= 0:
            raise DownloadError("download_timeout")
        return value

    remaining()
    if destination.exists() or destination.is_symlink():
        raise DownloadError("download_destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        downloaded = 0
        request = Request(url, headers={"User-Agent": "vending-vision-proof-fetcher/1"})
        response_timeout = min(SOCKET_TIMEOUT_SECONDS, remaining())
        with opener(request, timeout=response_timeout) as response, os.fdopen(
            descriptor, "wb"
        ) as output:
            descriptor = -1
            remaining()
            response_url = response.geturl()
            remaining()
            if response_url != url:
                raise DownloadError("download_redirect_identity")
            while True:
                remaining()
                chunk = response.read(min(CHUNK_BYTES, expected_bytes - downloaded + 1))
                remaining()
                if not chunk:
                    break
                if downloaded + len(chunk) > expected_bytes:
                    raise DownloadError("download_size")
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
            remaining()
            output.flush()
            remaining()
            os.fsync(output.fileno())
            remaining()
        remaining()
        if downloaded != expected_bytes:
            raise DownloadError("download_size")
        remaining()
        actual_sha256 = digest.hexdigest()
        remaining()
        if actual_sha256 != sha256:
            raise DownloadError("download_digest")
        remaining()
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise DownloadError("download_destination") from exc
        temporary.unlink()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--total-timeout-seconds", required=True, type=float)
    args = parser.parse_args()
    try:
        download_verified_file(
            args.url,
            args.sha256,
            args.expected_bytes,
            args.destination.resolve(),
            total_timeout_seconds=args.total_timeout_seconds,
        )
    except (DownloadError, OSError) as exc:
        print(f"VERIFIED_FILE_DOWNLOAD=FAIL:{exc}")
        return 1
    print("VERIFIED_FILE_DOWNLOAD=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
