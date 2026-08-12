"""Bounded stdlib-only downloader for an immutable HTTPS file identity."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_DOWNLOAD_BYTES = 16 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


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
        with opener(request, timeout=120.0) as response, os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            if response.geturl() != url:
                raise DownloadError("download_redirect_identity")
            while True:
                chunk = response.read(min(CHUNK_BYTES, expected_bytes - downloaded + 1))
                if not chunk:
                    break
                if downloaded + len(chunk) > expected_bytes:
                    raise DownloadError("download_size")
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if downloaded != expected_bytes:
            raise DownloadError("download_size")
        if digest.hexdigest() != sha256:
            raise DownloadError("download_digest")
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
    args = parser.parse_args()
    try:
        download_verified_file(
            args.url,
            args.sha256,
            args.expected_bytes,
            args.destination.resolve(),
        )
    except (DownloadError, OSError) as exc:
        print(f"VERIFIED_FILE_DOWNLOAD=FAIL:{exc}")
        return 1
    print("VERIFIED_FILE_DOWNLOAD=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
