"""Bounded stdlib-only downloader for an immutable HTTPS file identity."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
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


class DownloadRecoveryError(DownloadError):
    def __init__(
        self,
        recovery: Path,
        identity: tuple[int, int, int, int],
        failures: tuple[BaseException, ...] = (),
    ):
        self.recovery = recovery
        self.identity = identity
        self.failures = failures
        super().__init__(
            f"download_replacement_recovered:{recovery}:"
            f"{identity[0]}:{identity[1]}:{identity[2]}:{identity[3]}"
        )


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    facts = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(facts.st_mode):
        raise DownloadError("download_destination_identity")
    return facts.st_dev, facts.st_ino, facts.st_mode, facts.st_size


def _descriptor_identity(descriptor: int) -> tuple[int, int, int, int]:
    facts = os.fstat(descriptor)
    if not stat.S_ISREG(facts.st_mode):
        raise DownloadError("download_destination_identity")
    return facts.st_dev, facts.st_ino, facts.st_mode, facts.st_size


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        api = _WindowsFileApi()
        handle = api.open_directory(directory)
        try:
            api.flush(handle)
        finally:
            api.close(handle)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _WindowsFileApi:
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    def __init__(self, kernel32=None):
        if kernel32 is None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel32 = kernel32

    def _open(self, path: Path, flags: int):
        handle = self.kernel32.CreateFileW(
            str(path),
            self.GENERIC_READ,
            self.FILE_SHARE_READ | self.FILE_SHARE_DELETE,
            None,
            self.OPEN_EXISTING,
            flags,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, 0, -1, invalid}:
            raise DownloadError("download_integrity_handle_open")
        return handle

    def open_file(self, path: Path):
        return self._open(path, self.FILE_ATTRIBUTE_NORMAL)

    def open_directory(self, path: Path):
        return self._open(path, self.FILE_FLAG_BACKUP_SEMANTICS)

    def flush(self, handle) -> None:
        if not self.kernel32.FlushFileBuffers(handle):
            raise DownloadError("download_directory_flush")

    def close(self, handle) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise DownloadError("download_integrity_handle_close")


class _HeldFile:
    def __init__(self, path: Path):
        self.path = path
        self.windows_api = _WindowsFileApi() if os.name == "nt" else None
        if self.windows_api is not None:
            self.handle = self.windows_api.open_file(path)
            self.identity = _file_identity(path)
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            self.handle = os.open(path, flags)
            self.identity = _descriptor_identity(self.handle)

    def close(self) -> None:
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        if self.windows_api is not None:
            self.windows_api.close(handle)
        else:
            os.close(handle)


def _move_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    if os.name != "posix":
        raise DownloadError("download_atomic_rollback_unsupported")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise DownloadError("download_atomic_rollback_unsupported")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    ) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), source)


def _rollback_created_destination(
    destination: Path,
    expected: tuple[int, int, int, int],
) -> tuple[bool, DownloadRecoveryError | None]:
    recovery = destination.parent / (
        f".{destination.name}-recovery-{secrets.token_hex(8)}"
    )
    try:
        _move_no_replace(destination, recovery)
    except FileNotFoundError:
        return False, None
    if _file_identity(recovery) == expected:
        recovery.unlink()
        return True, None
    replacement_identity = _file_identity(recovery)
    try:
        _move_no_replace(recovery, destination)
        return False, None
    except OSError as restore_failure:
        try:
            _fsync_directory(destination.parent)
        except BaseException as durability_failure:
            return False, DownloadRecoveryError(
                recovery,
                replacement_identity,
                (restore_failure, durability_failure),
            )
        return False, DownloadRecoveryError(
            recovery, replacement_identity, (restore_failure,)
        )


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
    created_identity: tuple[int, int, int, int] | None = None
    held: _HeldFile | None = None
    temporary_removed = False
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
        held = _HeldFile(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise DownloadError("download_destination") from exc
        created_identity = _file_identity(destination)
        if created_identity != held.identity:
            raise DownloadError("download_destination_identity")
        _fsync_directory(destination.parent)
        temporary.unlink()
        temporary_removed = True
        _fsync_directory(destination.parent)
        if _file_identity(destination) != created_identity:
            raise DownloadError("download_destination_identity")
        held.close()
        held = None
    except BaseException as primary:
        rollback_failure: BaseException | None = None
        if created_identity is not None:
            try:
                removed, recovery = _rollback_created_destination(
                    destination, created_identity
                )
                if removed:
                    _fsync_directory(destination.parent)
                rollback_failure = recovery
            except BaseException as exc:
                rollback_failure = exc
        if rollback_failure is not None:
            primary.__cause__ = rollback_failure
        raise primary
    finally:
        cleanup_failures: list[BaseException] = []
        if held is not None:
            try:
                held.close()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if not temporary_removed:
            cleanup_error: OSError | None = None
            for _attempt in range(2):
                try:
                    temporary.unlink(missing_ok=True)
                    temporary_removed = True
                    break
                except OSError as exc:
                    cleanup_error = exc
            if not temporary_removed and cleanup_error is not None:
                cleanup_failures.append(cleanup_error)
        if cleanup_failures:
            active = sys.exception()
            if active is None:
                if len(cleanup_failures) == 1:
                    raise cleanup_failures[0]
                raise ExceptionGroup("download_cleanup_failed", cleanup_failures)
            active.add_note(
                "download cleanup failures: "
                + "; ".join(repr(item) for item in cleanup_failures)
            )


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
