"""Private installed-acceptance sink for completed AI regional evidence."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable
from pathlib import Path

_ATTEMPT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_SIDECAR_BYTES = 512 * 1024
_WINDOWS = os.name == "nt"


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("dwFileAttributes", ctypes.c_uint32),
        ("ftCreationTime", _FILETIME),
        ("ftLastAccessTime", _FILETIME),
        ("ftLastWriteTime", _FILETIME),
        ("dwVolumeSerialNumber", ctypes.c_uint32),
        ("nFileSizeHigh", ctypes.c_uint32),
        ("nFileSizeLow", ctypes.c_uint32),
        ("nNumberOfLinks", ctypes.c_uint32),
        ("nFileIndexHigh", ctypes.c_uint32),
        ("nFileIndexLow", ctypes.c_uint32),
    ]


class _WindowsEvidenceFileApi:
    """Held-handle file operations for the Windows acceptance-only sink."""

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    DELETE = 0x00010000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    CREATE_NEW = 1
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

    def __init__(self, kernel32):
        self._kernel32 = kernel32

    @staticmethod
    def _invalid(handle: object) -> bool:
        invalid = ctypes.c_void_p(-1).value
        return handle in {None, 0, -1, invalid}

    def _open(
        self,
        path: Path,
        access: int,
        creation: int,
        flags: int = FILE_ATTRIBUTE_NORMAL,
        share: int = FILE_SHARE_READ,
    ) -> int:
        handle = self._kernel32.CreateFileW(
            str(path),
            access,
            share,
            None,
            creation,
            flags,
            None,
        )
        if self._invalid(handle):
            raise RuntimeError("ai_acceptance_evidence_windows_handle_open")
        return int(handle)

    def create_temporary(self, path: Path) -> int:
        return self._open(
            path,
            self.GENERIC_READ | self.GENERIC_WRITE,
            self.CREATE_NEW,
        )

    def open_read(self, path: Path) -> int:
        # This view is opened while the writable source handle is still held.
        # Windows requires a new handle's share mask to admit the access of all
        # existing handles to the same file identity.  The source itself still
        # denies external write/delete opens until both final-fence views close.
        return self._open(
            path,
            self.GENERIC_READ,
            self.OPEN_EXISTING,
            share=self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
        )

    def open_delete(self, path: Path) -> int:
        return self._open(path, self.GENERIC_READ | self.DELETE, self.OPEN_EXISTING)

    def open_directory(self, path: Path) -> int:
        return self._open(
            path,
            self.GENERIC_READ,
            self.OPEN_EXISTING,
            self.FILE_FLAG_BACKUP_SEMANTICS,
            self.FILE_SHARE_READ | self.FILE_SHARE_WRITE,
        )

    def write_all(self, handle: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 64 * 1024]
            buffer = ctypes.create_string_buffer(chunk)
            written = ctypes.c_uint32()
            if not self._kernel32.WriteFile(
                handle, buffer, len(chunk), ctypes.byref(written), None
            ) or written.value != len(chunk):
                raise RuntimeError("ai_acceptance_evidence_windows_handle_write")
            offset += written.value

    def flush(self, handle: int) -> None:
        if not self._kernel32.FlushFileBuffers(handle):
            raise RuntimeError("ai_acceptance_evidence_windows_handle_flush")

    def close(self, handle: int) -> None:
        first_failed = False
        for _attempt in range(2):
            if self._kernel32.CloseHandle(handle):
                if first_failed:
                    raise RuntimeError("ai_acceptance_evidence_windows_handle_close")
                return
            first_failed = True
        raise RuntimeError("ai_acceptance_evidence_windows_handle_close")

    def hard_link(self, destination: Path, source: Path) -> None:
        if not self._kernel32.CreateHardLinkW(str(destination), str(source), None):
            raise RuntimeError("ai_acceptance_evidence_exists")

    def information(self, handle: int) -> tuple[tuple[int, int, int], int]:
        information = _BY_HANDLE_FILE_INFORMATION()
        if not self._kernel32.GetFileInformationByHandle(
            handle, ctypes.byref(information)
        ):
            raise RuntimeError("ai_acceptance_evidence_windows_handle_information")
        return (
            (
                information.dwVolumeSerialNumber,
                information.nFileIndexHigh,
                information.nFileIndexLow,
            ),
            (information.nFileSizeHigh << 32) | information.nFileSizeLow,
        )

    def sha256(self, handle: int) -> bytes:
        digest = hashlib.sha256()
        while True:
            buffer = ctypes.create_string_buffer(64 * 1024)
            read = ctypes.c_uint32()
            if not self._kernel32.ReadFile(
                handle, buffer, len(buffer), ctypes.byref(read), None
            ):
                raise RuntimeError("ai_acceptance_evidence_windows_handle_read")
            if read.value == 0:
                return digest.digest()
            digest.update(buffer.raw[: read.value])

    def delete_path(self, path: Path) -> None:
        if not self._kernel32.DeleteFileW(str(path)):
            raise RuntimeError("ai_acceptance_evidence_windows_path_delete")


def _windows_kernel32_factory():
    if not _WINDOWS or not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("ai_acceptance_evidence_windows_held_handle_unavailable")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.CreateFileW.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.WriteFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel32.WriteFile.restype = ctypes.c_int
    kernel32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
    kernel32.FlushFileBuffers.restype = ctypes.c_int
    kernel32.CreateHardLinkW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p
    ]
    kernel32.CreateHardLinkW.restype = ctypes.c_int
    kernel32.DeleteFileW.argtypes = [ctypes.c_wchar_p]
    kernel32.DeleteFileW.restype = ctypes.c_int
    kernel32.GetFileInformationByHandle.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)
    ]
    kernel32.GetFileInformationByHandle.restype = ctypes.c_int
    kernel32.ReadFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = ctypes.c_int
    return kernel32


def _windows_file_api_factory() -> _WindowsEvidenceFileApi:
    return _WindowsEvidenceFileApi(_windows_kernel32_factory())


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _acceptance_root() -> Path | None:
    raw = os.environ.get("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", "")
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute():
        raise RuntimeError("ai_acceptance_evidence_root_invalid")
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("ai_acceptance_evidence_root_invalid")
    return root.resolve(strict=True)


def _fsync_directory(root: Path) -> None:
    """Persist a directory entry where the platform exposes directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError:
        # Windows does not expose POSIX directory descriptors.  The file data
        # is still flushed before the non-overwriting hard-link publish.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_empty_invocation_root(root: Path, destination: Path) -> Path:
    """Reserve one empty, test-owned root without deleting another publisher's data."""
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("ai_acceptance_evidence_exists")
    if any(root.iterdir()):
        raise RuntimeError("ai_acceptance_evidence_root_not_empty")
    claim = root / ".ai-regional-evidence-publishing"
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("ai_acceptance_evidence_root_not_empty") from exc
    return claim


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return _identity_from_stat(info)


def _identity_from_stat(info: os.stat_result) -> tuple[int, int] | None:
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino


def _unlink_created_file(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is not None and _identity(path) == identity:
        path.unlink()


def _open_held_temporary(temporary: Path) -> int:
    if _WINDOWS:
        # The CRT open used here cannot promise a read-only sharing mode for a
        # handle held across the link and final fence.  Do not claim a Windows
        # success until that boundary has an explicit CreateFile implementation.
        raise RuntimeError("ai_acceptance_evidence_windows_held_handle_unavailable")
    return os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)


def _sha256_from_descriptor(descriptor: int) -> bytes:
    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
        return digest.digest()
    finally:
        os.lseek(descriptor, offset, os.SEEK_SET)


def _verify_final_destination(
    destination: Path,
    source_descriptor: int,
    expected_size: int,
    expected_digest: bytes,
) -> None:
    """Fence the published name against the source inode and canonical bytes."""
    source_identity = _identity_from_stat(os.fstat(source_descriptor))
    if source_identity is None or _identity(destination) != source_identity:
        raise RuntimeError("ai_acceptance_evidence_final_fence_failed")
    try:
        destination_descriptor = os.open(
            destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise RuntimeError("ai_acceptance_evidence_final_fence_failed") from exc
    try:
        info = os.fstat(destination_descriptor)
        if (
            _identity_from_stat(info) != source_identity
            or info.st_size != expected_size
            or _sha256_from_descriptor(destination_descriptor) != expected_digest
        ):
            raise RuntimeError("ai_acceptance_evidence_final_fence_failed")
    finally:
        os.close(destination_descriptor)


def _raise_with_cleanup(
    primary: BaseException | None, cleanup_errors: list[BaseException]
) -> None:
    if primary is None and not cleanup_errors:
        return
    if primary is not None and not cleanup_errors:
        raise primary
    errors = ([primary] if primary is not None else []) + cleanup_errors
    raise ExceptionGroup("AI acceptance evidence publication failed", errors)


def _publish_posix_ai_regional_evidence(
    root: Path, attempt_id: str, content: bytes
) -> Path:
    destination = root / f"{attempt_id}.regional-evidence.json"
    if destination.parent != root:
        raise RuntimeError("ai_acceptance_evidence_path_invalid")
    claim = _claim_empty_invocation_root(root, destination)
    temporary = root / f".{attempt_id}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = None
    temporary_identity = None
    published_identity = None
    primary = None
    cleanup_errors: list[BaseException] = []
    try:
        fd = _open_held_temporary(temporary)
        with os.fdopen(os.dup(fd), "wb", closefd=True) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_identity = _identity_from_stat(os.fstat(fd))
        if temporary_identity is None:
            raise RuntimeError("ai_acceptance_evidence_temporary_invalid")
        os.link(temporary, destination)
        published_identity = _identity(destination)
        if published_identity != temporary_identity:
            raise RuntimeError("ai_acceptance_evidence_publish_invalid")
        _fsync_directory(root)
        temporary.unlink()
        # Keep the claim and source descriptor through this publication fence.
        _fsync_directory(root)
        _verify_final_destination(
            destination,
            fd,
            len(content),
            hashlib.sha256(content).digest(),
        )
        claim.rmdir()
    except FileExistsError as exc:
        primary = RuntimeError("ai_acceptance_evidence_exists")
        primary.__cause__ = exc
    except BaseException as exc:
        primary = exc
    finally:
        def cleanup(action: Callable[[], None]) -> None:
            try:
                action()
            except FileNotFoundError:
                # A prior cleanup may already have removed our own entry.
                return
            except BaseException as exc:
                cleanup_errors.append(exc)

        if fd is not None:
            cleanup(lambda: os.close(fd))
        if primary is None and cleanup_errors:
            primary = cleanup_errors.pop(0)
        if primary is not None:
            cleanup(lambda: _unlink_created_file(destination, published_identity))
            cleanup(lambda: _unlink_created_file(temporary, temporary_identity))
            cleanup(claim.rmdir)
            # Retry one transient cleanup failure without hiding its evidence.
            cleanup(lambda: _unlink_created_file(temporary, temporary_identity))
            cleanup(claim.rmdir)
            cleanup(lambda: _fsync_directory(root))
        _raise_with_cleanup(primary, cleanup_errors)
    return destination


def _publish_windows_ai_regional_evidence(
    root: Path, attempt_id: str, content: bytes
) -> Path:
    """Publish under held Windows handles; no POSIX descriptor semantics leak here."""
    destination = root / f"{attempt_id}.regional-evidence.json"
    if destination.parent != root:
        raise RuntimeError("ai_acceptance_evidence_path_invalid")
    api = _windows_file_api_factory()
    claim = None
    temporary = root / f".{attempt_id}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    root_handle = None
    root_identity = None
    source_handle = None
    destination_handle = None
    claim_handle = None
    temporary_identity = None
    published_identity = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []

    def close_live(name: str) -> None:
        nonlocal root_handle, source_handle, destination_handle, claim_handle
        handle = {
            "root": root_handle,
            "source": source_handle,
            "destination": destination_handle,
            "claim": claim_handle,
        }[name]
        if handle is None:
            return
        if name == "root":
            root_handle = None
        elif name == "source":
            source_handle = None
        elif name == "destination":
            destination_handle = None
        else:
            claim_handle = None
        api.close(handle)

    def cleanup(action: Callable[[], None]) -> None:
        try:
            action()
        except FileNotFoundError:
            return
        except BaseException as exc:
            cleanup_errors.append(exc)

    def delete_owned(path: Path, identity: tuple[int, int, int] | None) -> None:
        if identity is None:
            return
        try:
            handle = api.open_delete(path)
        except RuntimeError:
            if not path.exists() and not path.is_symlink():
                return
            raise
        observed, _size = api.information(handle)
        api.close(handle)
        if observed != identity:
            return
        api.delete_path(path)

    def verify_root_path() -> None:
        if root_identity is None:
            raise RuntimeError("ai_acceptance_evidence_root_changed")
        current_handle = api.open_directory(root)
        try:
            current_identity, _size = api.information(current_handle)
        finally:
            api.close(current_handle)
        if current_identity != root_identity:
            raise RuntimeError("ai_acceptance_evidence_root_changed")

    try:
        root_handle = api.open_directory(root)
        root_identity, _root_size = api.information(root_handle)
        claim = _claim_empty_invocation_root(root, destination)
        verify_root_path()
        claim_handle = api.open_directory(claim)
        source_handle = api.create_temporary(temporary)
        temporary_identity, _temporary_initial_size = api.information(source_handle)
        api.write_all(source_handle, content)
        observed_identity, temporary_size = api.information(source_handle)
        if observed_identity != temporary_identity:
            raise RuntimeError("ai_acceptance_evidence_temporary_invalid")
        if temporary_size != len(content):
            raise RuntimeError("ai_acceptance_evidence_temporary_invalid")
        api.flush(source_handle)
        verify_root_path()
        api.hard_link(destination, temporary)
        destination_handle = api.open_read(destination)
        published_identity, published_size = api.information(destination_handle)
        if published_identity != temporary_identity or published_size != len(content):
            raise RuntimeError("ai_acceptance_evidence_publish_invalid")
        # FlushFileBuffers requires GENERIC_WRITE.  The held source has it;
        # the read-only destination must never be passed to that API.  Windows
        # exposes no directory durability primitive usable with this least-
        # privilege handle, so correctness is fenced by the flushed file and
        # held source/destination identities rather than a directory flush.
        api.flush(source_handle)
        verify_root_path()
        final_identity, final_size = api.information(destination_handle)
        if (
            final_identity != temporary_identity
            or final_size != len(content)
            or api.sha256(destination_handle) != hashlib.sha256(content).digest()
        ):
            raise RuntimeError("ai_acceptance_evidence_final_fence_failed")
        close_live("destination")
        close_live("source")
        delete_owned(temporary, temporary_identity)
        close_live("claim")
        claim.rmdir()
        verify_root_path()
        # Success is not observable until every held handle, including the
        # root rename/delete fence, reports a clean close.
        close_live("root")
    except BaseException as exc:
        primary = exc
    finally:
        cleanup(lambda: close_live("destination"))
        cleanup(lambda: close_live("source"))
        cleanup(lambda: close_live("claim"))
        if primary is None and cleanup_errors:
            primary = cleanup_errors.pop(0)
        if primary is not None:
            cleanup(lambda: delete_owned(destination, published_identity))
            cleanup(lambda: delete_owned(temporary, temporary_identity))
            if claim is not None:
                cleanup(claim.rmdir)
            cleanup(lambda: delete_owned(destination, published_identity))
            cleanup(lambda: delete_owned(temporary, temporary_identity))
            if claim is not None:
                cleanup(claim.rmdir)
        cleanup(lambda: close_live("root"))
        _raise_with_cleanup(primary, cleanup_errors)
    return destination


def publish_completed_ai_regional_evidence(attempt_id: str, content: bytes) -> Path | None:
    """Publish one already-committed sidecar; disabled unless an owner sets a root."""
    root = _acceptance_root()
    if root is None:
        return None
    if not _ATTEMPT_ID.fullmatch(attempt_id) or not 0 < len(content) <= _MAX_SIDECAR_BYTES:
        raise RuntimeError("ai_acceptance_evidence_invalid")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("ai_acceptance_evidence_invalid") from exc
    if _canonical(value) != content:
        raise RuntimeError("ai_acceptance_evidence_noncanonical")
    if _WINDOWS:
        return _publish_windows_ai_regional_evidence(root, attempt_id, content)
    return _publish_posix_ai_regional_evidence(root, attempt_id, content)
