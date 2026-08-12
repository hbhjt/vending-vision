from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import math
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


class ArchiveError(RuntimeError):
    pass


_MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 100_000
_SOCKET_TIMEOUT_SECONDS = 120.0
_DEFAULT_TOTAL_TIMEOUT_SECONDS = 1800.0
_MAX_TOTAL_TIMEOUT_SECONDS = 3600.0
_PROCESS_OUTPUT_BYTES = 64 * 1024
_PROCESS_CLEANUP_SECONDS = 1.0
_EXTRACT_EXIT_CODES = {
    "archive_unsafe_path": 20,
    "archive_symlink_or_special": 21,
    "archive_member_count": 22,
    "archive_path_collision": 23,
    "archive_extracted_size": 24,
    "archive_format": 25,
    "archive_member": 26,
}
_PUBLIC_RELEASE_PATH = re.compile(
    r"^/YKDZ/vem/releases/download/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_GITHUB_ASSET_PATH = re.compile(r"^/github-production-release-asset/[0-9]+/[0-9a-f-]+$")


class _BoundGithubReleaseRedirectHandler(HTTPRedirectHandler):
    """Record the one narrowly-authorized public Release redirect."""

    def __init__(self):
        super().__init__()
        self.redirects: list[tuple[str, str]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirects.append((req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _is_public_vem_release_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "github.com"
        and not parsed.query
        and not parsed.fragment
        and _PUBLIC_RELEASE_PATH.fullmatch(parsed.path) is not None
    )


def _is_github_asset_cdn_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "release-assets.githubusercontent.com"
        and bool(parsed.query)
        and not parsed.fragment
        and _GITHUB_ASSET_PATH.fullmatch(parsed.path) is not None
    )


def _verify_redirect_identity(
    original_url: str, response_url: str, redirects: tuple[tuple[str, str], ...]
) -> None:
    if response_url == original_url:
        if redirects:
            raise ArchiveError("archive_redirect_identity")
        return
    if (
        not _is_public_vem_release_url(original_url)
        or not _is_github_asset_cdn_url(response_url)
        or redirects != ((original_url, response_url),)
    ):
        raise ArchiveError("archive_redirect_identity")


class _WindowsJobApi:
    """Stdlib-only Windows process-tree boundary for the extractor."""

    def __init__(self):
        if os.name != "nt":
            raise ArchiveError("archive_windows_job_unavailable")
        from ctypes import wintypes

        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class _ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimit),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class _Accounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        class _StartupInfo(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE),
            ]

        class _ProcessInfo(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
            ]

        self.ExtendedLimit = _ExtendedLimit
        self.Accounting = _Accounting
        self.StartupInfo = _StartupInfo
        self.ProcessInfo = _ProcessInfo
        self._declare_functions()

    def _declare_functions(self) -> None:
        wintypes = self.wintypes
        k32 = self.kernel32
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.SetHandleInformation.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD
        ]
        k32.SetHandleInformation.restype = wintypes.BOOL
        k32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
            wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.POINTER(self.StartupInfo), ctypes.POINTER(self.ProcessInfo),
        ]
        k32.CreateProcessW.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.ResumeThread.argtypes = [wintypes.HANDLE]
        k32.ResumeThread.restype = wintypes.DWORD
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
        ]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL

    def create_job(self):
        job = self.kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ArchiveError("archive_windows_job_create")
        info = self.ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = 0x00002000
        if not self.kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            self.close(job)
            raise ArchiveError("archive_windows_job_configure")
        return job

    def create_suspended(self, command: list[str]):
        startup = self.StartupInfo()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = 0x00000100
        process = self.ProcessInfo()
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
        flags = 0x00000004 | 0x00000200 | 0x08000000
        nul = self.kernel32.CreateFileW(
            "NUL", 0xC0000000, 0x00000003, None, 3, 0x00000080, None
        )
        if nul in {None, 0, ctypes.c_void_p(-1).value}:
            raise ArchiveError("archive_windows_stdio")
        if not self.kernel32.SetHandleInformation(nul, 0x00000001, 0x00000001):
            self.close(nul)
            raise ArchiveError("archive_windows_stdio_inheritance")
        startup.hStdInput = nul
        startup.hStdOutput = nul
        startup.hStdError = nul
        try:
            if not self.kernel32.CreateProcessW(
                None, command_line, None, None, True, flags, None, None,
                ctypes.byref(startup), ctypes.byref(process),
            ):
                raise ArchiveError("archive_windows_process_create")
        finally:
            self.close(nul)
        return process.hProcess, process.hThread

    def assign(self, job, process) -> None:
        if not self.kernel32.AssignProcessToJobObject(job, process):
            raise ArchiveError("archive_windows_job_assign")

    def resume(self, thread) -> None:
        if self.kernel32.ResumeThread(thread) == 0xFFFFFFFF:
            raise ArchiveError("archive_windows_process_resume")

    def wait(self, process, timeout: float) -> int:
        wait = self.kernel32.WaitForSingleObject(process, max(1, int(timeout * 1000)))
        if wait == 0x00000102:
            raise TimeoutError
        if wait != 0:
            raise ArchiveError("archive_windows_process_wait")
        code = self.wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(process, ctypes.byref(code)):
            raise ArchiveError("archive_windows_process_exit")
        return int(code.value)

    def terminate(self, job) -> None:
        if not self.kernel32.TerminateJobObject(job, 1):
            raise ArchiveError("archive_windows_job_terminate")

    def terminate_process(self, process) -> None:
        if not self.kernel32.TerminateProcess(process, 1):
            raise ArchiveError("archive_windows_process_terminate")

    def wait_active_zero(self, job, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            info = self.Accounting()
            returned = self.wintypes.DWORD()
            if not self.kernel32.QueryInformationJobObject(
                job, 1, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)
            ):
                raise ArchiveError("archive_windows_job_query")
            if int(info.ActiveProcesses) == 0:
                return
            if time.monotonic() >= deadline:
                raise ArchiveError("archive_windows_tree_alive")
            time.sleep(0.01)

    def close(self, handle) -> None:
        if handle and not self.kernel32.CloseHandle(handle):
            raise ArchiveError("archive_windows_handle_close")


class _WindowsMoveApi:
    def __init__(self, kernel32=None):
        if kernel32 is None:
            if os.name != "nt":
                raise ArchiveError("archive_publish_unsupported")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.MoveFileExW.argtypes = [
                ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32
            ]
            kernel32.MoveFileExW.restype = ctypes.c_int
        self.kernel32 = kernel32

    def move_no_replace(self, source: Path, destination: Path) -> tuple[bool, int]:
        ok = bool(self.kernel32.MoveFileExW(str(source), str(destination), 0))
        return ok, 0 if ok else ctypes.get_last_error()


def _run_windows_extractor(
    command: list[str], timeout: float, api: object | None = None
) -> None:
    api = api or _WindowsJobApi()
    job = process = thread = None
    started = False
    failure: BaseException | None = None
    try:
        job = api.create_job()
        try:
            process, thread = api.create_suspended(command)
            api.assign(job, process)
            api.resume(thread)
            started = True
            code = api.wait(process, timeout)
            api.wait_active_zero(job, _PROCESS_CLEANUP_SECONDS)
            if code != 0:
                diagnostic = next(
                    (name for name, value in _EXTRACT_EXIT_CODES.items() if value == code),
                    "archive_extract_failed",
                )
                raise ArchiveError(diagnostic)
        except BaseException as original:
            cleanup_failure = None
            for cleanup in (
                (lambda: api.terminate(job))
                if started and process is not None
                else (lambda: api.terminate_process(process))
                if process is not None
                else None,
                (lambda: api.wait(process, _PROCESS_CLEANUP_SECONDS))
                if process is not None
                else None,
                lambda: api.wait_active_zero(job, _PROCESS_CLEANUP_SECONDS),
            ):
                if cleanup is not None:
                    try:
                        cleanup()
                    except BaseException as exc:
                        cleanup_failure = cleanup_failure or exc
            if cleanup_failure is not None:
                raise ArchiveError("archive_windows_cleanup_unproven") from cleanup_failure
            raise original
    except TimeoutError as exc:
        failure = ArchiveError("archive_timeout")
        failure.__cause__ = exc
    except BaseException as exc:
        failure = exc
    finally:
        for handle in (thread, process, job):
            if handle is not None:
                try:
                    api.close(handle)
                except BaseException as exc:
                    failure = failure or exc
    if failure is not None:
        raise failure


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_group_dead(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not _group_alive(pid)


def _terminate_posix_tree(process: subprocess.Popen) -> None:
    if _group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        _wait_group_dead(process.pid, 0.1)
    if _group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError("archive_process_leader_alive") from exc
    if not _wait_group_dead(process.pid, 1.0):
        raise ArchiveError("archive_process_tree_alive")


def _run_posix_extractor(command: list[str], timeout: float) -> None:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    streams = {process.stdout, process.stderr}
    totals = {process.stdout: 0, process.stderr: 0}
    captured = {process.stdout: bytearray(), process.stderr: bytearray()}
    for stream in streams:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ)
    try:
        deadline = time.monotonic() + timeout
        failure: ArchiveError | None = None
        while streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = ArchiveError("archive_timeout")
                break
            for key, _events in selector.select(min(remaining, 0.05)):
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 8192)
                if not chunk:
                    selector.unregister(stream)
                    streams.discard(stream)
                    continue
                totals[stream] += len(chunk)
                captured[stream].extend(chunk)
                if totals[stream] > _PROCESS_OUTPUT_BYTES:
                    failure = ArchiveError("archive_process_output")
                    break
            if failure is not None:
                break
        if failure is not None:
            _terminate_posix_tree(process)
            raise failure
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            _terminate_posix_tree(process)
            raise ArchiveError("archive_timeout") from exc
        if process.returncode != 0:
            diagnostic = bytes(captured[process.stderr]).decode("ascii", "strict").strip()
            if diagnostic.startswith("archive_") and diagnostic.replace("_", "a").isalnum():
                raise ArchiveError(diagnostic)
            raise ArchiveError("archive_extract_failed")
        if _group_alive(process.pid):
            _terminate_posix_tree(process)
            raise ArchiveError("archive_extract_descendants")
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


def _run_extractor(
    command: list[str], timeout: float, *, windows_api: object | None = None
) -> None:
    if os.name == "posix":
        _run_posix_extractor(command, timeout)
        return
    if os.name == "nt" or windows_api is not None:
        _run_windows_extractor(command, timeout, windows_api)
        return
    raise ArchiveError("archive_process_supervision_unsupported")


def _cleanup_owned_directory(path: Path) -> None:
    for _attempt in range(2):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except OSError:
            pass
        if not path.exists():
            return
    raise ArchiveError("archive_cleanup_failed")


def _publish_directory_no_replace(
    source: Path,
    destination: Path,
    check_deadline=lambda: None,
    *,
    platform: str | None = None,
    windows_api: object | None = None,
) -> None:
    check_deadline()
    platform = platform or os.name
    if platform == "nt":
        ok, error = (windows_api or _WindowsMoveApi()).move_no_replace(
            source, destination
        )
        if not ok:
            if error in {80, 183}:
                raise ArchiveError("archive_destination_exists")
            raise ArchiveError(f"archive_publish_windows:{error}")
        return
    if platform != "posix":
        raise ArchiveError("archive_publish_unsupported")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArchiveError("archive_publish_unsupported")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ArchiveError("archive_destination_exists")
        raise ArchiveError(f"archive_publish_posix:{error}")


def _safe_relative(name: str) -> PurePosixPath:
    value = PurePosixPath(name.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or ":" in name:
        raise ArchiveError("archive_unsafe_path")
    return value


def _validate_members(
    members: list[tuple[PurePosixPath, bool, int]],
    max_extracted_bytes: int,
    max_members: int,
) -> None:
    if len(members) > max_members:
        raise ArchiveError("archive_member_count")
    seen: set[str] = set()
    files: set[str] = set()
    total = 0
    for relative, is_dir, size in members:
        if relative == PurePosixPath("."):
            if is_dir:
                continue
            raise ArchiveError("archive_unsafe_path")
        key = relative.as_posix().casefold()
        parent_keys = {parent.as_posix().casefold() for parent in relative.parents if parent != PurePosixPath(".")}
        if key in seen or parent_keys & files or (not is_dir and any(item.startswith(key + "/") for item in seen)):
            raise ArchiveError("archive_path_collision")
        seen.add(key)
        if not is_dir:
            files.add(key)
            total += size
            if total > max_extracted_bytes:
                raise ArchiveError("archive_extracted_size")


def _extract_archive(
    archive_path: Path,
    destination: Path,
    *,
    max_extracted_bytes: int,
    max_members: int,
) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = []
            for member in archive.infolist():
                relative = _safe_relative(member.filename)
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ArchiveError("archive_symlink_or_special")
                members.append((relative, member.is_dir(), member.file_size))
            _validate_members(members, max_extracted_bytes, max_members)
            for member, (relative, _is_dir, _size) in zip(archive.infolist(), members):
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, 1024 * 1024)
        return
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except tarfile.TarError as exc:
        raise ArchiveError("archive_format") from exc
    with archive:
        members = []
        for member in archive.getmembers():
            relative = _safe_relative(member.name)
            if not (member.isfile() or member.isdir()):
                raise ArchiveError("archive_symlink_or_special")
            members.append((relative, member.isdir(), member.size))
        _validate_members(members, max_extracted_bytes, max_members)
        for member, (relative, _is_dir, _size) in zip(archive.getmembers(), members):
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveError("archive_member")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def download_verified_archive(
    url: str,
    sha256: str,
    destination: Path,
    *,
    expected_bytes: int,
    opener=None,
    max_download_bytes: int = _MAX_DOWNLOAD_BYTES,
    max_extracted_bytes: int = _MAX_EXTRACTED_BYTES,
    max_members: int = _MAX_ARCHIVE_MEMBERS,
    total_timeout_seconds: float = _DEFAULT_TOTAL_TIMEOUT_SECONDS,
    monotonic=time.monotonic,
    extractor_command: list[str] | None = None,
) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or len(sha256) != 64:
        raise ArchiveError("archive_source")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ArchiveError("archive_sha256") from exc
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(max_download_bytes) is not int
        or max_download_bytes <= 0
        or expected_bytes > max_download_bytes
        or type(max_members) is not int
        or max_members <= 0
    ):
        raise ArchiveError("archive_download_size")
    if (
        type(total_timeout_seconds) not in {int, float}
        or not math.isfinite(total_timeout_seconds)
        or total_timeout_seconds <= 0
        or total_timeout_seconds > _MAX_TOTAL_TIMEOUT_SECONDS
    ):
        raise ArchiveError("archive_timeout")
    deadline = monotonic() + total_timeout_seconds

    def remaining() -> float:
        value = deadline - monotonic()
        if not math.isfinite(value) or value <= 0:
            raise ArchiveError("archive_timeout")
        return value

    remaining()
    if destination.exists():
        raise ArchiveError("archive_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    archive_path = work / "payload.archive"
    extracted = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-extracted-", dir=destination.parent)
    )
    work_cleaned = False
    published = False
    try:
        digest = hashlib.sha256()
        request = Request(url, headers={"User-Agent": "vem-release-archive-fetcher/1"})
        redirect_handler = None
        if opener is None:
            redirect_handler = _BoundGithubReleaseRedirectHandler()
            opener = build_opener(redirect_handler).open
        with opener(request, timeout=min(_SOCKET_TIMEOUT_SECONDS, remaining())) as response:
            remaining()
            response_url = response.geturl()
            remaining()
            redirects = tuple(
                redirect_handler.redirects
                if redirect_handler is not None
                else getattr(opener, "redirects", ())
            )
            _verify_redirect_identity(url, response_url, redirects)
            with archive_path.open("xb") as output:
                downloaded = 0
                while True:
                    remaining()
                    chunk = response.read(
                        min(1024 * 1024, expected_bytes - downloaded + 1)
                    )
                    remaining()
                    if not chunk:
                        break
                    if downloaded + len(chunk) > expected_bytes or downloaded + len(chunk) > max_download_bytes:
                        raise ArchiveError("archive_download_size")
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
            raise ArchiveError("archive_download_size")
        remaining()
        actual_sha256 = digest.hexdigest()
        remaining()
        if actual_sha256 != sha256.lower():
            raise ArchiveError("archive_digest")
        remaining()
        command = extractor_command or [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("archive_extractor_worker.py")),
            "--archive",
            str(archive_path),
            "--destination",
            str(extracted),
            "--max-extracted-bytes",
            str(max_extracted_bytes),
            "--max-members",
            str(max_members),
        ]
        _run_extractor(command, remaining())
        remaining()
        _cleanup_owned_directory(work)
        work_cleaned = True
        _publish_directory_no_replace(extracted, destination, remaining)
        published = True
    finally:
        cleanup_failure = None
        for owned, needed in ((work, not work_cleaned), (extracted, not published)):
            if needed:
                try:
                    _cleanup_owned_directory(owned)
                except ArchiveError as exc:
                    cleanup_failure = cleanup_failure or exc
        if cleanup_failure is not None:
            raise cleanup_failure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--total-timeout-seconds", required=True, type=float)
    args = parser.parse_args()
    download_verified_archive(
        args.url,
        args.sha256,
        Path(args.destination).resolve(),
        expected_bytes=args.expected_bytes,
        total_timeout_seconds=args.total_timeout_seconds,
    )
    print("Verified archive extracted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
