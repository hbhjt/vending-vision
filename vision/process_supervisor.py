"""Bounded whole-tree process supervision for attempt-scoped AI children."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

_TAIL_LIMIT = 64 * 1024


class ProcessSupervisorError(RuntimeError):
    pass


@dataclass
class StreamTail:
    chunks: deque[bytes] = field(default_factory=deque)
    size: int = 0
    total: int = 0

    def append(self, chunk: bytes) -> None:
        self.total += len(chunk)
        self.chunks.append(chunk)
        self.size += len(chunk)
        while self.size > _TAIL_LIMIT and self.chunks:
            first = self.chunks.popleft()
            overflow = self.size - _TAIL_LIMIT
            if overflow < len(first):
                self.chunks.appendleft(first[overflow:])
                self.size -= overflow
                break
            self.size -= len(first)

    def bytes(self) -> bytes:
        return b"".join(self.chunks)


@dataclass
class SupervisedResult:
    returncode: int
    stdout_tail: bytes
    stderr_tail: bytes
    stdout_total: int
    stderr_total: int


class LinuxProcessTree:
    def __init__(self, command: list[str]):
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self.stdout = StreamTail()
        self.stderr = StreamTail()
        self._drainers: list[asyncio.Task] = []
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.process is not None:
            raise ProcessSupervisorError("supervisor_already_started")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._drainers = [
            asyncio.create_task(self._drain(self.process.stdout, self.stdout)),
            asyncio.create_task(self._drain(self.process.stderr, self.stderr)),
        ]

    async def wait(self, timeout: float) -> SupervisedResult:
        if self.process is None:
            raise ProcessSupervisorError("supervisor_not_started")
        try:
            code = await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise ProcessSupervisorError("supervised_process_timeout") from exc
        if self._pgid_alive():
            await self.close()
            raise ProcessSupervisorError("supervised_process_descendants_alive")
        await self._finish_drainers()
        return SupervisedResult(
            returncode=int(code),
            stdout_tail=self.stdout.bytes(),
            stderr_tail=self.stderr.bytes(),
            stdout_total=self.stdout.total,
            stderr_total=self.stderr.total,
        )

    async def close(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        await asyncio.shield(self._cleanup_task)

    async def _cleanup(self) -> None:
        process = self.process
        if process is None:
            return
        pgid = process.pid
        if self._pgid_alive():
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            await self._wait_group_dead(grace=0.5)
        if self._pgid_alive():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._wait_group_dead(grace=2.0)
        if process.poll() is None:
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 0.5)
            except asyncio.TimeoutError:
                pass
        await self._finish_drainers()

    async def _drain(self, stream, tail: StreamTail) -> None:
        if stream is None:
            return
        while True:
            chunk = await asyncio.to_thread(stream.read, 8192)
            if not chunk:
                return
            tail.append(chunk)

    async def _finish_drainers(self) -> None:
        if self._drainers:
            await asyncio.gather(*self._drainers, return_exceptions=True)

    def _pgid_alive(self) -> bool:
        if self.process is None:
            return False
        try:
            os.killpg(self.process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    async def _wait_group_dead(self, *, grace: float) -> None:
        deadline = asyncio.get_running_loop().time() + grace
        while self._pgid_alive() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.025)


class WindowsJobApiUnavailable(ProcessSupervisorError):
    pass


class WindowsJobProcess:
    """Windows Job Object launcher.

    The default API is ctypes-backed on Windows.  Tests inject a fake API with
    the same method names to verify call order and fail-closed behavior on
    non-Windows CI.
    """

    def __init__(self, command: list[str], *, api=None):
        self.command = command
        self.api = api or WindowsJobApi()
        self.job = None
        self.process_handle = None
        self.thread_handle = None
        self.pid: int | None = None
        self.resumed = False

    def start(self) -> None:
        self.job = self.api.create_job()
        try:
            self.api.set_kill_on_close_and_low_priority(self.job)
            created = self.api.create_process_suspended(self.command)
            self.process_handle = created["process"]
            self.thread_handle = created["thread"]
            self.pid = created["pid"]
            try:
                self.api.assign_process_to_job(self.job, self.process_handle)
            except WindowsJobApiUnavailable:
                self.api.terminate_process(self.process_handle, 1)
                raise
            self.api.resume_thread(self.thread_handle)
            self.resumed = True
        except Exception:
            self.close()
            raise

    def terminate_tree(self) -> None:
        if self.job is not None:
            self.api.terminate_job(self.job, 1)
            self.api.wait_active_processes_zero(self.job, timeout=3.0)

    def wait(self, timeout: float) -> SupervisedResult:
        if self.process_handle is None:
            raise ProcessSupervisorError("windows_process_not_started")
        try:
            code = self.api.wait_process(self.process_handle, timeout=timeout)
        except ProcessSupervisorError:
            self.terminate_tree()
            raise
        try:
            self.api.wait_active_processes_zero(self.job, timeout=3.0)
        except Exception as exc:
            self.terminate_tree()
            raise ProcessSupervisorError("windows_job_descendants_alive") from exc
        return SupervisedResult(code, b"", b"", 0, 0)

    def close(self) -> None:
        try:
            if self.job is not None:
                self.api.close_handle(self.job)
        finally:
            if self.thread_handle is not None:
                self.api.close_handle(self.thread_handle)
            if self.process_handle is not None:
                self.api.close_handle(self.process_handle)
            self.job = None
            self.thread_handle = None
            self.process_handle = None


class WindowsJobApi:
    ERROR_ACCESS_DENIED = 5

    def __init__(self):
        if os.name != "nt":
            raise WindowsJobApiUnavailable("windows_job_api_unavailable")
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._define_structures()

    def _define_structures(self) -> None:
        ctypes = self.ctypes
        wintypes = self.wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class STARTUPINFOW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD),
                ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD),
                ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE),
            ]

        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE),
                ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD),
            ]

        self.JOBOBJECT_EXTENDED_LIMIT_INFORMATION = JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        self.STARTUPINFOW = STARTUPINFOW
        self.PROCESS_INFORMATION = PROCESS_INFORMATION

    def create_job(self):
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise WindowsJobApiUnavailable("windows_job_create_failed")
        return handle

    def set_kill_on_close_and_low_priority(self, job) -> None:
        ctypes = self.ctypes
        info = self.JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x00002000 | 0x00000020
        info.BasicLimitInformation.PriorityClass = 0x00004000
        ok = self.kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise WindowsJobApiUnavailable("windows_job_set_failed")

    def create_process_suspended(self, command: list[str]) -> dict[str, object]:
        import subprocess as _subprocess

        ctypes = self.ctypes
        si = self.STARTUPINFOW()
        pi = self.PROCESS_INFORMATION()
        si.cb = ctypes.sizeof(si)
        cmdline = _subprocess.list2cmdline(command)
        flags = 0x00000004 | 0x08000000 | 0x00004000
        ok = self.kernel32.CreateProcessW(
            None,
            ctypes.create_unicode_buffer(cmdline),
            None,
            None,
            True,
            flags,
            None,
            None,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        if not ok:
            raise WindowsJobApiUnavailable("windows_create_process_failed")
        return {"process": pi.hProcess, "thread": pi.hThread, "pid": int(pi.dwProcessId)}

    def assign_process_to_job(self, job, process) -> None:
        ok = self.kernel32.AssignProcessToJobObject(job, process)
        if not ok:
            error = self.ctypes.get_last_error()
            if error == self.ERROR_ACCESS_DENIED:
                raise WindowsJobApiUnavailable("windows_nested_job_unavailable")
            raise WindowsJobApiUnavailable("windows_job_assign_failed")

    def resume_thread(self, thread) -> None:
        if self.kernel32.ResumeThread(thread) == 0xFFFFFFFF:
            raise WindowsJobApiUnavailable("windows_resume_failed")

    def terminate_process(self, process, code: int) -> None:
        self.kernel32.TerminateProcess(process, code)

    def terminate_job(self, job, code: int) -> None:
        self.kernel32.TerminateJobObject(job, code)

    def wait_active_processes_zero(self, job, *, timeout: float) -> None:
        return None

    def wait_process(self, process, *, timeout: float) -> int:
        ctypes = self.ctypes
        wait = self.kernel32.WaitForSingleObject(process, int(timeout * 1000))
        if wait == 0x00000102:
            raise ProcessSupervisorError("supervised_process_timeout")
        if wait != 0:
            raise ProcessSupervisorError("windows_process_wait_failed")
        exit_code = self.wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)):
            raise ProcessSupervisorError("windows_get_exit_code_failed")
        return int(exit_code.value)

    def close_handle(self, handle) -> None:
        self.kernel32.CloseHandle(handle)


def supervisor_for_command(command: list[str]):
    if os.name == "nt":
        return WindowsJobProcess(command)
    return LinuxProcessTree(command)


async def run_supervised(command: list[str], *, timeout: float) -> SupervisedResult:
    supervisor = supervisor_for_command(command)
    if isinstance(supervisor, WindowsJobProcess):
        await asyncio.to_thread(supervisor.start)
        try:
            return await asyncio.to_thread(supervisor.wait, timeout)
        finally:
            await asyncio.to_thread(supervisor.close)
    await supervisor.start()
    try:
        return await supervisor.wait(timeout)
    finally:
        await supervisor.close()


def taskkill_fallback_command(pid: int) -> list[str]:
    return ["taskkill", "/PID", str(pid), "/T", "/F"]


async def run_taskkill_fallback(pid: int) -> None:
    await asyncio.to_thread(
        subprocess.run,
        taskkill_fallback_command(pid),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
        check=False,
    )
