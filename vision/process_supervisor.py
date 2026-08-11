"""Bounded whole-tree process supervision for attempt-scoped AI children."""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
import time
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

    _cleanup_blocked = False

    def __init__(self, command: list[str], *, api=None):
        self.command = command
        self.api = api or WindowsJobApi()
        self.job = None
        self.process_handle = None
        self.thread_handle = None
        self.pid: int | None = None
        self.resumed = False
        self.stdout = StreamTail()
        self.stderr = StreamTail()
        self.pipes = None
        self._drainers = []
        self._dead_proven = False

    @classmethod
    def reset_cleanup_block_for_tests(cls) -> None:
        cls._cleanup_blocked = False

    def start(self) -> None:
        if self.__class__._cleanup_blocked:
            raise ProcessSupervisorError("windows_previous_cleanup_unproven")
        self.job = self.api.create_job()
        try:
            self.pipes = self.api.create_pipes()
            self.api.set_kill_on_close_and_low_priority(self.job)
            created = self.api.create_process_suspended(self.command, self.pipes)
            self.process_handle = created["process"]
            self.thread_handle = created["thread"]
            self.pid = created["pid"]
            try:
                self.api.assign_process_to_job(self.job, self.process_handle)
            except WindowsJobApiUnavailable:
                try:
                    self.api.terminate_process(self.process_handle, 1)
                except ProcessSupervisorError:
                    pass
                raise
            self._drainers = self.api.start_pipe_drainers(self.pipes, self.stdout, self.stderr)
            try:
                self.api.resume_thread(self.thread_handle)
            except WindowsJobApiUnavailable:
                try:
                    self.api.terminate_job(self.job, 1)
                except ProcessSupervisorError:
                    pass
                raise
            self.resumed = True
        except Exception:
            self.close()
            raise

    def terminate_tree(self) -> None:
        if self.job is not None:
            try:
                self.api.terminate_job(self.job, 1)
                self.api.wait_active_processes_zero(self.job, timeout=3.0)
                if self.process_handle is not None:
                    self.api.wait_process(self.process_handle, timeout=3.0)
                self._dead_proven = True
            except ProcessSupervisorError:
                self._fallback_and_prove_dead()
                raise

    def _fallback_and_prove_dead(self) -> bool:
        if self.pid is not None:
            try:
                self.api.taskkill_fallback(self.pid)
            except Exception:
                pass
        leader_dead = self.process_handle is None
        active_zero = self.job is None
        if self.process_handle is not None:
            try:
                self.api.wait_process(self.process_handle, timeout=3.0)
                leader_dead = True
            except ProcessSupervisorError:
                leader_dead = False
        if self.job is not None:
            try:
                self.api.wait_active_processes_zero(self.job, timeout=3.0)
                active_zero = True
            except ProcessSupervisorError:
                active_zero = False
        self._dead_proven = leader_dead and active_zero
        return self._dead_proven

    def wait(self, timeout: float) -> SupervisedResult:
        if self.process_handle is None:
            raise ProcessSupervisorError("windows_process_not_started")
        try:
            code = self.api.wait_process(self.process_handle, timeout=timeout)
        except ProcessSupervisorError:
            try:
                self.api.terminate_job(self.job, 1)
            except ProcessSupervisorError:
                pass
            self._fallback_and_prove_dead()
            raise
        self.api.finish_pipe_drainers(self._drainers, timeout=3.0)
        try:
            self.api.wait_active_processes_zero(self.job, timeout=3.0)
        except Exception as exc:
            try:
                self.api.terminate_job(self.job, 1)
            except ProcessSupervisorError:
                pass
            self._fallback_and_prove_dead()
            raise ProcessSupervisorError("windows_job_descendants_alive") from exc
        self._dead_proven = True
        return SupervisedResult(
            code,
            self.stdout.bytes(),
            self.stderr.bytes(),
            self.stdout.total,
            self.stderr.total,
        )

    def close(self) -> bool:
        if self.process_handle is not None and not self._dead_proven:
            self._fallback_and_prove_dead()
        if self.process_handle is not None and not self._dead_proven:
            self.__class__._cleanup_blocked = True
            return False
        try:
            if self.job is not None:
                self.api.close_handle(self.job)
        finally:
            if self.thread_handle is not None:
                self.api.close_handle(self.thread_handle)
            if self.process_handle is not None:
                self.api.close_handle(self.process_handle)
            if self.pipes is not None:
                self.api.close_pipes(self.pipes)
            self.job = None
            self.thread_handle = None
            self.process_handle = None
            self.pipes = None
        return True


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
        self._declare_functions()

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

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", wintypes.BOOL),
            ]

        class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
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

        self.JOBOBJECT_EXTENDED_LIMIT_INFORMATION = JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        self.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        self.STARTUPINFOW = STARTUPINFOW
        self.PROCESS_INFORMATION = PROCESS_INFORMATION
        self.SECURITY_ATTRIBUTES = SECURITY_ATTRIBUTES

    def _declare_functions(self) -> None:
        ctypes = self.ctypes
        wintypes = self.wintypes
        k32 = self.kernel32
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k32.QueryInformationJobObject.restype = wintypes.BOOL
        k32.CreatePipe.argtypes = [ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(self.SECURITY_ATTRIBUTES), wintypes.DWORD]
        k32.CreatePipe.restype = wintypes.BOOL
        k32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        k32.SetHandleInformation.restype = wintypes.BOOL
        k32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(self.STARTUPINFOW),
            ctypes.POINTER(self.PROCESS_INFORMATION),
        ]
        k32.CreateProcessW.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.ResumeThread.argtypes = [wintypes.HANDLE]
        k32.ResumeThread.restype = wintypes.DWORD
        k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateProcess.restype = wintypes.BOOL
        k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k32.TerminateJobObject.restype = wintypes.BOOL
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.ReadFile.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL

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

    def create_pipes(self) -> dict[str, object]:
        stdout_read = self.wintypes.HANDLE()
        stdout_write = self.wintypes.HANDLE()
        stderr_read = self.wintypes.HANDLE()
        stderr_write = self.wintypes.HANDLE()
        attributes = self.SECURITY_ATTRIBUTES()
        attributes.nLength = self.ctypes.sizeof(attributes)
        attributes.lpSecurityDescriptor = None
        attributes.bInheritHandle = True
        if not self.kernel32.CreatePipe(self.ctypes.byref(stdout_read), self.ctypes.byref(stdout_write), self.ctypes.byref(attributes), 0):
            raise WindowsJobApiUnavailable("windows_pipe_create_failed")
        if not self.kernel32.CreatePipe(self.ctypes.byref(stderr_read), self.ctypes.byref(stderr_write), self.ctypes.byref(attributes), 0):
            self.close_handle(stdout_read)
            self.close_handle(stdout_write)
            raise WindowsJobApiUnavailable("windows_pipe_create_failed")
        HANDLE_FLAG_INHERIT = 0x00000001
        for read_handle in (stdout_read, stderr_read):
            if not self.kernel32.SetHandleInformation(read_handle, HANDLE_FLAG_INHERIT, 0):
                self.close_handle(stdout_read)
                self.close_handle(stdout_write)
                self.close_handle(stderr_read)
                self.close_handle(stderr_write)
                raise WindowsJobApiUnavailable("windows_pipe_inheritance_failed")
        return {
            "stdout_read": stdout_read,
            "stdout_write": stdout_write,
            "stderr_read": stderr_read,
            "stderr_write": stderr_write,
        }

    def create_process_suspended(self, command: list[str], pipes: dict[str, object]) -> dict[str, object]:
        import subprocess as _subprocess

        ctypes = self.ctypes
        si = self.STARTUPINFOW()
        pi = self.PROCESS_INFORMATION()
        si.cb = ctypes.sizeof(si)
        si.dwFlags = 0x00000100
        si.hStdOutput = pipes["stdout_write"]
        si.hStdError = pipes["stderr_write"]
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
        self.close_handle(pipes["stdout_write"])
        self.close_handle(pipes["stderr_write"])
        pipes["stdout_write"] = None
        pipes["stderr_write"] = None
        return {"process": pi.hProcess, "thread": pi.hThread, "pid": int(pi.dwProcessId)}

    def start_pipe_drainers(self, pipes: dict[str, object], stdout: StreamTail, stderr: StreamTail) -> list[threading.Thread]:
        threads = [
            threading.Thread(target=self._drain_pipe, args=(pipes["stdout_read"], stdout), daemon=True),
            threading.Thread(target=self._drain_pipe, args=(pipes["stderr_read"], stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return threads

    def _drain_pipe(self, handle, tail: StreamTail) -> None:
        ctypes = self.ctypes
        buffer = ctypes.create_string_buffer(8192)
        read = self.wintypes.DWORD()
        while True:
            ok = self.kernel32.ReadFile(handle, buffer, 8192, ctypes.byref(read), None)
            if not ok or read.value == 0:
                return
            tail.append(buffer.raw[: read.value])

    def finish_pipe_drainers(self, drainers: list[threading.Thread], *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for thread in drainers:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)

    def close_pipes(self, pipes: dict[str, object]) -> None:
        for handle in pipes.values():
            if handle:
                self.close_handle(handle)

    def taskkill_fallback(self, pid: int) -> None:
        subprocess.run(
            taskkill_fallback_command(pid),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )

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
        if not self.kernel32.TerminateProcess(process, code):
            raise ProcessSupervisorError("windows_process_terminate_failed")

    def terminate_job(self, job, code: int) -> None:
        if not self.kernel32.TerminateJobObject(job, code):
            raise ProcessSupervisorError("windows_job_terminate_failed")

    def wait_active_processes_zero(self, job, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            info = self.JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
            returned = self.wintypes.DWORD()
            ok = self.kernel32.QueryInformationJobObject(
                job,
                1,
                self.ctypes.byref(info),
                self.ctypes.sizeof(info),
                self.ctypes.byref(returned),
            )
            if not ok:
                raise ProcessSupervisorError("windows_job_query_failed")
            if int(info.ActiveProcesses) == 0:
                return
            if time.monotonic() >= deadline:
                raise ProcessSupervisorError("windows_job_descendants_alive")
            time.sleep(0.025)

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
            if not await asyncio.to_thread(supervisor.close):
                raise ProcessSupervisorError("windows_cleanup_unproven")
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
