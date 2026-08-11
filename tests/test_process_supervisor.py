import asyncio
import os
import sys
from pathlib import Path

import pytest

from vision.process_supervisor import (
    LinuxProcessTree,
    ProcessSupervisorError,
    WindowsJobApiUnavailable,
    WindowsJobProcess,
    run_supervised,
    taskkill_fallback_command,
)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_supervisor_drains_large_stdout_stderr_without_deadlock_and_bounds_tail():
    script = (
        "import sys;"
        "sys.stdout.buffer.write(b'o' * (10 * 1024 * 1024));"
        "sys.stdout.flush();"
        "sys.stderr.buffer.write(b'e' * (10 * 1024 * 1024));"
        "sys.stderr.flush()"
    )

    result = asyncio.run(run_supervised([sys.executable, "-c", script], timeout=10))

    assert result.returncode == 0
    assert result.stdout_total == 10 * 1024 * 1024
    assert result.stderr_total == 10 * 1024 * 1024
    assert len(result.stdout_tail) <= 64 * 1024
    assert len(result.stderr_tail) <= 64 * 1024


def test_linux_timeout_kills_grandchild_and_great_grandchild_that_ignore_term(tmp_path):
    pidfile = tmp_path / "pids.txt"
    script = f"""
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *_: None)
grandchild = subprocess.Popen([sys.executable, "-c", "import signal, subprocess, sys, time, os; signal.signal(signal.SIGTERM, lambda *_: None); great = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, lambda *_: None); time.sleep(60)']); print(os.getpid(), great.pid, flush=True); time.sleep(60)"], stdout=subprocess.PIPE, text=True)
line = grandchild.stdout.readline().strip()
Path = __import__('pathlib').Path
Path({str(pidfile)!r}).write_text(str(os.getpid()) + " " + line, "utf-8")
time.sleep(60)
"""

    with pytest.raises(ProcessSupervisorError, match="supervised_process_timeout"):
        asyncio.run(run_supervised([sys.executable, "-c", script], timeout=0.5))

    pids = [int(value) for value in pidfile.read_text("utf-8").split()]
    assert pids
    assert all(not alive(pid) for pid in pids)


def test_linux_leader_zero_with_active_descendant_is_killed_and_failed(tmp_path):
    pidfile = tmp_path / "pids.txt"
    script = f"""
import os, signal, subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import signal,time,os; signal.signal(signal.SIGTERM, lambda *_: None); print(os.getpid(), flush=True); time.sleep(60)"], stdout=subprocess.PIPE, text=True)
pid = child.stdout.readline().strip()
Path = __import__('pathlib').Path
Path({str(pidfile)!r}).write_text(pid, "utf-8")
os._exit(0)
"""

    with pytest.raises(ProcessSupervisorError, match="supervised_process_descendants_alive"):
        asyncio.run(run_supervised([sys.executable, "-c", script], timeout=5))

    pid = int(pidfile.read_text("utf-8"))
    assert not alive(pid)


class FakeWinApi:
    def __init__(
        self,
        *,
        fail_set=False,
        fail_assign=False,
        fail_resume=False,
        fail_terminate_job=False,
        active_sequence=None,
        stdout_chunks=None,
        stderr_chunks=None,
    ):
        self.calls = []
        self.fail_set = fail_set
        self.fail_assign = fail_assign
        self.fail_resume = fail_resume
        self.fail_terminate_job = fail_terminate_job
        self.active_sequence = list(active_sequence or [0])
        self.stdout_chunks = list(stdout_chunks or [])
        self.stderr_chunks = list(stderr_chunks or [])

    def create_job(self):
        self.calls.append("create_job")
        return 0x1_0000_0001

    def create_pipes(self):
        self.calls.append("create_pipes")
        return {
            "stdout_read": 0x2_0000_0001,
            "stdout_write": 0x2_0000_0002,
            "stderr_read": 0x3_0000_0001,
            "stderr_write": 0x3_0000_0002,
        }

    def set_kill_on_close_and_low_priority(self, job):
        self.calls.append(("set", job))
        if self.fail_set:
            raise WindowsJobApiUnavailable("windows_job_set_failed")

    def create_process_suspended(self, command, pipes):
        self.calls.append(("create_suspended", tuple(command)))
        return {"process": 0x4_0000_0001, "thread": 0x4_0000_0002, "pid": 123}

    def assign_process_to_job(self, job, process):
        self.calls.append(("assign", job, process))
        if self.fail_assign:
            raise WindowsJobApiUnavailable("windows_nested_job_unavailable")

    def resume_thread(self, thread):
        self.calls.append(("resume", thread))
        if self.fail_resume:
            raise WindowsJobApiUnavailable("windows_resume_failed")

    def terminate_process(self, process, code):
        self.calls.append(("terminate_process", process, code))

    def terminate_job(self, job, code):
        self.calls.append(("terminate_job", job, code))
        if self.fail_terminate_job:
            raise ProcessSupervisorError("windows_job_terminate_failed")

    def wait_active_processes_zero(self, job, *, timeout):
        self.calls.append(("wait_active_zero", job, timeout))
        active = self.active_sequence.pop(0) if self.active_sequence else 0
        if active:
            raise ProcessSupervisorError("windows_job_descendants_alive")

    def wait_process(self, process, *, timeout):
        self.calls.append(("wait_process", process, timeout))
        return 0

    def start_pipe_drainers(self, pipes, stdout, stderr):
        self.calls.append(("start_drainers", pipes["stdout_read"], pipes["stderr_read"]))
        for chunk in self.stdout_chunks:
            stdout.append(chunk)
        for chunk in self.stderr_chunks:
            stderr.append(chunk)
        return ["stdout-drainer", "stderr-drainer"]

    def finish_pipe_drainers(self, drainers, *, timeout):
        self.calls.append(("finish_drainers", tuple(drainers), timeout))

    def close_pipes(self, pipes):
        self.calls.append(("close_pipes", pipes["stdout_read"], pipes["stderr_read"]))

    def close_handle(self, handle):
        self.calls.append(("close", handle))

    def taskkill_fallback(self, pid):
        self.calls.append(("taskkill_fallback", pid, 3))


def test_windows_job_starts_suspended_assigns_before_resume_and_terminates_tree():
    api = FakeWinApi()
    process = WindowsJobProcess(["worker.exe", "--probe"], api=api)

    process.start()
    process.terminate_tree()
    process.close()

    assert api.calls[:5] == [
        "create_job",
        "create_pipes",
        ("set", 0x1_0000_0001),
        ("create_suspended", ("worker.exe", "--probe")),
        ("assign", 0x1_0000_0001, 0x4_0000_0001),
    ]
    assert ("resume", 0x4_0000_0002) in api.calls
    assert ("terminate_job", 0x1_0000_0001, 1) in api.calls
    assert ("wait_active_zero", 0x1_0000_0001, 3.0) in api.calls


@pytest.mark.parametrize("fail_set, fail_assign", [(True, False), (False, True)])
def test_windows_job_failures_never_resume_and_close_handles(fail_set, fail_assign):
    api = FakeWinApi(fail_set=fail_set, fail_assign=fail_assign)
    process = WindowsJobProcess(["worker.exe"], api=api)

    with pytest.raises(WindowsJobApiUnavailable):
        process.start()

    assert not any(call[0] == "resume" if isinstance(call, tuple) else False for call in api.calls)
    if fail_assign:
        assert ("terminate_process", 0x4_0000_0001, 1) in api.calls
        assert ("wait_process", 0x4_0000_0001, 3.0) in api.calls
        assert ("close", 0x4_0000_0002) in api.calls
        assert ("close", 0x4_0000_0001) in api.calls
    assert ("close", 0x1_0000_0001) in api.calls


def test_windows_resume_failure_terminates_assigned_job_and_waits_before_close():
    api = FakeWinApi(fail_resume=True)
    process = WindowsJobProcess(["worker.exe"], api=api)

    with pytest.raises(WindowsJobApiUnavailable, match="windows_resume_failed"):
        process.start()

    assert ("terminate_job", 0x1_0000_0001, 1) in api.calls
    assert ("wait_active_zero", 0x1_0000_0001, 3.0) in api.calls
    assert ("close", 0x4_0000_0002) in api.calls
    assert ("close", 0x4_0000_0001) in api.calls


def test_windows_job_wait_returns_probe_stdout_json_and_bounded_10mb_tails():
    api = FakeWinApi(
        stdout_chunks=[b'{"probe":"official-catvton-worker"}\n', b"x" * (10 * 1024 * 1024)],
        stderr_chunks=[b"e" * (10 * 1024 * 1024)],
    )
    process = WindowsJobProcess(["worker.exe", "--probe"], api=api)

    process.start()
    result = process.wait(timeout=5)
    process.close()

    assert result.returncode == 0
    assert result.stdout_total > 10 * 1024 * 1024
    assert result.stderr_total == 10 * 1024 * 1024
    assert len(result.stdout_tail) <= 64 * 1024
    assert len(result.stderr_tail) <= 64 * 1024
    assert any(call[0] == "start_drainers" for call in api.calls if isinstance(call, tuple))


def test_windows_job_api_creates_inheritable_write_pipes_and_noninheritable_read_pipes():
    source = (Path(__file__).parents[1] / "vision" / "process_supervisor.py").read_text("utf-8")

    assert "class SECURITY_ATTRIBUTES" in source
    assert "(\"bInheritHandle\", wintypes.BOOL)" in source
    assert "attributes.bInheritHandle = True" in source
    assert "CreatePipe(self.ctypes.byref(stdout_read), self.ctypes.byref(stdout_write), self.ctypes.byref(attributes), 0)" in source
    assert "CreatePipe(self.ctypes.byref(stderr_read), self.ctypes.byref(stderr_write), self.ctypes.byref(attributes), 0)" in source
    assert "SetHandleInformation(read_handle, HANDLE_FLAG_INHERIT, 0)" in source
    assert "SetHandleInformation(stdout_write" not in source
    assert "SetHandleInformation(stderr_write" not in source


def test_windows_leader_exit_with_active_descendant_terminates_job_and_fails():
    api = FakeWinApi(active_sequence=[2])
    process = WindowsJobProcess(["worker.exe"], api=api)
    process.start()

    with pytest.raises(ProcessSupervisorError, match="windows_job_descendants_alive"):
        process.wait(timeout=5)

    assert ("terminate_job", 0x1_0000_0001, 1) in api.calls


def test_windows_job_api_terminate_failure_uses_bounded_taskkill_fallback():
    api = FakeWinApi(fail_terminate_job=True)
    process = WindowsJobProcess(["worker.exe"], api=api)
    process.start()

    with pytest.raises(ProcessSupervisorError, match="windows_job_terminate_failed"):
        process.terminate_tree()

    assert ("taskkill_fallback", 123, 3) in api.calls


def test_taskkill_fallback_is_bounded_devnull_command():
    assert taskkill_fallback_command(1234) == ["taskkill", "/PID", "1234", "/T", "/F"]
