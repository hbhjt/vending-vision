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
    def __init__(self, *, fail_set=False, fail_assign=False):
        self.calls = []
        self.fail_set = fail_set
        self.fail_assign = fail_assign

    def create_job(self):
        self.calls.append("create_job")
        return "job"

    def set_kill_on_close_and_low_priority(self, job):
        self.calls.append(("set", job))
        if self.fail_set:
            raise WindowsJobApiUnavailable("windows_job_set_failed")

    def create_process_suspended(self, command):
        self.calls.append(("create_suspended", tuple(command)))
        return {"process": "process", "thread": "thread", "pid": 123}

    def assign_process_to_job(self, job, process):
        self.calls.append(("assign", job, process))
        if self.fail_assign:
            raise WindowsJobApiUnavailable("windows_nested_job_unavailable")

    def resume_thread(self, thread):
        self.calls.append(("resume", thread))

    def terminate_process(self, process, code):
        self.calls.append(("terminate_process", process, code))

    def terminate_job(self, job, code):
        self.calls.append(("terminate_job", job, code))

    def wait_active_processes_zero(self, job, *, timeout):
        self.calls.append(("wait_active_zero", job, timeout))

    def close_handle(self, handle):
        self.calls.append(("close", handle))


def test_windows_job_starts_suspended_assigns_before_resume_and_terminates_tree():
    api = FakeWinApi()
    process = WindowsJobProcess(["worker.exe", "--probe"], api=api)

    process.start()
    process.terminate_tree()
    process.close()

    assert api.calls[:5] == [
        "create_job",
        ("set", "job"),
        ("create_suspended", ("worker.exe", "--probe")),
        ("assign", "job", "process"),
        ("resume", "thread"),
    ]
    assert ("terminate_job", "job", 1) in api.calls
    assert ("wait_active_zero", "job", 3.0) in api.calls


@pytest.mark.parametrize("fail_set, fail_assign", [(True, False), (False, True)])
def test_windows_job_failures_never_resume_and_close_handles(fail_set, fail_assign):
    api = FakeWinApi(fail_set=fail_set, fail_assign=fail_assign)
    process = WindowsJobProcess(["worker.exe"], api=api)

    with pytest.raises(WindowsJobApiUnavailable):
        process.start()

    assert not any(call[0] == "resume" if isinstance(call, tuple) else False for call in api.calls)
    if fail_assign:
        assert ("terminate_process", "process", 1) in api.calls
        assert ("close", "thread") in api.calls
        assert ("close", "process") in api.calls
    assert ("close", "job") in api.calls


def test_taskkill_fallback_is_bounded_devnull_command():
    assert taskkill_fallback_command(1234) == ["taskkill", "/PID", "1234", "/T", "/F"]
