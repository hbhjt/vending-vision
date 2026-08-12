import hashlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
import zipfile

import pytest

from scripts.download_verified_archive import ArchiveError, download_verified_archive


class _Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _SlowResponse(_Response):
    def __init__(self, payload, url, clock):
        super().__init__(payload, url)
        self._clock = clock

    def read(self, _size=-1):
        self._clock.advance(0.4)
        return super().read(1)


class _ExitAdvancingResponse(_Response):
    def __init__(self, payload, url, clock):
        super().__init__(payload, url)
        self._clock = clock

    def __exit__(self, *_args):
        super().__exit__(*_args)
        self._clock.advance(2.0)


def test_archive_total_deadline_aborts_slow_trickle_and_cleans_workdir(tmp_path):
    url = "https://example.invalid/slow.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _SlowResponse(payload, url, clock)

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_archive_opener_uses_remaining_timeout_and_checks_post_open(tmp_path):
    url = "https://example.invalid/open.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _Response(payload, url)
    timeouts = []

    def opener(_request, timeout):
        timeouts.append(timeout)
        clock.advance(5.1)
        return response

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=opener,
            total_timeout_seconds=5.0,
            monotonic=clock,
        )

    assert timeouts == [pytest.approx(5.0)]
    assert response.closed
    assert not destination.exists()


def test_archive_deadline_after_exact_digest_does_not_publish(tmp_path, monkeypatch):
    url = "https://example.invalid/exact.zip"
    payload = _zip("demo.whl")
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    response = _ExitAdvancingResponse(payload, url, clock)

    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            expected_sha256,
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: response,
            total_timeout_seconds=1.0,
            monotonic=clock,
        )

    assert response.closed
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_archive_publish_race_rejects_and_preserves_other_destination(
    tmp_path, monkeypatch
):
    from scripts import download_verified_archive as downloader

    url = "https://example.invalid/race.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    real_publish = downloader._publish_directory_no_replace

    def race_publish(source, target, check_deadline):
        target.mkdir()
        (target / "sentinel").write_text("other actor", "utf-8")
        return real_publish(source, target, check_deadline)

    monkeypatch.setattr(
        "scripts.download_verified_archive._publish_directory_no_replace",
        race_publish,
        raising=False,
    )
    with pytest.raises(ArchiveError, match="archive_destination_exists"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: _Response(payload, url),
        )

    assert (destination / "sentinel").read_text("utf-8") == "other actor"
    assert sorted(path.name for path in tmp_path.iterdir()) == [destination.name]


def test_windows_directory_publish_uses_movefileex_without_replace_flags(tmp_path):
    from scripts import download_verified_archive as downloader

    calls = []

    class Kernel32:
        def MoveFileExW(self, source, destination, flags):
            calls.append((source, destination, flags))
            return 1

    api = downloader._WindowsMoveApi(Kernel32())
    assert api.move_no_replace(tmp_path / "source", tmp_path / "destination") == (
        True,
        0,
    )
    assert calls == [(str(tmp_path / "source"), str(tmp_path / "destination"), 0)]

    calls.clear()

    class MoveApi:
        def move_no_replace(self, source, destination):
            calls.append((source, destination))
            return False, 183

    with pytest.raises(ArchiveError, match="archive_destination_exists"):
        downloader._publish_directory_no_replace(
            tmp_path / "source",
            tmp_path / "destination",
            platform="nt",
            windows_api=MoveApi(),
        )
    assert calls == [(tmp_path / "source", tmp_path / "destination")]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group tracer")
def test_archive_extract_deadline_kills_ignore_term_descendant_tree(tmp_path):
    from scripts import download_verified_archive as downloader

    url = "https://example.invalid/blocked.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    worker = tmp_path / "blocked-extractor.py"
    child_pid = tmp_path / "child.pid"
    worker.write_text(
        "import os,signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        "open(sys.argv[1],'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        "utf-8",
    )
    started = time.monotonic()
    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: _Response(payload, url),
            total_timeout_seconds=0.05,
            extractor_command=[sys.executable, str(worker), str(child_pid)],
        )

    assert time.monotonic() - started < 2.0
    assert child_pid.is_file()
    pid = int(child_pid.read_text("utf-8"))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"extractor descendant still alive: {pid}")
    assert not destination.exists()
    assert {path.name for path in tmp_path.iterdir()} == {
        child_pid.name,
        worker.name,
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group tracer")
@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("import sys; sys.exit(7)", "archive_extract_failed"),
        ("import sys; sys.stdout.write('x'*70000)", "archive_process_output"),
    ],
)
def test_archive_extractor_nonzero_and_output_cap_leave_no_partial(
    tmp_path, program, expected
):
    url = "https://example.invalid/extractor-failure.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    with pytest.raises(ArchiveError, match=expected):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: _Response(payload, url),
            extractor_command=[sys.executable, "-c", program],
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


class _FakeWindowsJobApi:
    def __init__(self, *, timeout=False):
        self.timeout = timeout
        self.events = []

    def create_job(self):
        self.events.append("create_job")
        return "job"

    def create_suspended(self, command):
        self.events.append(("create_suspended", command))
        return "process", "thread"

    def assign(self, job, process):
        self.events.append(("assign", job, process))

    def resume(self, thread):
        self.events.append(("resume", thread))

    def wait(self, process, timeout):
        self.events.append(("wait", process, timeout))
        if self.timeout:
            self.timeout = False
            raise TimeoutError
        return 0

    def terminate(self, job):
        self.events.append(("terminate", job))

    def terminate_process(self, process):
        self.events.append(("terminate_process", process))

    def wait_active_zero(self, job, timeout):
        self.events.append(("active_zero", job, timeout))

    def close(self, handle):
        self.events.append(("close", handle))


def test_windows_job_supervisor_assigns_suspended_process_before_resume():
    from scripts import download_verified_archive as downloader

    api = _FakeWindowsJobApi()
    downloader._run_windows_extractor(["extractor.exe"], 3.0, api)

    assert api.events[:4] == [
        "create_job",
        ("create_suspended", ["extractor.exe"]),
        ("assign", "job", "process"),
        ("resume", "thread"),
    ]
    assert ("active_zero", "job", 1.0) in api.events
    assert api.events[-3:] == [
        ("close", "thread"),
        ("close", "process"),
        ("close", "job"),
    ]


def test_windows_job_supervisor_timeout_terminates_and_proves_tree_dead():
    from scripts import download_verified_archive as downloader

    api = _FakeWindowsJobApi(timeout=True)
    with pytest.raises(ArchiveError, match="archive_timeout"):
        downloader._run_windows_extractor(["extractor.exe"], 0.05, api)

    assert ("terminate", "job") in api.events
    assert ("active_zero", "job", 1.0) in api.events
    assert api.events[-1] == ("close", "job")


def test_windows_job_supervisor_assign_failure_terminates_suspended_leader():
    from scripts import download_verified_archive as downloader

    class AssignFailure(_FakeWindowsJobApi):
        def assign(self, job, process):
            super().assign(job, process)
            raise ArchiveError("archive_windows_job_assign")

    api = AssignFailure()
    with pytest.raises(ArchiveError, match="archive_windows_job_assign"):
        downloader._run_windows_extractor(["extractor.exe"], 3.0, api)

    assert ("terminate_process", "process") in api.events
    assert ("active_zero", "job", 1.0) in api.events
    assert api.events[-1] == ("close", "job")


def test_windows_job_supervisor_cleanup_failure_is_failclosed():
    from scripts import download_verified_archive as downloader

    class StubbornTree(_FakeWindowsJobApi):
        def wait_active_zero(self, job, timeout):
            super().wait_active_zero(job, timeout)
            raise ArchiveError("archive_windows_tree_alive")

    api = StubbornTree(timeout=True)
    with pytest.raises(ArchiveError, match="archive_windows_cleanup_unproven"):
        downloader._run_windows_extractor(["extractor.exe"], 0.05, api)

    assert ("terminate", "job") in api.events
    assert ("active_zero", "job", 1.0) in api.events
    assert api.events[-1] == ("close", "job")


def test_archive_cleanup_retries_before_no_replace_publish(tmp_path, monkeypatch):
    from scripts import download_verified_archive as downloader

    url = "https://example.invalid/retry.zip"
    payload = _zip("demo.whl")
    real_rmtree = downloader.shutil.rmtree
    attempts = []

    def transient_rmtree(path):
        attempts.append(Path(path))
        if len(attempts) == 1:
            raise OSError("transient")
        return real_rmtree(path)

    monkeypatch.setattr(downloader.shutil, "rmtree", transient_rmtree)
    destination = tmp_path / "wheelhouse"
    download_verified_archive(
        url,
        hashlib.sha256(payload).hexdigest(),
        destination,
        expected_bytes=len(payload),
        opener=lambda _request, timeout: _Response(payload, url),
    )

    assert len(attempts) == 2
    assert (destination / "demo.whl").read_bytes() == b"wheel"


def test_archive_cleanup_failure_does_not_publish_and_retry_can_succeed(
    tmp_path, monkeypatch
):
    from scripts import download_verified_archive as downloader

    url = "https://example.invalid/cleanup.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    real_rmtree = downloader.shutil.rmtree
    monkeypatch.setattr(
        downloader.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(OSError("blocked")),
    )
    with pytest.raises(ArchiveError, match="archive_cleanup_failed"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: _Response(payload, url),
        )
    assert not destination.exists()

    monkeypatch.setattr(downloader.shutil, "rmtree", real_rmtree)
    for owned in tmp_path.glob(".wheelhouse-*"):
        real_rmtree(owned)
    download_verified_archive(
        url,
        hashlib.sha256(payload).hexdigest(),
        destination,
        expected_bytes=len(payload),
        opener=lambda _request, timeout: _Response(payload, url),
    )
    assert (destination / "demo.whl").read_bytes() == b"wheel"


def test_archive_deadline_is_checked_immediately_before_publish(tmp_path, monkeypatch):
    from scripts import download_verified_archive as downloader

    url = "https://example.invalid/publish-deadline.zip"
    payload = _zip("demo.whl")
    destination = tmp_path / "wheelhouse"
    clock = _Clock()
    real_publish = downloader._publish_directory_no_replace

    def boundary(source, target, check_deadline):
        clock.advance(2.0)
        return real_publish(source, target, check_deadline)

    monkeypatch.setattr(downloader, "_publish_directory_no_replace", boundary)
    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            destination,
            expected_bytes=len(payload),
            opener=lambda _request, timeout: _Response(payload, url),
            total_timeout_seconds=1.0,
            monotonic=clock,
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), 3600.1, True])
def test_archive_total_timeout_is_finite_positive_and_capped(tmp_path, timeout):
    with pytest.raises(ArchiveError, match="archive_timeout"):
        download_verified_archive(
            "https://example.invalid/file.zip",
            "0" * 64,
            tmp_path / "wheelhouse",
            expected_bytes=1,
            opener=lambda _request, timeout: _Response(
                b"x", "https://example.invalid/file.zip"
            ),
            total_timeout_seconds=timeout,
        )


def _zip(member):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(member, b"wheel")
    return output.getvalue()


def test_verified_archive_download_extracts_only_digest_bound_safe_members(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("demo.whl")

    def opener(_request, timeout):
        assert timeout == 120.0
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    download_verified_archive(
        url, hashlib.sha256(payload).hexdigest(), destination,
        expected_bytes=len(payload), opener=opener,
    )

    assert (destination / "demo.whl").read_bytes() == b"wheel"


def test_verified_archive_rejects_traversal_without_publishing_destination(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = _zip("../escape.whl")

    def opener(_request, timeout):
        return _Response(payload, url)

    destination = tmp_path / "wheelhouse"
    with pytest.raises(ArchiveError, match="archive_unsafe_path"):
        download_verified_archive(
            url, hashlib.sha256(payload).hexdigest(), destination,
            expected_bytes=len(payload), opener=opener,
        )

    assert not destination.exists()
    assert not (tmp_path / "escape.whl").exists()


@pytest.mark.parametrize("case", ["symlink", "collision", "oversize", "special"])
def test_verified_archive_rejects_symlink_special_collision_and_size(tmp_path, case):
    if case == "special":
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            member = tarfile.TarInfo("fifo")
            member.type = tarfile.FIFOTYPE
            archive.addfile(member)
        payload = output.getvalue()
    else:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            if case == "symlink":
                member = zipfile.ZipInfo("link")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "target")
            elif case == "collision":
                archive.writestr("Demo.whl", b"one")
                archive.writestr("demo.whl", b"two")
            else:
                archive.writestr("large.whl", b"12345")
        payload = output.getvalue()
    url = "https://example.invalid/core-wheelhouse.archive"

    def opener(_request, timeout):
        return _Response(payload, url)

    with pytest.raises(ArchiveError, match="archive_"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload),
            opener=opener,
            max_extracted_bytes=4 if case == "oversize" else 1024,
        )
    assert not (tmp_path / "wheelhouse").exists()


def test_archive_downloader_imports_in_clean_stdlib_only_python(tmp_path):
    clean_venv = tmp_path / "clean-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(clean_venv)],
        check=True,
    )
    clean_python = clean_venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = Path(__file__).parents[1] / "scripts" / "download_verified_archive.py"

    completed = subprocess.run(
        [str(clean_python), "-I", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    worker = script.with_name("archive_extractor_worker.py")
    worker_help = subprocess.run(
        [str(clean_python), "-I", str(worker), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert worker_help.returncode == 0, worker_help.stderr


def test_archive_downloader_aborts_oversized_stream_before_writing_chunk(tmp_path, monkeypatch):
    url = "https://example.invalid/core-wheelhouse.zip"
    payload = b"expected-plus-one"
    writes = []
    real_open = Path.open

    class Sink(io.BytesIO):
        def write(self, value):
            writes.append(len(value))
            return super().write(value)

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name == "payload.archive" and mode == "xb":
            return Sink()
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ArchiveError, match="archive_download_size"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload) - 1,
            max_download_bytes=len(payload) + 10,
            opener=lambda _request, timeout: _Response(payload, url),
        )

    assert writes == []
    assert not (tmp_path / "wheelhouse").exists()


def test_verified_archive_rejects_member_count_before_extracting(tmp_path):
    url = "https://example.invalid/core-wheelhouse.zip"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("one.whl", b"one")
        archive.writestr("two.whl", b"two")
    payload = output.getvalue()

    with pytest.raises(ArchiveError, match="archive_member_count"):
        download_verified_archive(
            url,
            hashlib.sha256(payload).hexdigest(),
            tmp_path / "wheelhouse",
            expected_bytes=len(payload),
            max_members=1,
            opener=lambda _request, timeout: _Response(payload, url),
        )
    assert not (tmp_path / "wheelhouse").exists()
