import ctypes
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

import vision.ai_acceptance_evidence as acceptance_evidence
from vision.ai_acceptance_evidence import publish_completed_ai_regional_evidence


def canonical_sidecar() -> bytes:
    return json.dumps(
        {
            "kind": "regional-evidence",
            "schemaVersion": "vem-ai-regional-evidence/v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


class FakeWindowsKernel32:
    """Small filesystem-backed Kernel32 double for the sink's held-handle ABI."""

    def __init__(self):
        self.calls = []
        self.handles = {}
        self.next_handle = 100
        self.write_failure = False
        self.flush_failure = False
        self.flush_failure_once = False
        self.flush_hook = None
        self.directory_open_failure = False
        self.delete_failure_once = False
        self.close_fail_once = False

    def CreateFileW(self, path, access, share, _security, creation, flags, _template):
        target = Path(path)
        self.calls.append(("CreateFileW", target, access, share, creation, flags))
        try:
            if target.exists():
                target_facts = target.stat()
                target_identity = target_facts.st_dev, target_facts.st_ino
                for entry in self.handles.values():
                    entry_facts = entry["path"].stat()
                    if (entry_facts.st_dev, entry_facts.st_ino) != target_identity:
                        continue
                    if (
                        access & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_READ
                        and not entry["share"]
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_READ
                    ) or (
                        access & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_WRITE
                        and not entry["share"]
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
                    ) or (
                        access & acceptance_evidence._WindowsEvidenceFileApi.DELETE
                        and not entry["share"]
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_DELETE
                    ) or (
                        entry["access"]
                        & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_READ
                        and not share
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_READ
                    ) or (
                        entry["access"]
                        & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_WRITE
                        and not share
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
                    ) or (
                        entry["access"] & acceptance_evidence._WindowsEvidenceFileApi.DELETE
                        and not share
                        & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_DELETE
                    ):
                        return -1
            if flags & acceptance_evidence._WindowsEvidenceFileApi.FILE_FLAG_BACKUP_SEMANTICS:
                if not target.is_dir() or self.directory_open_failure:
                    return -1
                stream = None
            elif creation == acceptance_evidence._WindowsEvidenceFileApi.CREATE_NEW:
                stream = target.open("x+b")
            else:
                stream = target.open("rb")
        except OSError:
            return -1
        handle = self.next_handle
        self.next_handle += 1
        self.handles[handle] = {
            "path": target,
            "stream": stream,
            "share": share,
            "access": access,
            "flags": flags,
        }
        return handle

    def WriteFile(self, handle, buffer, size, written, _overlapped):
        if self.write_failure:
            return 0
        stream = self.handles[handle]["stream"]
        stream.write(buffer.raw[:size])
        stream.flush()
        written._obj.value = size
        self.calls.append(("WriteFile", handle, size))
        return 1

    def FlushFileBuffers(self, handle):
        self.calls.append(
            ("FlushFileBuffers", handle, self.handles[handle]["access"])
        )
        stream = self.handles[handle]["stream"]
        if not self.handles[handle]["access"] & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_WRITE:
            return 0
        if self.flush_hook is not None:
            self.flush_hook(handle)
        if self.flush_failure:
            return 0
        if self.flush_failure_once:
            self.flush_failure_once = False
            return 0
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
        return 1

    def CreateHardLinkW(self, destination, source, _security):
        self.calls.append(("CreateHardLinkW", Path(destination), Path(source)))
        destination_parent = Path(destination).parent
        for entry in self.handles.values():
            if (
                entry["flags"]
                & acceptance_evidence._WindowsEvidenceFileApi.FILE_FLAG_BACKUP_SEMANTICS
                and entry["path"] == destination_parent
                and not entry["share"]
                & acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
            ):
                return 0
        try:
            os.link(source, destination)
        except OSError:
            return 0
        return 1

    def GetFileInformationByHandle(self, handle, information):
        entry = self.handles[handle]
        facts = entry["path"].stat()
        value = information._obj
        value.dwVolumeSerialNumber = facts.st_dev & 0xFFFFFFFF
        value.nFileIndexHigh = (facts.st_ino >> 32) & 0xFFFFFFFF
        value.nFileIndexLow = facts.st_ino & 0xFFFFFFFF
        value.nFileSizeHigh = (facts.st_size >> 32) & 0xFFFFFFFF
        value.nFileSizeLow = facts.st_size & 0xFFFFFFFF
        self.calls.append(("GetFileInformationByHandle", handle))
        return 1

    def ReadFile(self, handle, buffer, size, read, _overlapped):
        stream = self.handles[handle]["stream"]
        chunk = stream.read(size)
        buffer.raw = chunk + b"\0" * (size - len(chunk))
        read._obj.value = len(chunk)
        self.calls.append(("ReadFile", handle, len(chunk)))
        return 1

    def DeleteFileW(self, path):
        self.calls.append(("DeleteFileW", Path(path)))
        if self.delete_failure_once:
            self.delete_failure_once = False
            return 0
        try:
            Path(path).unlink()
        except OSError:
            return 0
        return 1

    def CloseHandle(self, handle):
        self.calls.append(("CloseHandle", handle))
        if self.close_fail_once:
            self.close_fail_once = False
            return 0
        entry = self.handles.pop(handle, None)
        if entry is None:
            return 0
        if entry["stream"] is not None:
            entry["stream"].close()
        return 1

    def mutation_is_blocked(self, path, share_flag):
        target = Path(path)
        target_identity = target.stat().st_dev, target.stat().st_ino
        for entry in self.handles.values():
            facts = entry["path"].stat()
            if (facts.st_dev, facts.st_ino) == target_identity:
                return not entry["share"] & share_flag
        return False

    def atomic_replace(self, destination, replacement):
        if self.mutation_is_blocked(
            destination, acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_DELETE
        ):
            return False
        os.replace(replacement, destination)
        return True

    def inplace_write(self, destination, content):
        if self.mutation_is_blocked(
            destination, acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
        ):
            return False
        destination.write_bytes(content)
        return True

    def rename_directory(self, root, old_root):
        if self.mutation_is_blocked(
            root, acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_DELETE
        ):
            return False
        root.rename(old_root)
        root.mkdir()
        return True


def windows_sink(monkeypatch, kernel):
    monkeypatch.setattr(acceptance_evidence, "_WINDOWS", True)
    monkeypatch.setattr(
        acceptance_evidence,
        "_windows_file_api_factory",
        lambda: acceptance_evidence._WindowsEvidenceFileApi(kernel),
    )


def test_windows_acceptance_sink_publishes_with_held_read_only_handles(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    windows_sink(monkeypatch, kernel)
    attempt_id = str(uuid4())

    destination = publish_completed_ai_regional_evidence(
        attempt_id, canonical_sidecar()
    )

    assert destination == root / f"{attempt_id}.regional-evidence.json"
    assert destination.read_bytes() == canonical_sidecar()
    assert {path.name for path in root.iterdir()} == {destination.name}
    file_opens = [call for call in kernel.calls if call[0] == "CreateFileW"]
    directory_opens = [
        call
        for call in file_opens
        if call[5]
        & acceptance_evidence._WindowsEvidenceFileApi.FILE_FLAG_BACKUP_SEMANTICS
    ]
    assert all(
        call[3]
        == (
            acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_READ
            | acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
        )
        for call in directory_opens
    )
    regular_file_opens = [call for call in file_opens if call not in directory_opens]
    published_read_opens = [
        call
        for call in regular_file_opens
        if call[2] == acceptance_evidence._WindowsEvidenceFileApi.GENERIC_READ
        and call[4] == acceptance_evidence._WindowsEvidenceFileApi.OPEN_EXISTING
    ]
    assert [call[3] for call in published_read_opens] == [
        acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_READ
        | acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_WRITE
    ]
    assert all(
        call[3] == acceptance_evidence._WindowsEvidenceFileApi.FILE_SHARE_READ
        for call in regular_file_opens
        if call not in published_read_opens
    )
    assert any(call[0] == "CreateHardLinkW" for call in kernel.calls)
    assert any(call[0] == "ReadFile" for call in kernel.calls)
    flushes = [call for call in kernel.calls if call[0] == "FlushFileBuffers"]
    assert flushes
    assert all(
        call[2] & acceptance_evidence._WindowsEvidenceFileApi.GENERIC_WRITE
        for call in flushes
    )
    assert any(call[0] == "DeleteFileW" for call in kernel.calls)
    source_open = next(
        call
        for call in file_opens
        if call[4] == acceptance_evidence._WindowsEvidenceFileApi.CREATE_NEW
    )
    source_handle = 100 + file_opens.index(source_open)
    read_index = next(
        index
        for index, call in enumerate(kernel.calls)
        if call[0] == "ReadFile"
    )
    close_source_index = next(
        index
        for index, call in enumerate(kernel.calls)
        if call == ("CloseHandle", source_handle)
    )
    assert close_source_index > read_index


def test_windows_acceptance_sink_preserves_preexisting_destination(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    destination.write_bytes(b"external publisher bytes\n")
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    windows_sink(monkeypatch, kernel)

    with pytest.raises(RuntimeError, match="exists"):
        publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())

    assert destination.read_bytes() == b"external publisher bytes\n"
    assert [call[0] for call in kernel.calls] == [
        "CreateFileW",
        "GetFileInformationByHandle",
        "CloseHandle",
    ]


def test_windows_acceptance_sink_rejects_root_replacement_after_claim(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    old_root = tmp_path / "acceptance-old"
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    windows_sink(monkeypatch, kernel)
    attempted = False

    def replace_root_after_claim(_handle):
        nonlocal attempted
        if not attempted:
            attempted = True
            assert kernel.rename_directory(root, old_root) is False
            kernel.flush_failure_once = True

    kernel.flush_hook = replace_root_after_claim

    with pytest.raises(RuntimeError, match="windows_handle_flush"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert attempted is True
    assert old_root.exists() is False
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("mutation", ["atomic", "inplace"])
def test_windows_acceptance_sink_held_handles_block_external_mutation(
    tmp_path, monkeypatch, mutation
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    windows_sink(monkeypatch, kernel)
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"external publisher bytes\n")

    def attempt_mutation():
        if mutation == "atomic":
            assert kernel.atomic_replace(destination, replacement) is False
        else:
            assert kernel.inplace_write(destination, b"in-place mutation\n") is False

    def mutate_after_publish(_handle):
        if destination.exists():
            attempt_mutation()

    kernel.flush_hook = mutate_after_publish
    assert publish_completed_ai_regional_evidence(
        attempt_id, canonical_sidecar()
    ) == destination
    assert destination.read_bytes() == canonical_sidecar()
    assert {path.name for path in root.iterdir()} == {destination.name}


def test_windows_acceptance_sink_rolls_back_on_write_flush_or_close_failure(
    tmp_path, monkeypatch
):
    for failure in ("write", "flush", "close"):
        root = tmp_path / failure
        root.mkdir()
        monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
        kernel = FakeWindowsKernel32()
        if failure == "write":
            kernel.write_failure = True
        elif failure == "flush":
            kernel.flush_failure_once = True
        else:
            kernel.close_fail_once = True
        windows_sink(monkeypatch, kernel)

        with pytest.raises(RuntimeError, match="windows_handle_(write|flush|close)"):
            publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

        assert list(root.iterdir()) == []


def test_windows_acceptance_sink_rolls_back_when_temp_delete_or_claim_cleanup_fails(
    tmp_path, monkeypatch
):
    for failure in ("delete", "claim"):
        root = tmp_path / failure
        root.mkdir()
        monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
        kernel = FakeWindowsKernel32()
        if failure == "delete":
            kernel.delete_failure_once = True
        else:
            original_rmdir = Path.rmdir
            failed = False

            def fail_claim_once(path):
                nonlocal failed
                if path.name == ".ai-regional-evidence-publishing" and not failed:
                    failed = True
                    raise OSError("claim cleanup failed")
                return original_rmdir(path)

            monkeypatch.setattr(Path, "rmdir", fail_claim_once)
        windows_sink(monkeypatch, kernel)

        with pytest.raises(
            (RuntimeError, OSError), match="windows_path_delete|claim cleanup failed"
        ):
            publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

        assert list(root.iterdir()) == []


def test_windows_acceptance_sink_rolls_back_when_claim_directory_handle_cannot_open(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    kernel.directory_open_failure = True
    windows_sink(monkeypatch, kernel)

    with pytest.raises(RuntimeError, match="windows_handle_open"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert list(root.iterdir()) == []


def test_windows_acceptance_sink_blocks_external_replacement_until_publish_finishes(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    kernel = FakeWindowsKernel32()
    windows_sink(monkeypatch, kernel)
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"external publisher bytes\n")

    replacement_attempted = False

    def attempt_replace_while_source_is_held(_handle):
        nonlocal replacement_attempted
        if destination.exists() and replacement.exists():
            replacement_attempted = True
            assert not kernel.atomic_replace(destination, replacement)

    kernel.flush_hook = attempt_replace_while_source_is_held

    published = publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())

    assert replacement_attempted
    assert published == destination
    assert destination.read_bytes() == canonical_sidecar()
    assert replacement.read_bytes() == b"external publisher bytes\n"
    assert {path.name for path in root.iterdir()} == {destination.name}


def test_acceptance_sink_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", raising=False)
    assert publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar()) is None


def test_acceptance_sink_requires_absolute_regular_root_and_exclusive_output(
    tmp_path, monkeypatch
):
    attempt_id = str(uuid4())
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    destination = publish_completed_ai_regional_evidence(
        attempt_id, canonical_sidecar()
    )
    assert destination == root / f"{attempt_id}.regional-evidence.json"
    original = destination.read_bytes()
    with pytest.raises(RuntimeError, match="exists"):
        publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())
    assert destination.read_bytes() == original
    assert {path.name for path in root.iterdir()} == {destination.name}


def test_acceptance_sink_rejects_symlink_root_and_noncanonical_bytes(
    tmp_path, monkeypatch
):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(linked))
    with pytest.raises(RuntimeError, match="root_invalid"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(target))
    with pytest.raises(RuntimeError, match="noncanonical"):
        publish_completed_ai_regional_evidence(
            str(uuid4()), b'{"schemaVersion": "vem-ai-regional-evidence/v1"}'
        )
    assert list(target.iterdir()) == []


def test_acceptance_sink_rejects_a_nonempty_invocation_root_without_touching_it(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    stale = root / "stale-regional-evidence.json"
    stale.write_bytes(canonical_sidecar())
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))

    with pytest.raises(RuntimeError, match="root_not_empty"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert stale.read_bytes() == canonical_sidecar()
    assert {path.name for path in root.iterdir()} == {stale.name}


@pytest.mark.skipif(os.name == "nt", reason="POSIX unlink publication path")
def test_acceptance_sink_rolls_back_post_link_temporary_unlink_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    original_unlink = Path.unlink
    failed = False

    def fail_once_for_publish_temporary(path, *args, **kwargs):
        nonlocal failed
        if path.parent == root and path.name.startswith(".") and not failed:
            failed = True
            raise OSError("post-link temporary unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once_for_publish_temporary)

    with pytest.raises(OSError, match="post-link temporary unlink failed"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert failed is True
    assert list(root.iterdir()) == []


def test_acceptance_sink_rolls_back_when_claim_cleanup_initially_fails(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    original_rmdir = Path.rmdir
    failed = False

    def fail_once_for_claim(path):
        nonlocal failed
        if (
            path.parent == root
            and path.name == ".ai-regional-evidence-publishing"
            and not failed
        ):
            failed = True
            raise OSError("claim cleanup failed")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_once_for_claim)

    with pytest.raises(OSError, match="claim cleanup failed"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert failed is True
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync publication path")
@pytest.mark.parametrize(
    ("failing_call", "message", "expected_calls"),
    [
        (1, "initial directory fsync failed", 2),
        (2, "final directory fsync failed", 3),
    ],
)
def test_acceptance_sink_rolls_back_when_directory_fsync_fails(
    tmp_path, monkeypatch, failing_call, message, expected_calls
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    calls: list[Path] = []

    def fail_selected_fsync(path):
        calls.append(path)
        if len(calls) == failing_call:
            raise OSError(message)

    monkeypatch.setattr(
        acceptance_evidence, "_fsync_directory", fail_selected_fsync
    )

    with pytest.raises(OSError, match=message):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert calls == [root] * expected_calls
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX unlink publication path")
def test_acceptance_sink_chains_claim_cleanup_without_hiding_post_link_failure(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir
    temporary_failed = False
    claim_failed = False

    def fail_once_for_publish_temporary(path, *args, **kwargs):
        nonlocal temporary_failed
        if path.parent == root and path.name.startswith(".") and not temporary_failed:
            temporary_failed = True
            raise OSError("post-link temporary unlink failed")
        return original_unlink(path, *args, **kwargs)

    def fail_once_for_claim(path):
        nonlocal claim_failed
        if (
            path.parent == root
            and path.name == ".ai-regional-evidence-publishing"
            and not claim_failed
        ):
            claim_failed = True
            raise OSError("claim cleanup failed")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "unlink", fail_once_for_publish_temporary)
    monkeypatch.setattr(Path, "rmdir", fail_once_for_claim)

    with pytest.raises(ExceptionGroup, match="publication failed") as raised:
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert [str(error) for error in raised.value.exceptions] == [
        "post-link temporary unlink failed",
        "claim cleanup failed",
    ]
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX unlink publication path")
def test_acceptance_sink_does_not_delete_an_external_post_link_replacement(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    replacement = tmp_path / "external-replacement.json"
    replacement.write_bytes(b"external publisher bytes\n")
    original_unlink = Path.unlink
    replaced = False

    def replace_then_fail_temporary(path, *args, **kwargs):
        nonlocal replaced
        if path.parent == root and path.name.startswith(".") and not replaced:
            replaced = True
            os.replace(replacement, destination)
            raise OSError("post-link temporary unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", replace_then_fail_temporary)

    with pytest.raises(OSError, match="post-link temporary unlink failed"):
        publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())

    assert replaced is True
    assert destination.read_bytes() == b"external publisher bytes\n"
    assert {path.name for path in root.iterdir()} == {destination.name}


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync publication path")
def test_acceptance_sink_rejects_replacement_during_first_directory_fsync(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    replacement = tmp_path / "external-replacement.json"
    replacement.write_bytes(b"external publisher bytes\n")
    fsync_calls = 0

    def replace_on_first_fsync(path):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            assert path == root
            os.replace(replacement, destination)

    monkeypatch.setattr(
        acceptance_evidence, "_fsync_directory", replace_on_first_fsync
    )

    with pytest.raises(RuntimeError, match="final_fence_failed"):
        publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())

    assert fsync_calls >= 2
    assert destination.read_bytes() == b"external publisher bytes\n"
    assert {path.name for path in root.iterdir()} == {destination.name}


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync publication path")
def test_acceptance_sink_rejects_inplace_rewrite_during_first_directory_fsync(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    attempt_id = str(uuid4())
    destination = root / f"{attempt_id}.regional-evidence.json"
    fsync_calls = 0

    def rewrite_on_first_fsync(path):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            assert path == root
            destination.write_bytes(b"in-place mutation\n")

    monkeypatch.setattr(
        acceptance_evidence, "_fsync_directory", rewrite_on_first_fsync
    )

    with pytest.raises(RuntimeError, match="final_fence_failed"):
        publish_completed_ai_regional_evidence(attempt_id, canonical_sidecar())

    assert fsync_calls >= 2
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="non-Windows Kernel32 absence")
def test_acceptance_sink_fails_closed_without_a_windows_held_handle(
    tmp_path, monkeypatch
):
    root = tmp_path / "acceptance"
    root.mkdir()
    monkeypatch.setenv("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", str(root))
    monkeypatch.setattr(acceptance_evidence, "_WINDOWS", True)

    with pytest.raises(RuntimeError, match="windows_held_handle_unavailable"):
        publish_completed_ai_regional_evidence(str(uuid4()), canonical_sidecar())

    assert list(root.iterdir()) == []
