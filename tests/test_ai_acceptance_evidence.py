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
