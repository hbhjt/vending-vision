import json
from pathlib import Path
from uuid import uuid4

import pytest

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
