from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"


def _workflow_call_inputs(source: str) -> set[str]:
    match = re.search(r"(?ms)^  workflow_call:\n    inputs:\n(?P<body>.*?)(?=^    outputs:)", source)
    assert match, "trusted builder must declare workflow_call inputs and outputs"
    return set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", match.group("body")))


def test_trusted_builder_has_a_closed_raw_material_interface_and_owns_attestation():
    workflow = TRUSTED_BUILDER.read_text("utf-8")

    assert _workflow_call_inputs(workflow) == {
        "source_commit",
        "core_wheelhouse_url",
        "core_wheelhouse_sha256",
        "core_wheelhouse_bytes",
    }
    assert "repository: ${{ job.workflow_repository }}" in workflow
    assert "ref: ${{ job.workflow_sha }}" in workflow
    assert "path: trusted-builder" in workflow
    assert "repository: hbhjt/vending-vision" in workflow
    assert "ref: ${{ inputs.source_commit }}" in workflow
    assert "path: source" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "id-token: write" in workflow
    assert "attestations: write" in workflow
    assert "secrets:" not in workflow
    assert "self-hosted" not in workflow
    for forbidden in ("artifact_path", "worker_path", "predicate", "custom_command", "command_input"):
        assert forbidden not in workflow

    verify = workflow.index("--require-ai-worker")
    attest = workflow.index("actions/attest-build-provenance@v4")
    upload = workflow.index("actions/upload-artifact@v4")
    assert verify < attest < upload
