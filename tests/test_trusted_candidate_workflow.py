from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"
PUBLISHER = ROOT / ".github" / "workflows" / "publish-candidate.yml"
TRUSTED_BUILDER_COMMIT = "fbb97d16f42b2c20a04831750c639fda6db1a3e9"
TRUST_POLICY = ROOT / "scripts" / "check_trusted_candidate_workflows.py"


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
    assert '"${{ inputs.source_commit }}" -cne "${{ github.sha }}"' in workflow
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


def _check_policy(publisher: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TRUST_POLICY),
            "--builder",
            str(TRUSTED_BUILDER),
            "--publisher",
            str(publisher),
            "--repository-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_publish_caller_is_pinned_to_commit_a_and_verifies_its_signer_identity():
    completed = _check_policy(PUBLISHER)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    workflow = PUBLISHER.read_text("utf-8")
    literal_use = (
        "uses: hbhjt/vending-vision/.github/workflows/"
        f"trusted-ai-candidate-builder.yml@{TRUSTED_BUILDER_COMMIT}"
    )
    assert literal_use in workflow
    assert f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"' in workflow
    assert '--signer-repo "hbhjt/vending-vision"' in workflow
    assert (
        '--signer-workflow "hbhjt/vending-vision/.github/workflows/'
        'trusted-ai-candidate-builder.yml"'
    ) in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "actions/attest-build-provenance" not in workflow
    assert "scripts/build_exe.ps1" not in workflow
    secret_step = workflow[workflow.index("      - name: Sign installed evidence"):]
    secret_step = secret_step[: secret_step.index("      - name: Publish immutable prerelease")]
    assert "trusted-evidence/scripts/sign_candidate_evidence.py" in secret_step
    assert "source/scripts" not in secret_step


def test_trust_policy_rejects_mutable_caller_and_missing_or_wrong_signer_digest(tmp_path):
    trusted = PUBLISHER.read_text("utf-8")
    mutations = {
        "mutable-use": trusted.replace(
            f"trusted-ai-candidate-builder.yml@{TRUSTED_BUILDER_COMMIT}",
            "trusted-ai-candidate-builder.yml@${{ github.sha }}",
        ),
        "missing-signer": trusted.replace(
            f' --signer-digest "{TRUSTED_BUILDER_COMMIT}"', ""
        ),
        "wrong-signer": trusted.replace(
            f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"',
            '--signer-digest "' + "0" * 40 + '"',
        ),
        "caller-attest": trusted + "\n# actions/attest-build-provenance@v4\n",
    }
    for name, source in mutations.items():
        candidate = tmp_path / f"{name}.yml"
        candidate.write_text(source, "utf-8")
        completed = _check_policy(candidate)
        assert completed.returncode != 0, name
