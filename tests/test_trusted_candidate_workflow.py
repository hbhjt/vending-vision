from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"
PUBLISHER = ROOT / ".github" / "workflows" / "publish-candidate.yml"
TRUSTED_BUILDER_COMMIT = "fbb97d16f42b2c20a04831750c639fda6db1a3e9"
TRUSTED_SIGNER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml"
TRUSTED_SIGNER_COMMIT = "8b9f19da1fe07ba3e484f60317db6d14a5b447de"
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


def _check_policy(
    publisher: Path, *, signer: Path = TRUSTED_SIGNER
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TRUST_POLICY),
            "--builder",
            str(TRUSTED_BUILDER),
            "--signer",
            str(signer),
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


def test_publish_caller_pins_builder_a_and_signer_s_without_holding_supplier_secrets():
    completed = _check_policy(PUBLISHER)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    workflow = PUBLISHER.read_text("utf-8")
    literal_use = (
        "uses: hbhjt/vending-vision/.github/workflows/"
        f"trusted-ai-candidate-builder.yml@{TRUSTED_BUILDER_COMMIT}"
    )
    assert literal_use in workflow
    literal_signer = (
        "uses: hbhjt/vending-vision/.github/workflows/"
        f"trusted-ai-candidate-signer.yml@{TRUSTED_SIGNER_COMMIT}"
    )
    assert literal_signer in workflow
    assert f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"' in workflow
    assert "--signer-repo" not in workflow
    assert (
        '--signer-workflow "hbhjt/vending-vision/.github/workflows/'
        'trusted-ai-candidate-builder.yml"'
    ) in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "actions/attest-build-provenance" not in workflow
    assert "scripts/build_exe.ps1" not in workflow
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in workflow
    assert "generate_candidate_evidence.py" not in workflow
    assert "sign_candidate_evidence.py" not in workflow
    publish = workflow[workflow.index("  publish:\n"):]
    assert "trusted_signer" in publish.split("steps:", 1)[0]
    assert f"ref: {TRUSTED_SIGNER_COMMIT}" in publish
    assert publish.count("actions/download-artifact@v4") == 2
    assert "--target $env:RELEASE_TARGET" in publish
    assert "verify_release_tag_ruleset.py" in publish
    assert "rulesets?targets=tag" in publish


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
        "mutable-signer": trusted.replace(
            f"trusted-ai-candidate-signer.yml@{TRUSTED_SIGNER_COMMIT}",
            "trusted-ai-candidate-signer.yml@${{ github.sha }}",
        ),
        "raw-input-injection": trusted.replace(
            "run: |\n          if ($env:SOURCE_COMMIT",
            'run: |\n          Write-Output "${{ github.event.inputs.source_ref }}"\n          if ($env:SOURCE_COMMIT',
            1,
        ),
    }
    for name, source in mutations.items():
        candidate = tmp_path / f"{name}.yml"
        candidate.write_text(source, "utf-8")
        completed = _check_policy(candidate)
        assert completed.returncode != 0, name


def test_trust_policy_rejects_any_signer_byte_change_and_does_not_require_mutually_exclusive_flags(tmp_path):
    mutated_signer = tmp_path / "trusted-ai-candidate-signer.yml"
    mutated_signer.write_text(
        TRUSTED_SIGNER.read_text("utf-8") + "\n# caller-controlled change\n", "utf-8"
    )

    completed = _check_policy(PUBLISHER, signer=mutated_signer)

    assert completed.returncode != 0
    assert "trusted_signer_bytes_changed" in completed.stdout
    policy = TRUST_POLICY.read_text("utf-8")
    assert 'f\'--signer-repo "{TRUSTED_REPOSITORY}"\'' not in policy
    assert "failed to open local artifact" in policy
