from __future__ import annotations

from pathlib import Path
import re
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"
PUBLISHER = ROOT / ".github" / "workflows" / "publish-candidate.yml"
TRUSTED_BUILDER_COMMIT = "c90a965d117fea49f318b18e0fcd50aa047bc41f"
TRUSTED_SIGNER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml"
TRUSTED_SIGNER_COMMIT = "fbb43d10bd65d477133d0005471a42b765ae39a5"
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
    input_lines = [line.strip() for line in workflow.splitlines() if "${{ inputs." in line]
    assert input_lines
    assert all(
        line.startswith("ref: ")
        or re.fullmatch(r"[A-Z][A-Z0-9_]*: \$\{\{ inputs\.[a-z][a-z0-9_]* \}\}", line)
        for line in input_lines
    )
    assert "$env:SOURCE_COMMIT" in workflow
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
    publisher: Path, *, builder: Path = TRUSTED_BUILDER, signer: Path = TRUSTED_SIGNER
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TRUST_POLICY),
            "--builder",
            str(builder),
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
    assert "verify_hosted_release_authority.py" in publish
    assert "--mode publish-admission" in publish
    assert "--mode publish-complete" in publish
    assert "rulesets?targets=tag" not in publish


def test_trust_policy_rejects_truncated_immutable_workflow_identity(tmp_path):
    truncated_policy = tmp_path / "check_trusted_candidate_workflows.py"
    truncated_policy.write_text(
        TRUST_POLICY.read_text("utf-8").replace(
            TRUSTED_BUILDER_COMMIT, TRUSTED_BUILDER_COMMIT[:-1], 1
        ),
        "utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(truncated_policy),
            "--builder", str(TRUSTED_BUILDER),
            "--signer", str(TRUSTED_SIGNER),
            "--publisher", str(PUBLISHER),
            "--repository-root", str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )

    assert completed.returncode != 0
    assert "trusted_builder_commit_invalid" in completed.stdout


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
        "raw-needs-injection": trusted.replace(
            "run: |\n          if ($env:SOURCE_COMMIT",
            'run: |\n          Write-Output "${{ needs.verify.outputs.subject_sha256 }}"\n          if ($env:SOURCE_COMMIT',
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


def test_policy_rejects_builder_source_input_shell_interpolation_without_execution(tmp_path):
    marker = tmp_path / "builder-injection-ran"
    mutated_builder = tmp_path / "trusted-ai-candidate-builder.yml"
    mutated_builder.write_text(
        TRUSTED_BUILDER.read_text("utf-8").replace(
            "run: |\n          if ($env:TRUSTED_WORKFLOW_REPOSITORY",
            'run: |\n          Write-Output "${{ inputs.source_commit }}"\n          if ($env:TRUSTED_WORKFLOW_REPOSITORY',
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(PUBLISHER, builder=mutated_builder)

    assert completed.returncode != 0
    assert "trusted_builder_workflow_expression_in_run" in completed.stdout
    assert not marker.exists()


def test_policy_rejects_publisher_needs_output_shell_injection_without_execution(tmp_path):
    marker = tmp_path / "publisher-injection-ran"
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "run: |\n          if ($env:SOURCE_COMMIT",
            'run: |\n          Write-Output "${{ needs.verify.outputs.artifact_name }}"; '
            f'New-Item -ItemType File -Path "{marker}"\n'
            "          if ($env:SOURCE_COMMIT",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout
    assert not marker.exists()


def test_policy_rejects_chomped_literal_run_scalar_injection(tmp_path):
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8")
        + '\n      - name: Chomped literal injection\n'
          '        shell: pwsh\n'
          '        run: |-\n'
          '          Write-Output "${{ needs.verify.outputs.subject_sha256 }}"\n',
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout


@pytest.mark.parametrize(
    ("target", "scalar", "expression", "expected_error"),
    [
        (
            "builder",
            "|",
            "${{inputs.source_commit}}",
            "trusted_builder_workflow_expression_in_run",
        ),
        (
            "signer",
            "|-",
            "${{  github.event.inputs.source_ref }}",
            "trusted_signer_workflow_expression_in_run",
        ),
        (
            "publisher",
            "|+",
            "${{\tneeds.verify.outputs.subject_sha256\t}}",
            "publisher_workflow_expression_in_run",
        ),
        (
            "builder",
            ">",
            "${{\ninputs.source_commit\n}}",
            "trusted_builder_workflow_expression_in_run",
        ),
        (
            "signer",
            ">-",
            "${{ github['event'].inputs['source_ref'] }}",
            "trusted_signer_workflow_expression_in_run",
        ),
        (
            "publisher",
            ">+",
            "${{ needs['verify']['outputs']['subject_sha256'] }}",
            "publisher_workflow_expression_in_run",
        ),
        (
            "publisher",
            "inline",
            "${{needs.x}}",
            "publisher_workflow_expression_in_run",
        ),
    ],
)
def test_policy_rejects_untrusted_expressions_in_every_legal_run_scalar_form(
    tmp_path, target, scalar, expression, expected_error
):
    sources = {
        "builder": TRUSTED_BUILDER.read_text("utf-8"),
        "signer": TRUSTED_SIGNER.read_text("utf-8"),
        "publisher": PUBLISHER.read_text("utf-8"),
    }
    if scalar == "inline":
        injected = f"        run: 'Write-Output \"{expression}\"'\n"
    else:
        command = f'Write-Output "{expression}"'
        body = "\n".join(f"          {line}" for line in command.splitlines())
        injected = f"        run: {scalar}\n{body}\n"
    mutated = tmp_path / f"{target}-{scalar.replace('|', 'literal').replace('>', 'folded')}.yml"
    mutated.write_text(
        sources[target]
        + "\n      - name: AST policy injection probe\n"
          "        shell: pwsh\n"
        + injected,
        "utf-8",
    )

    kwargs = {target: mutated} if target in {"builder", "signer"} else {}
    completed = _check_policy(mutated if target == "publisher" else PUBLISHER, **kwargs)

    assert completed.returncode != 0
    assert expected_error in completed.stdout


def test_policy_enumerates_nested_steps_across_multiple_jobs(tmp_path):
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8")
        + "\n  nested_policy_probe:\n"
          "    runs-on: windows-latest\n"
          "    steps:\n"
          "      - name: Nested folded injection\n"
          "        shell: pwsh\n"
          "        run: >-\n"
          '          Write-Output "${{ github.event.inputs.source_ref }}"\n',
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout


def test_policy_allows_workflow_expressions_in_non_run_env_and_with_fields(tmp_path):
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8")
        + "\n      - name: Allowed expression transport\n"
          "        shell: pwsh\n"
          "        env:\n"
          "          FROM_INPUT: ${{ inputs.source_commit }}\n"
          "          FROM_NEEDS: ${{ needs.verify.outputs.subject_sha256 }}\n"
          "        run: Write-Output $env:FROM_INPUT\n"
          "      - name: Allowed action input transport\n"
          "        uses: actions/github-script@v7\n"
          "        with:\n"
          "          script: ${{ github.event.inputs.source_ref }}\n",
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_policy_rejects_whitespace_before_needs_root_context(tmp_path):
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8")
        + "\n      - name: Whitespace expression injection\n"
          "        shell: pwsh\n"
          "        run: 'Write-Output \"${{  needs.verify.outputs.x }}\"'\n",
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout


@pytest.mark.parametrize(
    "expression",
    (
        "${{ runner.os }}",
        "${{ this expression is intentionally unterminated",
    ),
)
def test_policy_rejects_every_workflow_expression_prefix_in_run(tmp_path, expression):
    mutated_publisher = tmp_path / "publish-candidate.yml"
    mutated_publisher.write_text(
        PUBLISHER.read_text("utf-8")
        + "\n      - name: Conservative expression policy probe\n"
          "        shell: pwsh\n"
          f"        run: 'Write-Output \"{expression}\"'\n",
        "utf-8",
    )

    completed = _check_policy(mutated_publisher)

    assert completed.returncode != 0
    assert "publisher_workflow_expression_in_run" in completed.stdout
