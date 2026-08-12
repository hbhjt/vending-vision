"""Fail-closed static authority policy for the trusted Windows proof workflow."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from workflow_yaml import WorkflowYamlError, load_workflow_yaml, workflow_run_scalars


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
WORKFLOW_PATH = ".github/workflows/trusted-precutover-companion-proof.yml"
COMPANION_BUILDER_PATH = ".github/workflows/trusted-precutover-companion-builder.yml"
COMPANION_BUILDER_SHA = "154dfd47b55ba13a5a968447b9f175d45f9ab990"
CANDIDATE_BUILDER_PATH = ".github/workflows/trusted-ai-candidate-builder.yml"
CANDIDATE_BUILDER_SHA = "be8fe434855b94f61511e8c6c926e02c54230a38"
INPUTS = {
    f"{name}_{field}"
    for name in (
        "candidate_archive",
        "candidate_manifest",
        "candidate_attestation",
        "candidate_evidence",
        "model_pack",
    )
    for field in ("url", "sha256", "bytes")
}
OUTPUTS = {
    "artifact_name",
    "proof_sha256",
    "attestation_bundle_sha256",
    "companion_archive_sha256",
    "companion_descriptor_sha256",
    "source_commit",
}


class PolicyError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyError(message)


def _job_block(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
        source,
    )
    _require(match is not None, f"trusted_proof_job_missing:{name}")
    assert match is not None
    return match.group(0)


def _step_index(block: str, fragment: str, label: str) -> int:
    try:
        return block.index(fragment)
    except ValueError as exc:
        raise PolicyError(f"trusted_proof_step_missing:{label}") from exc


def _assert_gh_flags_parse(repository_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="trusted-proof-gh-") as temporary:
        completed = subprocess.run(
            [
                "gh",
                "attestation",
                "verify",
                str(Path(temporary) / "missing-proof.json"),
                "--bundle",
                str(Path(temporary) / "missing-proof.sigstore.json"),
                "--repo",
                TRUSTED_REPOSITORY,
                "--signer-workflow",
                f"{TRUSTED_REPOSITORY}/{WORKFLOW_PATH}",
                "--signer-digest",
                "0" * 40,
                "--source-ref",
                "refs/heads/main",
                "--source-digest",
                "0" * 40,
                "--deny-self-hosted-runners",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    output = (completed.stdout + completed.stderr).lower()
    _require(completed.returncode != 0, "trusted_proof_gh_missing_file_accepted")
    _require(
        "failed to open local artifact" in output or "no such file" in output,
        "trusted_proof_gh_flag_parse",
    )
    _require(
        "mutually exclusive" not in output and "cannot be used together" not in output,
        "trusted_proof_gh_flag_conflict",
    )


def check(workflow_path: Path, repository_root: Path) -> None:
    source = workflow_path.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    on = workflow.get("on")
    _require(isinstance(on, dict), "trusted_proof_on_shape")
    call = on.get("workflow_call")
    _require(isinstance(call, dict), "trusted_proof_workflow_call")
    inputs = call.get("inputs")
    _require(isinstance(inputs, dict) and set(inputs) == INPUTS, "trusted_proof_input_allowlist")
    for name, descriptor in inputs.items():
        _require(
            isinstance(descriptor, dict)
            and set(descriptor) == {"description", "required", "type"}
            and descriptor["required"] == "true"
            and descriptor["type"] == ("number" if name.endswith("_bytes") else "string"),
            f"trusted_proof_input_contract:{name}",
        )
    outputs = call.get("outputs")
    _require(
        isinstance(outputs, dict) and set(outputs) == OUTPUTS,
        "trusted_proof_output_allowlist",
    )
    jobs = workflow.get("jobs")
    _require(
        isinstance(jobs, dict)
        and set(jobs) == {"companion_builder", "execute", "sign", "verify"},
        "trusted_proof_job_set",
    )
    for run in workflow_run_scalars(source):
        _require("${{" not in run, "trusted_proof_workflow_expression_in_run")
    input_lines = [line.strip() for line in source.splitlines() if "${{ inputs." in line]
    _require(bool(input_lines), "trusted_proof_input_env_missing")
    _require(
        all(
            re.fullmatch(
                r"[A-Z][A-Z0-9_]*: \$\{\{ inputs\.[a-z][a-z0-9_]* \}\}", line
            )
            for line in input_lines
        ),
        "trusted_proof_input_not_step_env",
    )
    for forbidden in (
        "secrets:",
        "path: source",
        "ref: ${{ github.sha }}",
        "github.event.inputs",
        "candidate/scripts",
        "proof-input/scripts",
        "custom_command",
        "predicate",
        "worker_path",
        "artifact_path",
        "^refs/(heads|tags)/",
    ):
        _require(forbidden not in source, f"trusted_proof_forbidden:{forbidden}")

    builder = jobs["companion_builder"]
    expected_use = (
        f"{TRUSTED_REPOSITORY}/{COMPANION_BUILDER_PATH}@{COMPANION_BUILDER_SHA}"
    )
    _require(isinstance(builder, dict) and builder.get("uses") == expected_use, "trusted_proof_builder_pin")
    _require(
        isinstance(builder.get("with"), dict)
        and set(builder["with"])
        == {"core_wheelhouse_url", "core_wheelhouse_sha256", "core_wheelhouse_bytes"},
        "trusted_proof_builder_inputs",
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", COMPANION_BUILDER_SHA, "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    _require(ancestor.returncode == 0, "trusted_proof_builder_commit_not_in_history")
    committed = subprocess.run(
        ["git", "show", f"{COMPANION_BUILDER_SHA}:{COMPANION_BUILDER_PATH}"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    _require(committed.returncode == 0 and bool(committed.stdout), "trusted_proof_builder_commit_missing")

    execute = _job_block(source, "execute")
    sign = _job_block(source, "sign")
    verify = _job_block(source, "verify")
    _require(
        execute.count("runs-on: windows-latest") == 1,
        "trusted_proof_execute_runner",
    )
    _require(sign.count("runs-on: windows-latest") == 1, "trusted_proof_sign_runner")
    _require(verify.count("runs-on: windows-latest") == 1, "trusted_proof_verify_runner")
    _require(source.count("actions/checkout@v4") == 3, "trusted_proof_checkout_count")
    for block, label in (
        (execute, "execute"),
        (sign, "sign"),
        (verify, "verify"),
    ):
        for fragment in (
            "repository: ${{ job.workflow_repository }}",
            "ref: ${{ job.workflow_sha }}",
            "path: trusted-proof",
            "persist-credentials: false",
        ):
            _require(fragment in block, f"trusted_proof_{label}_checkout:{fragment}")

    execution_job = jobs["execute"]
    signing_job = jobs["sign"]
    _require(isinstance(execution_job, dict), "trusted_proof_execute_shape")
    _require(isinstance(signing_job, dict), "trusted_proof_sign_shape")
    execution_permissions = execution_job.get("permissions")
    signing_permissions = signing_job.get("permissions")
    _require(
        isinstance(execution_permissions, dict)
        and "id-token" not in execution_permissions
        and execution_permissions.get("attestations") != "write",
        "trusted_proof_execute_privilege",
    )
    _require(
        isinstance(signing_permissions, dict)
        and signing_permissions.get("id-token") == "write"
        and signing_permissions.get("attestations") == "write",
        "trusted_proof_sign_privilege",
    )
    _require(
        signing_job.get("needs") == ["companion_builder", "execute"],
        "trusted_proof_sign_needs_execute",
    )
    verify_job = jobs["verify"]
    _require(
        isinstance(verify_job, dict)
        and verify_job.get("needs") == ["companion_builder", "sign"],
        "trusted_proof_verify_needs_sign",
    )
    for name, expected_timeout in {"execute": 180, "sign": 180, "verify": 30}.items():
        _require(
            jobs[name].get("timeout-minutes") == str(expected_timeout)
            and _job_block(source, name).count(
                f"    timeout-minutes: {expected_timeout}\n"
            )
            == 1,
            f"trusted_proof_job_timeout:{name}",
        )
    download_commands = [
        line.strip()
        for run in workflow_run_scalars(source)
        for line in run.splitlines()
        if "$downloader --url" in line
    ]
    _require(len(download_commands) == 2, "trusted_proof_download_command_count")
    _require(
        all(
            re.search(r"--total-timeout-seconds 1800(?:\s|$)", line) is not None
            for line in download_commands
        ),
        "trusted_proof_download_total_timeout",
    )
    write_capable = [
        name
        for name, job in jobs.items()
        if isinstance(job, dict)
        and isinstance(job.get("permissions"), dict)
        and (
            job["permissions"].get("id-token") == "write"
            or job["permissions"].get("attestations") == "write"
        )
    ]
    _require(
        write_capable == ["companion_builder", "sign"],
        "trusted_proof_privileged_job_set",
    )

    companion_attestation = _step_index(
        execute,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{COMPANION_BUILDER_PATH}"',
        "companion_attestation",
    )
    companion_extract = _step_index(
        execute, "precutover_companion_descriptor.py).Path", "companion_archive_verify"
    )
    source_approval = _step_index(
        execute, "trusted-proof/scripts/approve_candidate_source.py", "source_approval"
    )
    tag_ruleset = _step_index(
        execute, "trusted-proof/scripts/verify_release_tag_ruleset.py", "tag_ruleset"
    )
    candidate_attestation = _step_index(
        execute,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{CANDIDATE_BUILDER_PATH}"',
        "candidate_attestation",
    )
    frozen_execute = _step_index(
        execute,
        "vending-vision-precutover-verifier.exe).Path",
        "frozen_companion_execute",
    )
    proof_verify = _step_index(execute, "verify-proof --proof", "proof_binding_verify")
    handoff_verify = _step_index(
        execute, "verify-execution-handoff", "execution_handoff_verify"
    )
    handoff = _step_index(
        execute, "name: Upload fixed exact execution handoff", "execution_handoff"
    )
    _require(
        companion_attestation
        < companion_extract
        < source_approval
        < tag_ruleset
        < candidate_attestation
        < frozen_execute
        < proof_verify
        < handoff_verify
        < handoff,
        "trusted_proof_execute_order",
    )
    _require(
        "actions/attest-build-provenance" not in execute,
        "trusted_proof_execute_self_attest",
    )
    _require(
        "id-token: write" not in execute and "attestations: write" not in execute,
        "trusted_proof_execute_write_permission",
    )
    _require(
        "& $companionExe --candidate-artifact" in execute,
        "trusted_proof_frozen_companion_command",
    )
    for fragment in (
        "verify-execution-handoff --directory execution-handoff",
        "path: execution-handoff/precutover-ai-proof.json",
        "if-no-files-found: error",
    ):
        _require(fragment in execute, f"trusted_proof_execution_handoff:{fragment}")
    for fragment in (
        f'--signer-digest "{COMPANION_BUILDER_SHA}"',
        f'--signer-digest "{CANDIDATE_BUILDER_SHA}"',
        "--source-ref $env:CALLER_REF",
        "--source-digest $env:CALLER_SHA",
        "--source-digest $env:PROOF_SOURCE_COMMIT",
        "--deny-self-hosted-runners",
        "scripts/download_verified_file.py",
        "inspect-inputs --input-root proof-input",
        "verify-proof --proof precutover-ai-proof.json",
        "+refs/heads/main:refs/remotes/origin/main",
        "--protected-main refs/remotes/origin/main",
        "rulesets?targets=tag&includes_parents=true&per_page=100",
        "--rulesets proof-tag-rulesets.json",
    ):
        _require(fragment in execute, f"trusted_proof_execute_policy:{fragment}")

    sign_companion = _step_index(
        sign,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{COMPANION_BUILDER_PATH}"',
        "sign_companion_attestation",
    )
    sign_download = _step_index(
        sign, "scripts/download_verified_file.py", "sign_fresh_download"
    )
    sign_inspect = _step_index(sign, "inspect-inputs", "sign_fresh_identity")
    sign_source = _step_index(
        sign, "trusted-proof/scripts/approve_candidate_source.py", "sign_source_approval"
    )
    sign_ruleset = _step_index(
        sign, "trusted-proof/scripts/verify_release_tag_ruleset.py", "sign_tag_ruleset"
    )
    sign_candidate = _step_index(
        sign,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{CANDIDATE_BUILDER_PATH}"',
        "sign_candidate_attestation",
    )
    sign_download_handoff = _step_index(
        sign, "name: Download fixed execution handoff", "sign_download_handoff"
    )
    sign_revalidate = _step_index(
        sign, "verify-execution-handoff", "sign_revalidate_handoff"
    )
    sign_attest = _step_index(
        sign, "actions/attest-build-provenance@v4", "sign_proof_attestation"
    )
    sign_seal = _step_index(sign, "seal-evidence", "sign_seal_evidence")
    sign_upload = _step_index(
        sign, "name: Upload signed proof handoff", "sign_upload_handoff"
    )
    _require(
        sign_companion
        < sign_download
        < sign_inspect
        < sign_source
        < sign_ruleset
        < sign_candidate
        < sign_download_handoff
        < sign_revalidate
        < sign_attest
        < sign_seal
        < sign_upload,
        "trusted_proof_sign_order",
    )
    _require(
        source.count("actions/attest-build-provenance@v4") == 1,
        "trusted_proof_attestation_count",
    )
    _require(
        "subject-path: ${{ env.TRUSTED_PROOF_PATH }}" in sign,
        "trusted_proof_subject",
    )
    for forbidden in (
        "& $companionExe",
        "--candidate-artifact",
        "path: source",
        "candidate/scripts",
        "Get-Command",
    ):
        _require(forbidden not in sign, f"trusted_proof_sign_execution:{forbidden}")
    for step in signing_job.get("steps", []):
        if not isinstance(step, dict) or not isinstance(step.get("run"), str):
            continue
        for line in step["run"].splitlines():
            command = line.strip()
            _require(
                re.match(r"(?i)^(?:&\s*)?(?:gh|git|python(?:\.exe)?)\b", command)
                is None,
                "trusted_proof_sign_path_command",
            )
            if command.startswith("& "):
                _require(
                    re.match(
                        r"^& \$env:TRUSTED_(?:PYTHON|GH|GIT)(?:\s|$)", command
                    )
                    is not None,
                    "trusted_proof_sign_untrusted_call_operator",
                )
                if command.startswith("& $env:TRUSTED_PYTHON "):
                    _require(
                        re.match(
                            r"^& \$env:TRUSTED_PYTHON \$(?:verifier|downloader|proofTool|sourceApproval|tagPolicy)(?:\s|$)",
                            command,
                        )
                        is not None,
                        "trusted_proof_sign_untrusted_python_script",
                    )
    for assignment in (
        "$verifier = (Resolve-Path -LiteralPath trusted-proof/scripts/precutover_companion_descriptor.py).Path",
        "$downloader = (Resolve-Path -LiteralPath trusted-proof/scripts/download_verified_file.py).Path",
        "$proofTool = (Resolve-Path -LiteralPath trusted-proof/scripts/trusted_precutover_proof.py).Path",
        "$sourceApproval = (Resolve-Path -LiteralPath trusted-proof/scripts/approve_candidate_source.py).Path",
        "$tagPolicy = (Resolve-Path -LiteralPath trusted-proof/scripts/verify_release_tag_ruleset.py).Path",
    ):
        _require(assignment in sign, f"trusted_proof_sign_trusted_script:{assignment}")
    for fragment in (
        'C:\\Program Files\\GitHub CLI\\gh.exe',
        'C:\\Program Files\\Git\\cmd\\git.exe',
        'Join-Path $env:pythonLocation "python.exe"',
        "& $env:TRUSTED_GH attestation verify",
        "& $env:TRUSTED_GIT --git-dir",
        "& $env:TRUSTED_PYTHON $proofTool verify-execution-handoff",
        "inspect-inputs --input-root signer-proof-input",
        "+refs/heads/main:refs/remotes/origin/main",
        "--protected-main refs/remotes/origin/main",
        "rulesets?targets=tag&includes_parents=true&per_page=100",
    ):
        _require(fragment in sign, f"trusted_proof_sign_policy:{fragment}")

    evidence = _step_index(verify, "verify-evidence --directory proof-handoff", "fresh_evidence")
    gh_verify = _step_index(verify, "gh attestation verify", "fresh_attestation")
    final_upload = _step_index(verify, "name: Upload fresh-runner verified proof", "final_upload")
    _require(evidence < gh_verify < final_upload, "trusted_proof_fresh_verify_order")
    for fragment in (
        f'--repo "{TRUSTED_REPOSITORY}"',
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{WORKFLOW_PATH}"',
        "--signer-digest $env:TRUSTED_WORKFLOW_SHA",
        "--source-ref $env:CALLER_REF",
        "--source-digest $env:CALLER_SHA",
        "--deny-self-hosted-runners",
        "path: proof-handoff/*",
    ):
        _require(fragment in verify, f"trusted_proof_verify_policy:{fragment}")
    _require("actions/attest-build-provenance" not in verify, "trusted_proof_verify_self_attest")
    _assert_gh_flags_parse(repository_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        check(args.workflow.resolve(), args.repository_root.resolve())
    except (OSError, PolicyError, WorkflowYamlError) as exc:
        print(f"TRUSTED_PRECUTOVER_PROOF_POLICY=FAIL:{exc}")
        return 1
    print("TRUSTED_PRECUTOVER_PROOF_POLICY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
