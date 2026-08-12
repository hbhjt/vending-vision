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
COMPANION_BUILDER_SHA = "ebefe97377e4597ab62bc7ea6ab4849219df6fdd"
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
        isinstance(jobs, dict) and set(jobs) == {"companion_builder", "prove", "verify"},
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

    prove = _job_block(source, "prove")
    verify = _job_block(source, "verify")
    _require(prove.count("runs-on: windows-latest") == 1, "trusted_proof_prove_runner")
    _require(verify.count("runs-on: windows-latest") == 1, "trusted_proof_verify_runner")
    _require(source.count("actions/checkout@v4") == 2, "trusted_proof_checkout_count")
    for block, label in ((prove, "prove"), (verify, "verify")):
        for fragment in (
            "repository: ${{ job.workflow_repository }}",
            "ref: ${{ job.workflow_sha }}",
            "path: trusted-proof",
            "persist-credentials: false",
        ):
            _require(fragment in block, f"trusted_proof_{label}_checkout:{fragment}")

    companion_attestation = _step_index(
        prove,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{COMPANION_BUILDER_PATH}"',
        "companion_attestation",
    )
    companion_extract = _step_index(
        prove, "precutover_companion_descriptor.py).Path", "companion_archive_verify"
    )
    source_approval = _step_index(
        prove, "trusted-proof/scripts/approve_candidate_source.py", "source_approval"
    )
    tag_ruleset = _step_index(
        prove, "trusted-proof/scripts/verify_release_tag_ruleset.py", "tag_ruleset"
    )
    candidate_attestation = _step_index(
        prove,
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{CANDIDATE_BUILDER_PATH}"',
        "candidate_attestation",
    )
    execute = _step_index(
        prove, "vending-vision-precutover-verifier.exe).Path", "frozen_companion_execute"
    )
    proof_verify = _step_index(prove, "verify-proof --proof", "proof_binding_verify")
    attest = _step_index(prove, "actions/attest-build-provenance@v4", "proof_attestation")
    handoff = _step_index(prove, "name: Upload cross-job proof handoff", "proof_handoff")
    _require(
        companion_attestation
        < companion_extract
        < source_approval
        < tag_ruleset
        < candidate_attestation
        < execute
        < proof_verify
        < attest
        < handoff,
        "trusted_proof_prove_order",
    )
    _require(source.count("actions/attest-build-provenance@v4") == 1, "trusted_proof_attestation_count")
    _require("subject-path: ${{ env.TRUSTED_PROOF_PATH }}" in prove, "trusted_proof_subject")
    _require("& $companionExe --candidate-artifact" in prove, "trusted_proof_frozen_companion_command")
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
        "seal-evidence --directory proof-handoff",
        "+refs/heads/main:refs/remotes/origin/main",
        "--protected-main refs/remotes/origin/main",
        "rulesets?targets=tag&includes_parents=true&per_page=100",
        "--rulesets proof-tag-rulesets.json",
    ):
        _require(fragment in prove, f"trusted_proof_prove_policy:{fragment}")

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
