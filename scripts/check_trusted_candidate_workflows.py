from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

from workflow_yaml import WorkflowYamlError, workflow_run_scalars


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
TRUSTED_BUILDER_COMMIT = "3fe9e00c98d9df59c71ce9be5b980a713ddd3110"
TRUSTED_BUILDER_PATH = ".github/workflows/trusted-ai-candidate-builder.yml"
TRUSTED_SIGNER_COMMIT = "af9f7bb766e8a467e8c9a24396a76b616fd68188"
TRUSTED_SIGNER_PATH = ".github/workflows/trusted-ai-candidate-signer.yml"
HOSTED_AUTHORITY_COMMIT = "41afbd9bd07b67df9f93de1dea1a9f9b0cea0228"
HOSTED_AUTHORITY_PATH = "scripts/verify_hosted_release_authority.py"
BUILDER_INPUTS = {
    "source_commit",
    "core_wheelhouse_url",
    "core_wheelhouse_sha256",
    "core_wheelhouse_bytes",
}
SIGNER_INPUTS = {
    "source_commit",
    "source_ref",
    "artifact_name",
    "subject_sha256",
    "manifest_sha256",
    "attestation_bundle_sha256",
}
TRUSTED_SIGNER_FILES = {
    TRUSTED_SIGNER_PATH,
    "scripts/approve_candidate_source.py",
    "scripts/candidate_artifact_manifest.py",
    "scripts/evidence_artifact.py",
    "scripts/generate_trusted_candidate_evidence.py",
    "scripts/sign_candidate_evidence.py",
    "scripts/verify_release_tag_ruleset.py",
    "scripts/verify_trusted_candidate_inputs.py",
    "scripts/verify_trusted_script_set.py",
    "trusted-signer-scripts.json",
}


class TrustPolicyError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrustPolicyError(message)


def _workflow_call_inputs(source: str) -> set[str]:
    match = re.search(r"(?ms)^  workflow_call:\n    inputs:\n(?P<body>.*?)(?=^    outputs:)", source)
    _require(match is not None, "trusted_builder_workflow_call_shape")
    assert match is not None
    return set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", match.group("body")))


def _job_block(source: str, job: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(job)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", source)
    _require(match is not None, f"publisher_job_missing:{job}")
    assert match is not None
    return match.group(0)


def _assert_no_untrusted_run_expressions(source: str, label: str) -> None:
    blocks = workflow_run_scalars(source)
    _require(bool(blocks), f"{label}_run_blocks_missing")
    for block in blocks:
        _require("${{" not in block, f"{label}_workflow_expression_in_run")


def _assert_files_are_immutable(
    *, commit: str, paths: dict[str, Path], repository_root: Path, label: str
) -> None:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    _require(ancestor.returncode == 0, f"{label}_commit_not_in_history")
    for relative, path in paths.items():
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        _require(committed.returncode == 0, f"{label}_commit_file_missing:{relative}")
        _require(path.read_bytes() == committed.stdout, f"{label}_bytes_changed:{relative}")


def _assert_gh_attestation_flags_parse(repository_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="trusted-gh-policy-") as temporary:
        missing = Path(temporary) / "missing-candidate.zip"
        completed = subprocess.run(
            [
                "gh", "attestation", "verify", str(missing),
                "--bundle", str(Path(temporary) / "missing-bundle.json"),
                "--repo", TRUSTED_REPOSITORY,
                "--signer-workflow", f"{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}",
                "--signer-digest", TRUSTED_BUILDER_COMMIT,
                "--source-ref", "refs/tags/v0.0.0-rc.0",
                "--source-digest", "0" * 40,
                "--deny-self-hosted-runners",
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    output = (completed.stdout + completed.stderr).lower()
    _require(completed.returncode != 0, "gh_missing_fixture_unexpectedly_verified")
    _require("failed to open local artifact" in output or "no such file" in output, "gh_policy_parse_failure_reason")
    _require("mutually exclusive" not in output and "cannot be used together" not in output, "gh_policy_mutually_exclusive_flags")


def check_trusted_candidate_workflows(
    *, builder: Path, signer: Path, publisher: Path, repository_root: Path
) -> None:
    for name, commit in (
        ("trusted_builder", TRUSTED_BUILDER_COMMIT),
        ("trusted_signer", TRUSTED_SIGNER_COMMIT),
        ("hosted_authority", HOSTED_AUTHORITY_COMMIT),
    ):
        _require(
            re.fullmatch(r"[a-f0-9]{40}", commit) is not None,
            f"{name}_commit_invalid",
        )
    builder_source = builder.read_text("utf-8")
    signer_source = signer.read_text("utf-8")
    publisher_source = publisher.read_text("utf-8")
    _assert_no_untrusted_run_expressions(builder_source, "trusted_builder")
    _assert_no_untrusted_run_expressions(signer_source, "trusted_signer")
    _assert_no_untrusted_run_expressions(publisher_source, "publisher")
    _assert_files_are_immutable(
        commit=TRUSTED_BUILDER_COMMIT,
        paths={TRUSTED_BUILDER_PATH: builder},
        repository_root=repository_root,
        label="trusted_builder",
    )
    _assert_files_are_immutable(
        commit=TRUSTED_SIGNER_COMMIT,
        paths={
            relative: signer if relative == TRUSTED_SIGNER_PATH else repository_root / relative
            for relative in TRUSTED_SIGNER_FILES
        },
        repository_root=repository_root,
        label="trusted_signer",
    )
    _assert_files_are_immutable(
        commit=HOSTED_AUTHORITY_COMMIT,
        paths={HOSTED_AUTHORITY_PATH: repository_root / HOSTED_AUTHORITY_PATH},
        repository_root=repository_root,
        label="hosted_authority",
    )
    _require(_workflow_call_inputs(builder_source) == BUILDER_INPUTS, "trusted_builder_input_allowlist")
    for forbidden in ("secrets:", "artifact_path", "worker_path", "predicate", "custom_command"):
        _require(forbidden not in builder_source, f"trusted_builder_forbidden_input:{forbidden}")

    _require(_workflow_call_inputs(signer_source) == SIGNER_INPUTS, "trusted_signer_input_allowlist")
    for forbidden in (
        "source_path", "artifact_path", "custom_command", "predicate", "private_key",
    ):
        _require(forbidden not in signer_source, f"trusted_signer_forbidden_input:{forbidden}")
    verify_evidence = _job_block(signer_source, "verify_evidence")
    sign_evidence = _job_block(signer_source, "sign_evidence")
    for fragment in (
        "runs-on: windows-latest",
        "repository: ${{ job.workflow_repository }}",
        "ref: ${{ job.workflow_sha }}",
        "path: trusted-signer",
        "actions/download-artifact@v4",
        "gh attestation verify",
        f'--repo "{TRUSTED_REPOSITORY}"',
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}"',
        f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"',
        "--source-ref $env:SOURCE_REF",
        "--source-digest $env:SOURCE_COMMIT",
        "--deny-self-hosted-runners",
        "https://github.com/hbhjt/vending-vision.git",
        "+refs/heads/main:refs/remotes/origin/main",
        "trusted-signer/scripts/approve_candidate_source.py",
        "--protected-main refs/remotes/origin/main",
        "trusted-signer/scripts/verify_trusted_candidate_inputs.py",
        "trusted-signer/scripts/generate_trusted_candidate_evidence.py",
        "trusted-signer/scripts/evidence_artifact.py",
    ):
        _require(fragment in verify_evidence, f"trusted_signer_verify_policy:{fragment}")
    _require("environment:" not in verify_evidence, "trusted_signer_verify_environment")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in verify_evidence, "trusted_signer_verify_secret")
    _require("--signer-repo" not in verify_evidence, "trusted_signer_mutually_exclusive_repo_flags")

    for fragment in (
        "needs: verify_evidence",
        "runs-on: windows-latest",
        "environment: experimental-candidate",
        "repository: ${{ job.workflow_repository }}",
        "ref: ${{ job.workflow_sha }}",
        "path: trusted-signer",
        "trusted-signer/scripts/verify_trusted_script_set.py",
        "trusted-signer/scripts/evidence_artifact.py",
        "--expected-digest $env:UNSIGNED_EVIDENCE_SHA256",
        "trusted-signer/scripts/sign_candidate_evidence.py",
        "--openssl $env:TRUSTED_OPENSSL",
        "VISION_SUPPLIER_PRIVATE_KEY_PEM",
    ):
        _require(fragment in sign_evidence, f"trusted_signer_sign_policy:{fragment}")
    _require(signer_source.count("actions/checkout@v4") == 2, "trusted_signer_checkout_count")
    for forbidden in ("path: source", "actions/setup-python", "source/scripts", ".spec"):
        _require(forbidden not in signer_source, f"trusted_signer_candidate_execution:{forbidden}")
    for forbidden in ("candidate-input", "verified-candidate", "source-approval", ".venv"):
        _require(forbidden not in sign_evidence, f"trusted_signer_cross_job_leak:{forbidden}")
    secret_step = re.search(r"(?ms)^      - name: Sign only verified evidence.*?(?=^      - name:)", sign_evidence)
    _require(secret_step is not None, "trusted_signer_secret_step_missing")
    assert secret_step is not None
    _require("trusted-signer/scripts/sign_candidate_evidence.py" in secret_step.group(0), "trusted_signer_secret_script")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in sign_evidence[: secret_step.start()], "trusted_signer_secret_exposed_early")

    trusted_call = (
        f"uses: {TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}@{TRUSTED_BUILDER_COMMIT}"
    )
    _require(publisher_source.count(trusted_call) == 1, "publisher_literal_builder_pin")
    trusted_job = _job_block(publisher_source, "trusted_builder")
    _require(trusted_call in trusted_job, "publisher_builder_job_pin")
    _require("${{" not in next(line for line in trusted_job.splitlines() if "uses:" in line), "publisher_mutable_builder_pin")
    with_match = re.search(r"(?ms)^    with:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|^    [a-zA-Z0-9_-]+:|\Z)", trusted_job)
    _require(with_match is not None, "publisher_builder_inputs_missing")
    assert with_match is not None
    caller_inputs = set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):", with_match.group("body")))
    _require(caller_inputs == BUILDER_INPUTS, "publisher_builder_input_allowlist")

    for forbidden in (
        "actions/attest-build-provenance",
        "scripts/build_exe.ps1",
        "scripts/candidate_artifact_manifest.py",
    ):
        _require(forbidden not in publisher_source, f"publisher_owns_trusted_step:{forbidden}")

    verify = _job_block(publisher_source, "verify")
    required_verifier_fragments = (
        "needs: trusted_builder",
        "runs-on: windows-latest",
        "actions/download-artifact@v4",
        "gh attestation verify",
        f'--repo "{TRUSTED_REPOSITORY}"',
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}"',
        f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"',
        "--source-digest $env:SOURCE_COMMIT",
        "--source-ref $env:SOURCE_REF",
        "--deny-self-hosted-runners",
        "--require-ai-worker",
    )
    for fragment in required_verifier_fragments:
        _require(fragment in verify, f"publisher_verify_policy:{fragment}")
    _require("--signer-repo" not in verify, "publisher_verify_mutually_exclusive_repo_flags")
    _require(verify.index("gh attestation verify") < verify.index("--require-ai-worker"), "publisher_attestation_before_probe")

    trusted_signer_call = (
        f"uses: {TRUSTED_REPOSITORY}/{TRUSTED_SIGNER_PATH}@{TRUSTED_SIGNER_COMMIT}"
    )
    _require(publisher_source.count(trusted_signer_call) == 1, "publisher_literal_signer_pin")
    signer_call = _job_block(publisher_source, "trusted_signer")
    _require("needs: verify" in signer_call, "publisher_signer_requires_verify")
    _require(trusted_signer_call in signer_call, "publisher_signer_job_pin")
    _require("${{" not in next(line for line in signer_call.splitlines() if "uses:" in line), "publisher_mutable_signer_pin")
    signer_with = re.search(r"(?ms)^    with:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)", signer_call)
    _require(signer_with is not None, "publisher_signer_inputs_missing")
    assert signer_with is not None
    signer_caller_inputs = set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):", signer_with.group("body")))
    _require(signer_caller_inputs == SIGNER_INPUTS, "publisher_signer_input_allowlist")

    publish = _job_block(publisher_source, "publish")
    _require("needs: [verify, trusted_signer]" in publish, "publisher_requires_trusted_signer")
    _require(publish.count("actions/download-artifact@v4") == 2, "publisher_downloads_candidate_and_evidence")
    for fragment in (
        f"ref: {TRUSTED_SIGNER_COMMIT}",
        "path: trusted-policy",
        "trusted-policy/scripts/verify_trusted_script_set.py",
        "trusted-policy/scripts/evidence_artifact.py",
        "--expected-digest $env:SIGNED_EVIDENCE_SHA256",
        "supplier.attestedSubjectDigest",
        "supplier.approvedSourceCommit",
        "descriptor.sourceApproval.commit",
        "git init --bare release-authority.git",
        "+refs/heads/main:refs/remotes/origin/main",
        "trusted-policy/scripts/approve_candidate_source.py",
        f"ref: {HOSTED_AUTHORITY_COMMIT}",
        "path: hosted-authority",
        "hosted-authority/scripts/verify_hosted_release_authority.py",
        "--mode publish-admission",
        "--mode publish-complete",
        "environment: experimental-candidate",
        "gh release create $env:RELEASE_TAG",
        "--target $env:RELEASE_TARGET",
        "--verify-tag",
    ):
        _require(fragment in publish, f"publisher_release_authority:{fragment}")
    _require(publish.count("actions/checkout@v4") == 2, "publisher_trusted_policy_checkout")
    for forbidden in (
        "VISION_SUPPLIER_PRIVATE_KEY_PEM",
        "generate_candidate_evidence.py", "sign_candidate_evidence.py",
    ):
        _require(forbidden not in publish, f"publisher_forbidden_capability:{forbidden}")
    _require("rulesets?targets=tag" not in publish, "publisher_unavailable_rulesets_api")
    _require(publish.count("environment: experimental-candidate") == 1, "publisher_environment_authority")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in publisher_source, "publisher_supplier_key_present")
    _assert_gh_attestation_flags_parse(repository_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", required=True, type=Path)
    parser.add_argument("--signer", required=True, type=Path)
    parser.add_argument("--publisher", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        check_trusted_candidate_workflows(
            builder=args.builder.resolve(),
            signer=args.signer.resolve(),
            publisher=args.publisher.resolve(),
            repository_root=args.repository_root.resolve(),
        )
    except (OSError, StopIteration, TrustPolicyError, WorkflowYamlError) as exc:
        print(f"TRUSTED_CANDIDATE_WORKFLOW_POLICY=FAIL:{exc}")
        return 1
    print("TRUSTED_CANDIDATE_WORKFLOW_POLICY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
