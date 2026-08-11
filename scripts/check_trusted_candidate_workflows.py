from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
TRUSTED_BUILDER_COMMIT = "fbb97d16f42b2c20a04831750c639fda6db1a3e9"
TRUSTED_BUILDER_PATH = ".github/workflows/trusted-ai-candidate-builder.yml"
ALLOWED_INPUTS = {
    "source_commit",
    "core_wheelhouse_url",
    "core_wheelhouse_sha256",
    "core_wheelhouse_bytes",
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


def _assert_builder_is_commit_a(builder: Path, repository_root: Path) -> None:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TRUSTED_BUILDER_COMMIT, "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    _require(ancestor.returncode == 0, "trusted_builder_commit_not_in_history")
    committed = subprocess.run(
        ["git", "show", f"{TRUSTED_BUILDER_COMMIT}:{TRUSTED_BUILDER_PATH}"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    _require(committed.returncode == 0, "trusted_builder_commit_file_missing")
    _require(builder.read_bytes() == committed.stdout, "trusted_builder_bytes_changed_after_commit_a")


def check_trusted_candidate_workflows(
    *, builder: Path, publisher: Path, repository_root: Path
) -> None:
    builder_source = builder.read_text("utf-8")
    publisher_source = publisher.read_text("utf-8")
    _assert_builder_is_commit_a(builder, repository_root)
    _require(_workflow_call_inputs(builder_source) == ALLOWED_INPUTS, "trusted_builder_input_allowlist")
    for forbidden in ("secrets:", "artifact_path", "worker_path", "predicate", "custom_command"):
        _require(forbidden not in builder_source, f"trusted_builder_forbidden_input:{forbidden}")

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
    _require(caller_inputs == ALLOWED_INPUTS, "publisher_builder_input_allowlist")

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
        f'--signer-repo "{TRUSTED_REPOSITORY}"',
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}"',
        f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"',
        '--source-digest "${{ github.sha }}"',
        '--source-ref "${{ github.ref }}"',
        "--deny-self-hosted-runners",
        "--require-ai-worker",
    )
    for fragment in required_verifier_fragments:
        _require(fragment in verify, f"publisher_verify_policy:{fragment}")
    _require(verify.index("gh attestation verify") < verify.index("--require-ai-worker"), "publisher_attestation_before_probe")

    publish = _job_block(publisher_source, "publish")
    _require("needs: verify" in publish, "publisher_requires_fresh_verify")
    _require("environment: experimental-candidate" in publish, "publisher_secret_environment")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" in publish, "publisher_supplier_key_missing")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in publisher_source[: publisher_source.index("  publish:\n")], "supplier_key_before_publish")
    secret_step_match = re.search(
        r"(?ms)^      - name: Sign installed evidence.*?(?=^      - name:)", publish
    )
    _require(secret_step_match is not None, "supplier_signing_step_missing")
    assert secret_step_match is not None
    secret_step = secret_step_match.group(0)
    _require(
        "trusted-evidence/scripts/sign_candidate_evidence.py" in secret_step,
        "supplier_signer_not_from_commit_a",
    )
    _require("source/scripts" not in secret_step, "supplier_key_exposed_to_source_script")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", required=True, type=Path)
    parser.add_argument("--publisher", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        check_trusted_candidate_workflows(
            builder=args.builder.resolve(),
            publisher=args.publisher.resolve(),
            repository_root=args.repository_root.resolve(),
        )
    except (OSError, StopIteration, TrustPolicyError) as exc:
        print(f"TRUSTED_CANDIDATE_WORKFLOW_POLICY=FAIL:{exc}")
        return 1
    print("TRUSTED_CANDIDATE_WORKFLOW_POLICY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
