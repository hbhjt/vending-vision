from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

import pytest

from scripts.approve_candidate_source import approve_source
from scripts.verify_release_tag_ruleset import verify_rulesets
from scripts.workflow_yaml import load_workflow_yaml, workflow_run_scalars


ROOT = Path(__file__).parents[1]
TRUSTED_PROOF = (
    ROOT / ".github" / "workflows" / "trusted-precutover-companion-proof.yml"
)
COMPANION_BUILDER_COMMIT = "ebefe97377e4597ab62bc7ea6ab4849219df6fdd"
POLICY = ROOT / "scripts" / "check_trusted_precutover_proof_workflow.py"
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


def test_trusted_windows_companion_proof_workflow_exists():
    assert TRUSTED_PROOF.is_file(), (
        "no trusted Windows workflow executes the frozen companion and attests "
        "its canonical proof"
    )


def test_trusted_proof_has_closed_https_inputs_and_pins_companion_builder():
    source = TRUSTED_PROOF.read_text("utf-8")
    match = re.search(
        r"(?ms)^  workflow_call:\n    inputs:\n(?P<body>.*?)(?=^    outputs:)", source
    )
    assert match is not None
    assert set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", match.group("body"))) == INPUTS
    assert (
        "uses: hbhjt/vending-vision/.github/workflows/"
        f"trusted-precutover-companion-builder.yml@{COMPANION_BUILDER_COMMIT}"
    ) in source
    assert "runs-on: windows-latest" in source
    assert "secrets:" not in source
    for forbidden in ("path_input", "command", "predicate", "worker_path", "artifact_path"):
        assert forbidden not in source


def test_proof_and_fresh_verify_jobs_use_only_immutable_trusted_code_and_safe_env():
    source = TRUSTED_PROOF.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    jobs = workflow["jobs"]

    assert set(jobs) == {"companion_builder", "prove", "verify"}
    assert jobs["prove"]["runs-on"] == "windows-latest"
    assert jobs["verify"]["runs-on"] == "windows-latest"
    assert source.count("repository: ${{ job.workflow_repository }}") == 2
    assert source.count("ref: ${{ job.workflow_sha }}") == 2
    assert "path: source" not in source
    assert "ref: ${{ github.sha }}" not in source
    assert "actions/checkout@v4" in source
    assert all("${{" not in run for run in workflow_run_scalars(source))

    prove = source[source.index("  prove:\n") : source.index("  verify:\n")]
    verify = source[source.index("  verify:\n") :]
    assert "vending-vision-precutover-verifier.exe" in prove
    assert "$identity.modelPack.descriptorSha256" in prove
    assert "proof-input/candidate/trusted-builder-evidence.json" in prove
    assert "proof-input/model/official-model-pack.zip" in prove
    assert "actions/attest-build-provenance@v4" in prove
    assert prove.index("vending-vision-precutover-verifier.exe") < prove.index(
        "actions/attest-build-provenance@v4"
    )
    assert "gh attestation verify" in verify
    assert "--deny-self-hosted-runners" in verify
    assert verify.index("gh attestation verify") < verify.index(
        "name: Upload fresh-runner verified proof"
    )


def _check_policy(workflow: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(POLICY),
            "--workflow",
            str(workflow),
            "--repository-root",
            str(ROOT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_trusted_proof_workflow_passes_executable_trust_policy():
    completed = _check_policy(TRUSTED_PROOF)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_trusted_proof_requires_real_tag_peel_main_ancestry_and_active_ruleset():
    source = TRUSTED_PROOF.read_text("utf-8")

    assert "trusted-proof/scripts/approve_candidate_source.py" in source
    assert "--protected-main refs/remotes/origin/main" in source
    assert "+refs/heads/main:refs/remotes/origin/main" in source
    assert "trusted-proof/scripts/verify_release_tag_ruleset.py" in source
    assert "rulesets?targets=tag&includes_parents=true&per_page=100" in source
    assert "^refs/(heads|tags)/" not in source


def test_trusted_proof_policy_rejects_mutable_authority_and_execution_bypasses(tmp_path):
    trusted = TRUSTED_PROOF.read_text("utf-8")
    attest_match = re.search(
        r"(?ms)^      - name: Attest only the verified canonical companion proof\n"
        r".*?(?=^      - name: Seal exact proof handoff)",
        trusted,
    )
    assert attest_match is not None
    attest_step = attest_match.group(0)
    attest_early = trusted.replace(attest_step, "").replace(
        "      - name: Validate canonical proof bindings before attestation\n",
        attest_step + "      - name: Validate canonical proof bindings before attestation\n",
        1,
    )
    mutations = {
        "mutable-builder-pin": trusted.replace(
            COMPANION_BUILDER_COMMIT, "${{ github.sha }}", 1
        ),
        "self-source-checkout": trusted.replace(
            "ref: ${{ job.workflow_sha }}", "ref: ${{ github.sha }}", 1
        ),
        "raw-input-expression": trusted.replace(
            "run: |\n          if ($env:TRUSTED_WORKFLOW_REPOSITORY",
            'run: |\n          Write-Output "${{ inputs.candidate_archive_url }}"\n'
            "          if ($env:TRUSTED_WORKFLOW_REPOSITORY",
            1,
        ),
        "candidate-script-execution": trusted.replace(
            "run: |\n          New-Item -ItemType Directory -Path proof-input/candidate",
            "run: |\n          & proof-input/scripts/untrusted.py\n"
            "          New-Item -ItemType Directory -Path proof-input/candidate",
            1,
        ),
        "json-emitter-instead-of-companion": trusted.replace(
            "& $companionExe --candidate-artifact",
            "Set-Content -LiteralPath precutover-ai-proof.json -Value '{}'\n          # removed frozen invocation --candidate-artifact",
            1,
        ),
        "attestation-before-proof-verify": attest_early,
        "missing-final-upload": trusted[: trusted.index("      - name: Upload fresh-runner verified proof\n")],
    }
    for name, source in mutations.items():
        candidate = tmp_path / f"{name}.yml"
        candidate.write_text(source, "utf-8")
        completed = _check_policy(candidate)
        assert completed.returncode != 0, name


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_repository(root: Path) -> tuple[Path, str, str]:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "precutover-proof@example.test")
    _git(root, "config", "user.name", "Precutover Proof")
    (root / "source.txt").write_text("approved\n", "utf-8")
    _git(root, "add", "source.txt")
    _git(root, "commit", "-m", "approved")
    approved = _git(root, "rev-parse", "HEAD")
    (root / "source.txt").write_text("main tip\n", "utf-8")
    _git(root, "commit", "-am", "main tip")
    main_tip = _git(root, "rev-parse", "HEAD")
    return root, approved, main_tip


@pytest.mark.parametrize("case", ["head-ref", "moved-tag", "non-main-ancestor"])
def test_real_source_approval_rejects_branch_moved_tag_and_non_main_commit(tmp_path, case):
    repository, approved, main_tip = _source_repository(tmp_path / "source")
    source_commit = approved
    source_ref = "refs/tags/v1.2.3-rc.1"
    if case == "head-ref":
        source_ref = "refs/heads/1.2.3-rc.1"
    elif case == "moved-tag":
        _git(repository, "tag", "-a", "v1.2.3-rc.1", "-m", "moved", main_tip)
    else:
        _git(repository, "checkout", "--orphan", "unapproved")
        (repository / "source.txt").write_text("unapproved\n", "utf-8")
        _git(repository, "add", "source.txt")
        _git(repository, "commit", "-m", "unapproved")
        source_commit = _git(repository, "rev-parse", "HEAD")
        _git(repository, "tag", "-a", "v1.2.3-rc.1", "-m", "unapproved", source_commit)

    with pytest.raises(AssertionError):
        approve_source(
            git_dir=repository / ".git",
            source_commit=source_commit,
            source_ref=source_ref,
            protected_main="refs/heads/main",
        )


def test_real_source_approval_accepts_exact_protected_tag_only_with_failclosed_ruleset(
    tmp_path,
):
    repository, approved, _ = _source_repository(tmp_path / "source")
    source_ref = "refs/tags/v1.2.3-rc.1"
    _git(repository, "tag", "-a", "v1.2.3-rc.1", "-m", "approved", approved)
    approve_source(
        git_dir=repository / ".git",
        source_commit=approved,
        source_ref=source_ref,
        protected_main="refs/heads/main",
    )
    protected = {
        "id": 91,
        "target": "tag",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {"include": ["refs/tags/v*.*.*-rc.*"], "exclude": []}
        },
        "rules": [{"type": "update"}, {"type": "deletion"}],
    }
    assert verify_rulesets(
        [protected], repository="hbhjt/vending-vision", source_ref=source_ref
    ) == 91
    for unsafe in (
        [],
        [{**protected, "enforcement": "disabled"}],
        [{**protected, "bypass_actors": [{"actor_id": 1}]}],
        [{**protected, "rules": [{"type": "update"}]}],
    ):
        with pytest.raises(AssertionError):
            verify_rulesets(
                unsafe, repository="hbhjt/vending-vision", source_ref=source_ref
            )
