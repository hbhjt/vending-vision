from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.check_trusted_candidate_workflows import (
    _logical_shell_commands,
    _shell_statements,
    _shell_tokens,
)
from scripts.check_trusted_precutover_proof_workflow import (
    CANONICAL_GH_PATH,
    _assert_candidate_builder_authority_sync,
)
from scripts.verify_hosted_release_authority import AuthorityError, verify_release_fence
from scripts.verify_hosted_environment import verify_environment
from scripts.workflow_yaml import load_workflow_yaml


REPOSITORY = "hbhjt/vending-vision"
SOURCE_REF = "refs/tags/v1.2.3-rc.4"
SOURCE_COMMIT = "a" * 40
AUTHORITY_COMMIT = "41afbd9bd07b67df9f93de1dea1a9f9b0cea0228"


def _release(value: object) -> dict:
    return {"data": {"repository": {"release": value}}}


def _steps(workflow: dict, job_name: str) -> list[dict]:
    jobs = workflow["jobs"]
    job = jobs[job_name]
    steps = job["steps"]
    assert isinstance(steps, list)
    return steps


def _named_step_index(steps: list[dict], name: str) -> int:
    matches = [index for index, step in enumerate(steps) if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _assert_real_guarded_candidate_attestation(step: dict, archive: str, message: str):
    run = step["run"]
    statements = [
        (statement, call_operator, _shell_tokens(statement, "pwsh"))
        for command in _logical_shell_commands(run, "pwsh")
        for statement, call_operator in _shell_statements(command, "pwsh")
    ]
    guard = (
        'if (-not (Test-Path -LiteralPath "C:\\Program Files\\GitHub CLI\\gh.exe" '
        f'-PathType Leaf)) {{ throw "{message}" }}'
    )
    guard_indexes = [
        index for index, (statement, _, _) in enumerate(statements) if statement == guard
    ]
    calls = []
    for index, (_, call_operator, token_facts) in enumerate(statements):
        tokens = tuple(token for token, _ in token_facts)
        if tokens[1:4] == ("attestation", "verify", archive):
            calls.append((index, call_operator, token_facts))
    assert len(guard_indexes) == 1
    assert len(calls) == 1
    call_index, call_operator, token_facts = calls[0]
    executable, quoted = token_facts[0]
    assert call_operator and quoted
    assert executable.replace("/", "\\").casefold() == CANONICAL_GH_PATH
    assert guard_indexes[0] < call_index


def test_publish_admission_accepts_only_an_absent_release_for_the_exact_rc_tag():
    verify_release_fence(
        _release(None),
        mode="publish-admission",
        repository=REPOSITORY,
        source_ref=SOURCE_REF,
        source_commit=SOURCE_COMMIT,
    )

    with pytest.raises(AuthorityError, match="already_exists"):
        verify_release_fence(
            _release(
                {
                    "tagName": "v1.2.3-rc.4",
                    "targetCommitish": SOURCE_COMMIT,
                    "isDraft": False,
                    "isPrerelease": True,
                }
            ),
            mode="publish-admission",
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


def test_published_and_proof_fences_bind_a_nondraft_prerelease_to_the_exact_commit():
    value = {
        "tagName": "v1.2.3-rc.4",
        "targetCommitish": SOURCE_COMMIT,
        "isDraft": False,
        "isPrerelease": True,
    }
    for mode in ("publish-complete", "proof"):
        verify_release_fence(
            value,
            mode=mode,
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    "release",
    [
        None,
        {
            "tagName": "v1.2.3-rc.5",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": False,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": "b" * 40,
            "isDraft": False,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": True,
            "isPrerelease": True,
        },
        {
            "tagName": "v1.2.3-rc.4",
            "targetCommitish": SOURCE_COMMIT,
            "isDraft": False,
            "isPrerelease": False,
        },
    ],
)
def test_proof_fails_closed_for_missing_or_rewritten_release_identity(release):
    with pytest.raises(AuthorityError):
        verify_release_fence(
            _release(release),
            mode="proof",
            repository=REPOSITORY,
            source_ref=SOURCE_REF,
            source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    ("repository", "source_ref", "source_commit"),
    [
        ("fork/vending-vision", SOURCE_REF, SOURCE_COMMIT),
        (REPOSITORY, "refs/heads/main", SOURCE_COMMIT),
        (REPOSITORY, SOURCE_REF, "A" * 40),
    ],
)
def test_authority_rejects_repository_ref_and_digest_scope_drift(
    repository, source_ref, source_commit
):
    with pytest.raises(AuthorityError):
        verify_release_fence(
            _release(None),
            mode="publish-admission",
            repository=repository,
            source_ref=source_ref,
            source_commit=source_commit,
        )


def test_exact4_publish_drops_hosted_authority_while_proof_keeps_its_own_fence():
    publisher = (ROOT / ".github/workflows/publish-candidate.yml").read_text("utf-8")
    proof = (
        ROOT / ".github/workflows/trusted-precutover-companion-proof.yml"
    ).read_text("utf-8")

    assert "rulesets?targets=tag" not in publisher
    assert "rulesets?targets=tag" not in proof
    publish_job = publisher[publisher.index("  publish:\n") :]
    assert "environment: production" in publish_job.split("steps:", 1)[0]
    for removed in (
        f"ref: {AUTHORITY_COMMIT}",
        "verify_hosted_release_authority.py",
        "--mode publish-admission",
        "--mode publish-complete",
        "approve_candidate_source.py",
        "refs/heads/main",
        "refs/tags/",
    ):
        assert removed not in publish_job
    assert "gh release create" in publish_job

    _assert_candidate_builder_authority_sync(proof, ROOT)
    parsed_proof = load_workflow_yaml(proof)
    execute = proof[proof.index("  execute:\n") : proof.index("  sign:\n")]
    sign = proof[proof.index("  sign:\n") : proof.index("  verify:\n")]
    expected = (
        (
            "execute",
            execute,
            "Approve exact RC tag and its existing release target",
            "Verify candidate GitHub provenance before executing frozen companion",
            "proof-input/candidate/candidate.zip",
            "candidate GitHub CLI is unavailable",
        ),
        (
            "sign",
            sign,
            "Fresh approve exact protected source before signing",
            "Fresh verify candidate GitHub provenance without execution",
            "signer-proof-input/candidate/candidate.zip",
            "signer candidate GitHub CLI is unavailable",
        ),
    )
    for job_name, job, approval_name, attestation_name, archive, guard_message in expected:
        assert "environment: experimental-candidate" in job.split("steps:", 1)[0]
        assert f"ref: {AUTHORITY_COMMIT}" in job
        assert "verify_hosted_release_authority.py" in job
        assert "--mode proof" in job
        steps = _steps(parsed_proof, job_name)
        approval_index = _named_step_index(steps, approval_name)
        attestation_index = _named_step_index(steps, attestation_name)
        assert approval_index < attestation_index
        _assert_real_guarded_candidate_attestation(
            steps[attestation_index], archive, guard_message
        )

    combined = publisher + proof
    for forbidden in (
        "git tag -f",
        "git push --delete",
        "gh release delete",
        "gh release edit",
    ):
        assert forbidden not in combined


def test_operator_preflight_requires_the_exact_existing_rc_tag_environment_policy():
    preflight = (ROOT / "scripts/verify_hosted_environment.py").read_text("utf-8")
    assert "experimental-candidate" in preflight
    assert "v*.*.*-rc.*" in preflight
    assert "PUT" not in preflight and "POST" not in preflight and "DELETE" not in preflight

    environment = {
        "name": "experimental-candidate",
        "deployment_branch_policy": {
            "protected_branches": False,
            "custom_branch_policies": True,
        },
    }
    policies = {
        "branch_policies": [{"id": 1, "type": "tag", "name": "v*.*.*-rc.*"}]
    }
    verify_environment(environment, policies)

    for drift in (
        {"branch_policies": []},
        {"branch_policies": [*policies["branch_policies"], policies["branch_policies"][0]]},
        {"branch_policies": [{"id": 1, "type": "branch", "name": "v*.*.*-rc.*"}]},
    ):
        with pytest.raises(AssertionError, match="configuration_drift"):
            verify_environment(environment, drift)
