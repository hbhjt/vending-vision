from __future__ import annotations

from pathlib import Path
import re
import os
import subprocess
import sys

import pytest

from scripts.workflow_yaml import load_workflow_yaml
from scripts.check_trusted_candidate_workflows import (
    TrustPolicyError,
    _logical_shell_commands,
    _shell_statements,
    _shell_tokens,
)


ROOT = Path(__file__).parents[1]
TRUSTED_BUILDER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml"
PUBLISHER = ROOT / ".github" / "workflows" / "publish-candidate.yml"
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"
TRUSTED_SIGNER = ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml"
TRUSTED_SIGNER_COMMIT = "59b4fee088db08f3008c137409f98577de987595"
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
    wheelhouse_download = next(
        line.strip()
        for line in workflow.splitlines()
        if "download_verified_archive.py" in line
    )
    assert "--url $env:CORE_WHEELHOUSE_URL" in wheelhouse_download
    assert "--sha256 $env:CORE_WHEELHOUSE_SHA256" in wheelhouse_download
    assert "--expected-bytes $env:CORE_WHEELHOUSE_BYTES" in wheelhouse_download
    assert "--destination wheelhouse" in wheelhouse_download
    assert "--total-timeout-seconds 1800" in wheelhouse_download
    for forbidden in ("artifact_path", "worker_path", "predicate", "custom_command", "command_input"):
        assert forbidden not in workflow

    verify = workflow.index("--require-ai-worker")
    attest = workflow.index("actions/attest-build-provenance@v4")
    upload = workflow.index("actions/upload-artifact@v4")
    assert verify < attest < upload
    bundle = workflow[workflow.index("- name: Create the canonical candidate ZIP") : verify]
    verifier = workflow[verify:attest]
    attestation = workflow[attest:upload]
    assert "$artifact = (Resolve-Path -LiteralPath $artifact).Path" in bundle
    assert "TRUSTED_CANDIDATE_ARTIFACT=$artifact" in bundle
    assert "$env:TRUSTED_CANDIDATE_ARTIFACT" in verifier
    assert "subject-path: ${{ env.TRUSTED_CANDIDATE_ARTIFACT }}" in attestation
    assert "..\\trusted-output" not in attestation


def _workflow_step_run(source: str, name: str) -> tuple[int, str]:
    workflow = load_workflow_yaml(source)
    steps = workflow["jobs"]["build"]["steps"]
    matches = [
        (index, step["run"])
        for index, step in enumerate(steps)
        if isinstance(step, dict) and step.get("name") == name
    ]
    assert len(matches) == 1
    return matches[0]


def _has_post_attestation_subject_fence(run: str) -> bool:
    statements = [
        _shell_tokens(statement, "pwsh")
        for command in _logical_shell_commands(run, "pwsh")
        for statement, _ in _shell_statements(command, "pwsh")
    ]
    digest = (
        ("$subjectDigest", False),
        ("=", False),
        ("(Get-FileHash", False),
        ("-LiteralPath", False),
        ("$env:TRUSTED_CANDIDATE_ARTIFACT", False),
        ("-Algorithm", False),
        ("SHA256).Hash", False),
    )
    comparison = (
        ("if", False),
        ("(-not", False),
        ("[string]::Equals($subjectDigest,", False),
        ("$env:SUBJECT_SHA256,", False),
        ("[System.StringComparison]::OrdinalIgnoreCase))", False),
        ("{", False),
        ("throw", False),
        ("attested candidate subject digest changed", True),
        ("}", False),
    )
    return digest in statements and comparison in statements


def _hide_fence_in_here_string(source: str, opener: str) -> str:
    fence = (
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }'
    )
    mutated = source.replace(
        fence,
        f"          {opener}\n"
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }\n'
        "          '@",
        1,
    )
    assert mutated != source
    return mutated


def _hide_fence_in_block_comment(source: str, opener: str, closer: str) -> str:
    fence = (
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }'
    )
    mutated = source.replace(
        fence,
        f"          {opener}\n"
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }\n'
        f"          {closer}",
        1,
    )
    assert mutated != source
    return mutated


def _hide_fence_in_multiline_quote(source: str, quote: str) -> str:
    fence = (
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }'
    )
    if quote == '"':
        replacement = (
            '          $literal = "start`\n'
            '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
            '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }\n'
            '          end"'
        )
    else:
        replacement = (
            "          $literal = 'start\n"
            '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
            '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }\n'
            "          end'"
        )
    mutated = source.replace(fence, replacement, 1)
    assert mutated != source
    return mutated


def test_builder_fences_the_attested_canonical_subject_before_evidence_or_upload():
    source = TRUSTED_BUILDER.read_text("utf-8")
    attest = source.index("actions/attest-build-provenance@v4")
    evidence_index, evidence = _workflow_step_run(source, "Record trusted builder evidence")
    upload = source.index("actions/upload-artifact@v4")

    assert attest < source.index("- name: Record trusted builder evidence") < upload
    assert evidence_index < next(
        index
        for index, step in enumerate(load_workflow_yaml(source)["jobs"]["build"]["steps"])
        if isinstance(step, dict) and step.get("name") == "Upload only trusted builder outputs"
    )
    assert _has_post_attestation_subject_fence(evidence)

    commented = source.replace(
        "Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256",
        "# Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256",
        1,
    )
    _, commented_evidence = _workflow_step_run(commented, "Record trusted builder evidence")
    assert not _has_post_attestation_subject_fence(commented_evidence)

    wrong_path = source.replace(
        "$env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256",
        '"trusted-output/*.zip" -Algorithm SHA256',
        1,
    )
    _, wrong_path_evidence = _workflow_step_run(wrong_path, "Record trusted builder evidence")
    assert not _has_post_attestation_subject_fence(wrong_path_evidence)

    here_string = _hide_fence_in_here_string(source, "$fence = @'")
    _, here_string_evidence = _workflow_step_run(here_string, "Record trusted builder evidence")
    assert not _has_post_attestation_subject_fence(here_string_evidence)

    double_here_string = here_string.replace("$fence = @'", '$fence = @"').replace(
        "'@", '"@'
    )
    _, double_here_string_evidence = _workflow_step_run(
        double_here_string, "Record trusted builder evidence"
    )
    assert not _has_post_attestation_subject_fence(double_here_string_evidence)

    for opener in ("$fence = [string]@'", "Write-Output @'", "$fence = @(@'"):
        hidden = _hide_fence_in_here_string(source, opener)
        _, hidden_evidence = _workflow_step_run(hidden, "Record trusted builder evidence")
        assert not _has_post_attestation_subject_fence(hidden_evidence)

    quoted_literal = source.replace(
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        '          $literal = "@\'"\n          $bundleFile = "github-build-provenance.sigstore.json"',
        1,
    )
    _, quoted_literal_evidence = _workflow_step_run(
        quoted_literal, "Record trusted builder evidence"
    )
    assert _has_post_attestation_subject_fence(quoted_literal_evidence)

    block_comment = _hide_fence_in_block_comment(source, "<#", "#>")
    _, block_comment_evidence = _workflow_step_run(
        block_comment, "Record trusted builder evidence"
    )
    assert not _has_post_attestation_subject_fence(block_comment_evidence)

    escaped_double_quote_block = source.replace(
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }',
        '          $literal = "foo`"" <#\n'
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }\n'
        '          #>',
        1,
    )
    _, escaped_double_quote_evidence = _workflow_step_run(
        escaped_double_quote_block, "Record trusted builder evidence"
    )
    assert not _has_post_attestation_subject_fence(escaped_double_quote_evidence)

    inline_block_comment = source.replace(
        '          $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash\n'
        '          if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" }',
        '          <# $subjectDigest = (Get-FileHash -LiteralPath $env:TRUSTED_CANDIDATE_ARTIFACT -Algorithm SHA256).Hash #>\n'
        '          <# if (-not [string]::Equals($subjectDigest, $env:SUBJECT_SHA256, [System.StringComparison]::OrdinalIgnoreCase)) { throw "attested candidate subject digest changed" } #>',
        1,
    )
    _, inline_block_evidence = _workflow_step_run(
        inline_block_comment, "Record trusted builder evidence"
    )
    assert not _has_post_attestation_subject_fence(inline_block_evidence)

    unterminated_block = _hide_fence_in_block_comment(source, "<#", "")
    _, unterminated_evidence = _workflow_step_run(
        unterminated_block, "Record trusted builder evidence"
    )
    with pytest.raises(TrustPolicyError, match="unterminated_block_comment"):
        _has_post_attestation_subject_fence(unterminated_evidence)

    quoted_block_literal = source.replace(
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        '          $literal = "<#"\n          $bundleFile = "github-build-provenance.sigstore.json"',
        1,
    )
    _, quoted_block_evidence = _workflow_step_run(
        quoted_block_literal, "Record trusted builder evidence"
    )
    assert _has_post_attestation_subject_fence(quoted_block_evidence)

    doubled_single_quote_literal = source.replace(
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        "          $literal = 'foo'' <# still literal'\n          $bundleFile = \"github-build-provenance.sigstore.json\"",
        1,
    )
    _, doubled_single_quote_evidence = _workflow_step_run(
        doubled_single_quote_literal, "Record trusted builder evidence"
    )
    assert _has_post_attestation_subject_fence(doubled_single_quote_evidence)

    for quote in ('"', "'"):
        multiline_quote = _hide_fence_in_multiline_quote(source, quote)
        _, multiline_quote_evidence = _workflow_step_run(
            multiline_quote, "Record trusted builder evidence"
        )
        assert not _has_post_attestation_subject_fence(multiline_quote_evidence)

    real_fence_after_quote = source.replace(
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        '          $literal = "start`\n          decoy\n          end"\n'
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        1,
    )
    _, real_fence_after_quote_evidence = _workflow_step_run(
        real_fence_after_quote, "Record trusted builder evidence"
    )
    assert _has_post_attestation_subject_fence(real_fence_after_quote_evidence)

    unterminated_quote = source.replace(
        '          $bundleFile = "github-build-provenance.sigstore.json"',
        '          $literal = "start`\n          $bundleFile = "github-build-provenance.sigstore.json"',
        1,
    )
    _, unterminated_quote_evidence = _workflow_step_run(
        unterminated_quote, "Record trusted builder evidence"
    )
    with pytest.raises(TrustPolicyError, match="unterminated_quote"):
        _has_post_attestation_subject_fence(unterminated_quote_evidence)


def test_active_verified_archive_downloads_have_an_explicit_bounded_total_timeout():
    completed = _check_policy(PUBLISHER)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("name", "mutate", "expected_error"),
    (
        (
            "missing",
            lambda source: source.replace(" --total-timeout-seconds 1800", "", 1),
            "publisher_archive_downloader_timeout_count",
        ),
        (
            "commented-only",
            lambda source: source.replace(
                "          python trusted-verifier/scripts/download_verified_archive.py",
                "          # python trusted-verifier/scripts/download_verified_archive.py",
                1,
            ),
            "publisher_archive_downloader_count",
        ),
        (
            "string-only",
            lambda source: source.replace(
                "          python trusted-verifier/scripts/download_verified_archive.py",
                "          Write-Output \"python trusted-verifier/scripts/download_verified_archive.py",
                1,
            ).replace("--total-timeout-seconds 1800\n", "--total-timeout-seconds 1800\"\n", 1),
            "publisher_archive_downloader_count",
        ),
        (
            "commented-timeout",
            lambda source: source.replace(
                " --total-timeout-seconds 1800",
                " # --total-timeout-seconds 1800",
                1,
            ),
            "publisher_archive_downloader_timeout_count",
        ),
        (
            "duplicate",
            lambda source: source.replace(
                "--total-timeout-seconds 1800",
                "--total-timeout-seconds 1800 --total-timeout-seconds 1800",
                1,
            ),
            "publisher_archive_downloader_timeout_count",
        ),
        (
            "zero",
            lambda source: source.replace("--total-timeout-seconds 1800", "--total-timeout-seconds 0", 1),
            "publisher_archive_downloader_timeout_bounds",
        ),
        (
            "too-large",
            lambda source: source.replace("--total-timeout-seconds 1800", "--total-timeout-seconds 3601", 1),
            "publisher_archive_downloader_timeout_bounds",
        ),
        (
            "second-downloader-without-timeout",
            lambda source: source.replace(
                "--total-timeout-seconds 1800",
                "--total-timeout-seconds 1800; python trusted-verifier/scripts/"
                "download_verified_archive.py --url $env:CORE_WHEELHOUSE_URL "
                "--sha256 $env:CORE_WHEELHOUSE_SHA256 --expected-bytes "
                "$env:CORE_WHEELHOUSE_BYTES --destination verifier-wheelhouse",
                1,
            ),
            "publisher_archive_downloader_count",
        ),
        (
            "timeout-in-another-statement",
            lambda source: source.replace(
                " --total-timeout-seconds 1800",
                "; Write-Output --total-timeout-seconds 1800",
                1,
            ),
            "publisher_archive_downloader_timeout_count",
        ),
        (
            "escaped-quoted-timeout-in-another-statement",
            lambda source: source.replace(
                " --total-timeout-seconds 1800",
                "; Write-Output `\"--total-timeout-seconds 1800`\"",
                1,
            ),
            "publisher_archive_downloader_timeout_count",
        ),
        (
            "absolute-fourth-downloader",
            lambda source: source.replace(
                "--total-timeout-seconds 1800",
                "--total-timeout-seconds 1800; /usr/bin/python "
                "trusted-verifier/scripts/download_verified_archive.py "
                "--url $env:CORE_WHEELHOUSE_URL --sha256 "
                "$env:CORE_WHEELHOUSE_SHA256 --expected-bytes "
                "$env:CORE_WHEELHOUSE_BYTES --destination verifier-wheelhouse "
                "--total-timeout-seconds 1800",
                1,
            ),
            "publisher_archive_downloader_count",
        ),
        (
            "malformed-downloader-invocation",
            lambda source: source.replace(
                "python trusted-verifier/scripts/download_verified_archive.py",
                "Write-Output trusted-verifier/scripts/download_verified_archive.py",
                1,
            ),
            "archive_downloader_unsupported_invocation",
        ),
        (
            "unknown-downloader-shell",
            lambda source: source.replace(
                "      - name: Safely extract and run the full candidate verifier\n"
                "        shell: pwsh",
                "      - name: Safely extract and run the full candidate verifier\n"
                "        shell: cmd",
                1,
            ),
            "archive_downloader_shell_unknown",
        ),
        (
            "single-quoted-backslash-does-not-hide-second-downloader",
            lambda source: source.replace(
                "--total-timeout-seconds 1800",
                "--total-timeout-seconds 1800; Write-Output 'x\\'; python "
                "trusted-verifier/scripts/download_verified_archive.py "
                "--url $env:CORE_WHEELHOUSE_URL --sha256 "
                "$env:CORE_WHEELHOUSE_SHA256 --expected-bytes "
                "$env:CORE_WHEELHOUSE_BYTES --destination verifier-wheelhouse",
                1,
            ),
            "publisher_archive_downloader_count",
        ),
        (
            "single-quoted-backtick-does-not-hide-second-downloader",
            lambda source: source.replace(
                "--total-timeout-seconds 1800",
                "--total-timeout-seconds 1800; Write-Output 'x`'; python "
                "trusted-verifier/scripts/download_verified_archive.py "
                "--url $env:CORE_WHEELHOUSE_URL --sha256 "
                "$env:CORE_WHEELHOUSE_SHA256 --expected-bytes "
                "$env:CORE_WHEELHOUSE_BYTES --destination verifier-wheelhouse",
                1,
            ),
            "publisher_archive_downloader_count",
        ),
    ),
)
def test_trust_policy_rejects_non_executable_or_unbounded_archive_downloader(
    tmp_path, name, mutate, expected_error
):
    candidate = tmp_path / f"{name}.yml"
    candidate.write_text(mutate(PUBLISHER.read_text("utf-8")), "utf-8")

    completed = _check_policy(candidate)

    assert completed.returncode != 0
    assert expected_error in completed.stdout


@pytest.mark.parametrize("continuation", ("`", "\\"))
def test_trust_policy_accepts_a_continued_archive_downloader(tmp_path, continuation):
    candidate = tmp_path / "continued.yml"
    source = PUBLISHER.read_text("utf-8")
    if continuation == "\\":
        source = source.replace(
            "      - name: Safely extract and run the full candidate verifier\n"
            "        shell: pwsh",
            "      - name: Safely extract and run the full candidate verifier\n"
            "        shell: bash",
            1,
        )
    candidate.write_text(
        source.replace(
            "--destination verifier-wheelhouse --total-timeout-seconds 1800",
            f"--destination verifier-wheelhouse {continuation}\n"
            "          --total-timeout-seconds 1800",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_trust_policy_ignores_quoted_statement_separators(tmp_path):
    candidate = tmp_path / "quoted-separator.yml"
    candidate.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "          python trusted-verifier/scripts/download_verified_archive.py",
            "          Write-Output '; && || | &' ; python trusted-verifier/"
            "scripts/download_verified_archive.py",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "executable",
    ("/usr/bin/python", r"& C:\Python311\python.exe"),
)
def test_trust_policy_accepts_an_absolute_python_downloader_invocation(tmp_path, executable):
    candidate = tmp_path / "absolute-python.yml"
    candidate.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "python trusted-verifier/scripts/download_verified_archive.py",
            f"{executable} trusted-verifier/scripts/download_verified_archive.py",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_trust_policy_rejects_a_pwsh_quoted_python_without_call_operator(tmp_path):
    candidate = tmp_path / "quoted-pwsh-no-call.yml"
    candidate.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "python trusted-verifier/scripts/download_verified_archive.py",
            r"'C:\Python311\python.exe' trusted-verifier/scripts/download_verified_archive.py",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode != 0
    assert "archive_downloader_unsupported_invocation" in completed.stdout


@pytest.mark.parametrize(
    "prefix",
    ("'&' ", '"&" ', "`& "),
)
def test_trust_policy_rejects_a_non_syntactic_pwsh_call_operator(tmp_path, prefix):
    candidate = tmp_path / "non-syntactic-call-operator.yml"
    candidate.write_text(
        PUBLISHER.read_text("utf-8").replace(
            "python trusted-verifier/scripts/download_verified_archive.py",
            prefix + "python trusted-verifier/scripts/download_verified_archive.py",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode != 0
    assert "archive_downloader_unsupported_invocation" in completed.stdout


@pytest.mark.parametrize(
    ("shell", "executable"),
    (
        ("pwsh", r"& 'C:\Python311\python.exe'"),
        ("bash", r"'C:\Python311\python.exe'"),
    ),
)
def test_trust_policy_accepts_a_quoted_python_for_its_shell(tmp_path, shell, executable):
    candidate = tmp_path / f"quoted-{shell}.yml"
    source = PUBLISHER.read_text("utf-8").replace(
        "      - name: Safely extract and run the full candidate verifier\n"
        "        shell: pwsh",
        "      - name: Safely extract and run the full candidate verifier\n"
        f"        shell: {shell}",
        1,
    )
    candidate.write_text(
        source.replace(
            "python trusted-verifier/scripts/download_verified_archive.py",
            f"{executable} trusted-verifier/scripts/download_verified_archive.py",
            1,
        ),
        "utf-8",
    )

    completed = _check_policy(candidate)

    assert completed.returncode == 0, completed.stdout + completed.stderr


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


def test_publish_caller_pins_builder_a_and_directly_signs_only_verified_evidence():
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
    assert "signer_identity: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}" in workflow
    assert f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"' in workflow
    assert "--signer-repo" not in workflow
    assert (
        '--signer-workflow "hbhjt/vending-vision/.github/workflows/'
        'trusted-ai-candidate-builder.yml"'
    ) in workflow
    assert "--deny-self-hosted-runners" in workflow
    assert "actions/attest-build-provenance" not in workflow
    assert "scripts/build_exe.ps1" not in workflow
    assert workflow.count("${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}") == 1
    signer_call = workflow[workflow.index("  trusted_signer:\n"):workflow.index("  sign_evidence:\n")]
    direct_sign = workflow[workflow.index("  sign_evidence:\n"):workflow.index("  publish:\n")]
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in signer_call
    assert "environment: experimental-candidate" in direct_sign
    assert "needs: [verify, trusted_signer]" in direct_sign
    assert "${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}" in direct_sign
    assert "scripts/verify_trusted_script_set.py" in direct_sign
    assert "scripts/evidence_artifact.py" in direct_sign
    assert "scripts/sign_candidate_evidence.py" in direct_sign
    assert "--kind unsigned --expected-digest $env:UNSIGNED_EVIDENCE_SHA256" in direct_sign
    assert "--private-key $key --signer-identity $env:VISION_SUPPLIER_SIGNER_IDENTITY" in direct_sign
    publish = workflow[workflow.index("  publish:\n"):]
    assert "needs: [verify, sign_evidence]" in publish
    assert f"ref: {TRUSTED_SIGNER_COMMIT}" in publish
    assert publish.count("actions/download-artifact@v4") == 2
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in publish
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
        "missing-signer-identity": trusted.replace(
            "      signer_identity: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}\n", ""
        ),
        "wrong-signer-identity": trusted.replace(
            "signer_identity: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}",
            "signer_identity: ${{ vars.WRONG_SIGNER_IDENTITY }}",
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


@pytest.mark.parametrize(
    ("name", "old", "new"),
    (
        (
            "missing-direct-environment",
            "    environment: experimental-candidate\n    permissions:\n      contents: read\n    outputs:\n      signed_evidence_artifact_name:",
            "    permissions:\n      contents: read\n    outputs:\n      signed_evidence_artifact_name:",
        ),
        (
            "identity-drift",
            "VISION_SUPPLIER_SIGNER_IDENTITY: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}",
            "VISION_SUPPLIER_SIGNER_IDENTITY: ${{ vars.WRONG_SIGNER_IDENTITY }}",
        ),
        (
            "extra-secret-in-verifier",
            "          SOURCE_REF: ${{ github.ref }}\n",
            "          SOURCE_REF: ${{ github.ref }}\n          VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n",
        ),
        (
            "secret-exposed-before-sign-step",
            "          VISION_SUPPLIER_SIGNER_IDENTITY: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}\n        run: |",
            "          VISION_SUPPLIER_SIGNER_IDENTITY: ${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}\n          VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n        run: |",
        ),
        (
            "second-secret-expression",
            "          VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n",
            "          VISION_SUPPLIER_PRIVATE_KEY_PEM: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n          DUPLICATE_KEY: ${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}\n",
        ),
        (
            "wrong-unsigned-artifact-binding",
            "UNSIGNED_EVIDENCE_ARTIFACT_NAME: ${{ needs.trusted_signer.outputs.unsigned_evidence_artifact_name }}",
            "UNSIGNED_EVIDENCE_ARTIFACT_NAME: ${{ needs.verify.outputs.artifact_name }}",
        ),
        (
            "wrong-unsigned-digest-binding",
            "UNSIGNED_EVIDENCE_SHA256: ${{ needs.trusted_signer.outputs.unsigned_evidence_sha256 }}",
            "UNSIGNED_EVIDENCE_SHA256: ${{ needs.verify.outputs.subject_sha256 }}",
        ),
        (
            "immutable-signer-ref-drift",
            f"ref: {TRUSTED_SIGNER_COMMIT}",
            "ref: 0000000000000000000000000000000000000000",
        ),
        (
            "direct-sign-extra-permission",
            "    permissions:\n      contents: read\n    outputs:\n      signed_evidence_artifact_name:",
            "    permissions:\n      contents: read\n      id-token: write\n    outputs:\n      signed_evidence_artifact_name:",
        ),
        (
            "direct-sign-secrets-inherit",
            "    permissions:\n      contents: read\n    outputs:\n      signed_evidence_artifact_name:",
            "    permissions:\n      contents: read\n    secrets: inherit\n    outputs:\n      signed_evidence_artifact_name:",
        ),
        (
            "bracket-secret-in-seal-step",
            "        id: seal\n        shell: pwsh",
            "        id: seal\n        env:\n          LATE_KEY: ${{ secrets['VISION_SUPPLIER_PRIVATE_KEY_PEM'] }}\n        shell: pwsh",
        ),
    ),
)
def test_trust_policy_rejects_direct_signing_boundary_mutations(tmp_path, name, old, new):
    source = PUBLISHER.read_text("utf-8")
    if name == "extra-secret-in-verifier":
        start = source.index("  verify:\n")
        end = source.index("  trusted_signer:\n")
        candidate_source = source[:start] + source[start:end].replace(old, new, 1) + source[end:]
    else:
        start = source.index("  sign_evidence:\n")
        end = source.index("  publish:\n")
        candidate_source = source[:start] + source[start:end].replace(old, new, 1) + source[end:]
    candidate = tmp_path / f"{name}.yml"
    candidate.write_text(candidate_source, "utf-8")

    completed = _check_policy(candidate)

    assert completed.returncode != 0, name


@pytest.mark.parametrize(
    ("name", "old", "new"),
    (
        ("missing-finally", "          } finally {\n", "          } catch {\n"),
        ("silent-cleanup", "Remove-Item -LiteralPath $key -Force -ErrorAction Stop", "Remove-Item -LiteralPath $key -Force -ErrorAction SilentlyContinue"),
        ("wrong-key-cleanup-path", "Remove-Item -LiteralPath $key", "Remove-Item -LiteralPath (Join-Path $env:RUNNER_TEMP 'wrong.pem')"),
    ),
)
def test_trust_policy_rejects_direct_signing_key_cleanup_mutations(tmp_path, name, old, new):
    source = PUBLISHER.read_text("utf-8")
    start = source.index("  sign_evidence:\n")
    end = source.index("  publish:\n")
    direct = source[start:end]
    candidate_direct = direct.replace(old, new, 1)
    candidate = tmp_path / f"{name}.yml"
    candidate.write_text(source[:start] + candidate_direct + source[end:], "utf-8")

    completed = _check_policy(candidate)

    assert completed.returncode != 0, name


@pytest.mark.parametrize(
    ("primary", "cleanup", "expected"),
    (
        (True, False, ("PRIMARY",)),
        (False, True, ("SECONDARY",)),
        (True, True, ("PRIMARY", "WARNING: supplier key cleanup failed after signing failure: SECONDARY")),
    ),
)
def test_direct_sign_cleanup_preserves_primary_failure_in_real_pwsh(primary, cleanup, expected):
    script = f"""
$WarningPreference = 'Continue'
$primaryFailure = $null
$cleanupFailure = $null
try {{
  if (${str(primary).lower()}) {{ throw 'PRIMARY' }}
}} catch {{
  $primaryFailure = $_
}} finally {{
  try {{
    if (${str(cleanup).lower()}) {{ throw 'SECONDARY' }}
  }} catch {{
    $cleanupFailure = $_
  }}
}}
if ($null -ne $primaryFailure) {{
  if ($null -ne $cleanupFailure) {{ Write-Warning "supplier key cleanup failed after signing failure: $($cleanupFailure.Exception.Message)" }}
  throw $primaryFailure
}}
if ($null -ne $cleanupFailure) {{ throw $cleanupFailure }}
"""
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    for item in expected:
        assert item in output
    if primary and cleanup:
        assert output.index("WARNING: supplier key cleanup failed after signing failure: SECONDARY") < output.rindex("PRIMARY")
        assert "ErrorRecord" not in output


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
