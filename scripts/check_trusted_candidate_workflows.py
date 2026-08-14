from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

if __package__:
    from scripts.workflow_yaml import WorkflowYamlError, load_workflow_yaml, workflow_run_scalars
else:
    from workflow_yaml import WorkflowYamlError, load_workflow_yaml, workflow_run_scalars


TRUSTED_REPOSITORY = "hbhjt/vending-vision"
TRUSTED_BUILDER_COMMIT = "691b5056e8b9bf2667bc527b2170780b05863946"
TRUSTED_BUILDER_PATH = ".github/workflows/trusted-ai-candidate-builder.yml"
TRUSTED_SIGNER_COMMIT = "59b4fee088db08f3008c137409f98577de987595"
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
    "signer_identity",
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


def _workflow_jobs(source: str, label: str) -> dict[str, object]:
    try:
        workflow = load_workflow_yaml(source)
    except WorkflowYamlError as error:
        raise TrustPolicyError(f"{label}_yaml_invalid") from error
    jobs = workflow.get("jobs")
    _require(isinstance(jobs, dict), f"{label}_jobs_missing")
    return jobs


def _job_steps(job: object, label: str) -> dict[str, dict[str, object]]:
    _require(isinstance(job, dict), f"{label}_job_invalid")
    steps = job.get("steps")
    _require(isinstance(steps, list), f"{label}_steps_missing")
    named: dict[str, dict[str, object]] = {}
    for step in steps:
        _require(isinstance(step, dict), f"{label}_step_invalid")
        name = step.get("name")
        _require(isinstance(name, str) and name not in named, f"{label}_step_name_invalid")
        named[name] = step
    return named


def _step_env(step: object, label: str) -> dict[str, object]:
    _require(isinstance(step, dict), f"{label}_step_invalid")
    env = step.get("env")
    _require(isinstance(env, dict), f"{label}_env_missing")
    return env


def _assert_signer_identity_channel(signer_source: str, publisher_source: str) -> None:
    signer_jobs = _workflow_jobs(signer_source, "trusted_signer")
    verify = signer_jobs.get("verify_evidence")
    _require(isinstance(verify, dict), "trusted_signer_verify_job_missing")
    _require(set(signer_jobs) == {"verify_evidence"}, "trusted_signer_verify_only")
    _require("environment" not in verify, "trusted_signer_verify_environment")
    verify_steps = _job_steps(verify, "trusted_signer_verify")
    env = _step_env(verify_steps.get("Validate identities without shell interpolation"), "trusted_signer_verify")
    _require(
        env.get("VISION_SUPPLIER_SIGNER_IDENTITY") == "${{ inputs.signer_identity }}",
        "trusted_signer_verify_signer_identity_input",
    )
    _require(
        "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in signer_source,
        "trusted_signer_verify_secret",
    )

    publisher_jobs = _workflow_jobs(publisher_source, "publisher")
    signer_call = publisher_jobs.get("trusted_signer")
    _require(isinstance(signer_call, dict), "publisher_signer_job_missing")
    signer_with = signer_call.get("with")
    _require(isinstance(signer_with, dict), "publisher_signer_inputs_missing")
    _require(
        signer_with.get("signer_identity") == "${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}",
        "publisher_signer_identity_repository_var",
    )
    direct_sign = publisher_jobs.get("sign_evidence")
    _require(isinstance(direct_sign, dict), "publisher_direct_sign_job_missing")
    _require(direct_sign.get("environment") == "experimental-candidate", "publisher_direct_sign_environment")
    _require(direct_sign.get("permissions") == {"contents": "read"}, "publisher_direct_sign_permissions")
    _require("secrets" not in direct_sign, "publisher_direct_sign_job_secrets")
    direct_steps = _job_steps(direct_sign, "publisher_direct_sign")
    identities = _step_env(direct_steps.get("Validate direct signing identities without shell interpolation"), "publisher_direct_sign_identity")
    _require(
        identities.get("VISION_SUPPLIER_SIGNER_IDENTITY") == "${{ vars.VISION_SUPPLIER_SIGNER_IDENTITY }}",
        "publisher_direct_signer_identity_repository_var",
    )
    key_step = _step_env(direct_steps.get("Sign only verified evidence with the protected supplier key"), "publisher_direct_sign_key")
    _require(
        key_step == {"VISION_SUPPLIER_PRIVATE_KEY_PEM": "${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}"},
        "publisher_direct_sign_key_scope",
    )
    for step_name, step in direct_steps.items():
        env = step.get("env")
        if env is None:
            continue
        _require(isinstance(env, dict), "publisher_direct_sign_step_env_invalid")
        if step_name == "Sign only verified evidence with the protected supplier key":
            continue
        for value in env.values():
            rendered = str(value)
            _require(
                not re.search(r"(?i)\bsecrets\s*(?:\.|\[)|PRIVATE_KEY", rendered),
                "publisher_direct_sign_secret_outside_sign_step",
            )
    _require(
        publisher_source.count("${{ secrets.VISION_SUPPLIER_PRIVATE_KEY_PEM }}") == 1,
        "publisher_direct_sign_key_unique",
    )
    _require(
        "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in publisher_source.replace(_job_block(publisher_source, "sign_evidence"), ""),
        "publisher_direct_sign_key_outside_boundary",
    )


def _assert_no_untrusted_run_expressions(source: str, label: str) -> None:
    blocks = workflow_run_scalars(source)
    _require(bool(blocks), f"{label}_run_blocks_missing")
    for block in blocks:
        _require("${{" not in block, f"{label}_workflow_expression_in_run")


_PYTHON_EXECUTABLE = re.compile(r"python(?:3(?:\.\d+)?)?(?:\.exe)?", re.IGNORECASE)


def _shell_dialect(shell: object) -> str:
    if not isinstance(shell, str):
        raise TrustPolicyError("archive_downloader_shell_unknown")
    normalized = shell.lower()
    if normalized in {"pwsh", "powershell"}:
        return "pwsh"
    if normalized in {"bash", "sh"}:
        return "bash"
    raise TrustPolicyError("archive_downloader_shell_unknown")


def _is_shell_escape(character: str, following: str | None, dialect: str) -> bool:
    if dialect == "pwsh":
        return character == "`" and following is not None
    return character == "\\" and following in {" ", "\t", "#", ";", "|", "&", "'", '"', "\\"}


def _strip_shell_comment(line: str, dialect: str) -> str:
    quote = ""
    escaped = False
    for index, character in enumerate(line):
        if quote == "'":
            if character == "'":
                quote = ""
            continue
        if escaped:
            escaped = False
            index += 1
            continue
        if _is_shell_escape(character, line[index + 1] if index + 1 < len(line) else None, dialect):
            escaped = True
            continue
        if character in {"'", '"'}:
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            continue
        if character == "#" and not quote and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def _pwsh_preprocess_line(
    line: str,
    *,
    in_block_comment: bool,
    quote: str,
    escaped: bool,
) -> tuple[str, bool, str, bool, str, bool]:
    """Return executable PowerShell text and persistent lexical state for one line."""
    output: list[str] = []
    spans_physical_line = bool(quote)
    index = 0
    while index < len(line):
        character = line[index]
        following = line[index + 1] if index + 1 < len(line) else None
        if in_block_comment:
            if character == "#" and following == ">":
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote and escaped:
            output.append(character)
            escaped = False
            index += 1
            continue
        if quote:
            output.append(character)
            if quote == '"' and character == "`":
                if following is None:
                    escaped = True
                    index += 1
                else:
                    output.append(following)
                    index += 2
                continue
            if quote == "'" and character == "'" and following == "'":
                output.append(following)
                index += 2
                continue
            if character == quote:
                quote = ""
            index += 1
            continue
        if escaped:
            output.append(character)
            escaped = False
            index += 1
            continue
        if _is_shell_escape(character, following, "pwsh"):
            output.append(character)
            escaped = True
            index += 1
            continue
        if character in {"'", '"'}:
            output.append(character)
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            index += 1
            continue
        if character == "@" and following in {"'", '"'}:
            return "", in_block_comment, quote, escaped, following, True
        if character == "<" and following == "#":
            in_block_comment = True
            index += 2
            continue
        if character == "#" and (index == 0 or line[index - 1].isspace()):
            break
        output.append(character)
        index += 1
    return (
        "".join(output).rstrip(),
        in_block_comment,
        quote,
        escaped,
        "",
        spans_physical_line or bool(quote),
    )


def _logical_shell_commands(run: str, dialect: str) -> tuple[str, ...]:
    """Return executable shell statements, excluding comment-only source lines."""
    commands: list[str] = []
    pending = ""
    here_string_quote = ""
    block_comment = False
    quote = ""
    escaped = False
    for raw in run.splitlines():
        stripped = raw.strip()
        if here_string_quote:
            if stripped == f"{here_string_quote}@":
                here_string_quote = ""
            continue
        if dialect == "pwsh":
            line, block_comment, quote, escaped, here_string_quote, quoted_line = (
                _pwsh_preprocess_line(
                    stripped,
                    in_block_comment=block_comment,
                    quote=quote,
                    escaped=escaped,
                )
            )
            if here_string_quote:
                continue
            if quoted_line:
                continue
        else:
            line = _strip_shell_comment(stripped, dialect)
        if not line:
            continue
        continued = line.endswith("`" if dialect == "pwsh" else "\\")
        pending += (line[:-1] if continued else line) + " "
        if not continued:
            commands.append(pending.strip())
            pending = ""
    if here_string_quote:
        raise TrustPolicyError("archive_downloader_unterminated_here_string")
    if block_comment:
        raise TrustPolicyError("archive_downloader_unterminated_block_comment")
    if quote:
        raise TrustPolicyError("archive_downloader_unterminated_quote")
    if pending:
        commands.append(pending.strip())
    return tuple(commands)


def _shell_statements(command: str, dialect: str) -> tuple[tuple[str, bool], ...]:
    """Split executable shell statements without treating quoted/escaped operators as syntax."""
    statements: list[tuple[str, bool]] = []
    current: list[str] = []
    call_operator = False
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if quote == "'":
            current.append(character)
            if character == "'":
                quote = ""
            index += 1
            continue
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if _is_shell_escape(character, command[index + 1] if index + 1 < len(command) else None, dialect):
            current.append(character)
            escaped = True
            index += 1
            continue
        if character in {"'", '"'}:
            current.append(character)
            if not quote:
                quote = character
            elif quote == character:
                quote = ""
            index += 1
            continue
        if character == "#" and not quote and (not current or current[-1].isspace()):
            break
        if not quote and character in {";", "|", "&"}:
            if (
                character == "&"
                and not "".join(current).strip()
                and not (index + 1 < len(command) and command[index + 1] == "&")
            ):
                call_operator = True
                index += 1
                continue
            statement = "".join(current).strip()
            if statement:
                statements.append((statement, call_operator))
            current = []
            call_operator = False
            if index + 1 < len(command) and command[index + 1] == character and character in {"|", "&"}:
                index += 1
            index += 1
            continue
        current.append(character)
        index += 1
    statement = "".join(current).strip()
    if statement:
        statements.append((statement, call_operator))
    return tuple(statements)


def _shell_tokens(statement: str, dialect: str) -> tuple[tuple[str, bool], ...]:
    tokens: list[tuple[str, bool]] = []
    current: list[str] = []
    quote = ""
    quoted = False
    escaped = False
    index = 0
    while index < len(statement):
        character = statement[index]
        if quote == "'":
            if character == "'":
                quote = ""
            else:
                current.append(character)
            index += 1
            continue
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if _is_shell_escape(character, statement[index + 1] if index + 1 < len(statement) else None, dialect):
            escaped = True
            index += 1
            continue
        if character in {"'", '"'}:
            if not quote:
                quote = character
                quoted = True
            elif quote == character:
                quote = ""
            else:
                current.append(character)
            index += 1
            continue
        if character.isspace() and not quote:
            if current:
                tokens.append(("".join(current), quoted))
                current = []
                quoted = False
            index += 1
            continue
        current.append(character)
        index += 1
    if current:
        tokens.append(("".join(current), quoted))
    return tuple(tokens)


def _normalized_basename(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1]


def _archive_downloader_timeout_values(
    statement: str, dialect: str, call_operator: bool
) -> tuple[str, ...] | None:
    tokens = _shell_tokens(statement, dialect)
    has_downloader = any(
        _normalized_basename(token) == "download_verified_archive.py" for token, _ in tokens
    )
    if not has_downloader:
        return None
    executable_index = 0
    script_index = executable_index + 1
    if (
        len(tokens) <= script_index
        or (call_operator and dialect != "pwsh")
        or _PYTHON_EXECUTABLE.fullmatch(_normalized_basename(tokens[executable_index][0])) is None
        or _normalized_basename(tokens[script_index][0]) != "download_verified_archive.py"
        or (dialect == "pwsh" and tokens[executable_index][1] and not call_operator)
    ):
        raise TrustPolicyError("archive_downloader_unsupported_invocation")
    values: list[str] = []
    for index, (token, _) in enumerate(tokens):
        if token == "--total-timeout-seconds":
            if index + 1 < len(tokens):
                values.append(tokens[index + 1][0])
            continue
        if token.startswith("--total-timeout-seconds="):
            values.append(token.removeprefix("--total-timeout-seconds="))
    return tuple(values)


def _workflow_run_steps(source: str) -> tuple[tuple[str, object], ...]:
    workflow = load_workflow_yaml(source)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise WorkflowYamlError("jobs_not_mapping")
    steps_with_shell: list[tuple[str, object]] = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            raise WorkflowYamlError(f"job_not_mapping:{job_name}")
        steps = job.get("steps")
        if steps is None:
            continue
        if not isinstance(steps, list):
            raise WorkflowYamlError(f"steps_not_sequence:{job_name}")
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise WorkflowYamlError(f"step_not_mapping:{job_name}:{index}")
            run = step.get("run")
            if run is None:
                continue
            if not isinstance(run, str):
                raise WorkflowYamlError(f"run_not_string:{job_name}:{index}")
            steps_with_shell.append((run, step.get("shell")))
    return tuple(steps_with_shell)


def _archive_downloader_timeouts(source: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    invocations: list[tuple[str, tuple[str, ...]]] = []
    for run, shell in _workflow_run_steps(source):
        if "download_verified_archive.py" not in run:
            continue
        dialect = _shell_dialect(shell)
        for command in _logical_shell_commands(run, dialect):
            for statement, call_operator in _shell_statements(command, dialect):
                values = _archive_downloader_timeout_values(statement, dialect, call_operator)
                if values is not None:
                    invocations.append((statement, values))
    return tuple(invocations)


def _require_bounded_archive_downloads(source: str, label: str, expected_count: int) -> None:
    invocations = _archive_downloader_timeouts(source)
    _require(len(invocations) == expected_count, f"{label}_archive_downloader_count")
    for _, values in invocations:
        _require(len(values) == 1, f"{label}_archive_downloader_timeout_count")
        try:
            timeout = int(values[0])
        except ValueError:
            raise TrustPolicyError(f"{label}_archive_downloader_timeout_numeric") from None
        _require(0 < timeout <= 3600, f"{label}_archive_downloader_timeout_bounds")


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
    _require_bounded_archive_downloads(builder_source, "trusted_builder", 1)
    _require_bounded_archive_downloads(publisher_source, "publisher", 1)
    active_archive_invocations = []
    workflow_directory = repository_root / ".github" / "workflows"
    for workflow in sorted(workflow_directory.glob("*.yml")):
        source = {
            TRUSTED_BUILDER_PATH: builder_source,
            TRUSTED_SIGNER_PATH: signer_source,
            ".github/workflows/publish-candidate.yml": publisher_source,
        }.get(workflow.relative_to(repository_root).as_posix(), workflow.read_text("utf-8"))
        active_archive_invocations.extend(_archive_downloader_timeouts(source))
    _require(
        len(active_archive_invocations) == 3,
        "active_archive_downloader_count",
    )
    for _, values in active_archive_invocations:
        _require(len(values) == 1, "active_archive_downloader_timeout_count")
        try:
            timeout = int(values[0])
        except ValueError:
            raise TrustPolicyError("active_archive_downloader_timeout_numeric") from None
        _require(0 < timeout <= 3600, "active_archive_downloader_timeout_bounds")
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
    _assert_signer_identity_channel(signer_source, publisher_source)
    for forbidden in (
        "source_path", "artifact_path", "custom_command", "predicate", "private_key",
    ):
        _require(forbidden not in signer_source, f"trusted_signer_forbidden_input:{forbidden}")
    verify_evidence = _job_block(signer_source, "verify_evidence")
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
    _require("sign_evidence:" not in signer_source, "trusted_signer_sign_job_present")
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in signer_source, "trusted_signer_key_present")
    _require("sign_candidate_evidence.py" not in signer_source, "trusted_signer_sign_script_present")
    _require(signer_source.count("actions/checkout@v4") == 1, "trusted_signer_checkout_count")
    for forbidden in ("path: source", "actions/setup-python", "source/scripts", ".spec"):
        _require(forbidden not in signer_source, f"trusted_signer_candidate_execution:{forbidden}")

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

    direct_sign = _job_block(publisher_source, "sign_evidence")
    _require("needs: [verify, trusted_signer]" in direct_sign, "publisher_direct_sign_requires_verified_evidence")
    _require("runs-on: windows-latest" in direct_sign, "publisher_direct_sign_runner")
    _require("environment: experimental-candidate" in direct_sign, "publisher_direct_sign_environment")
    for fragment in (
        f"ref: {TRUSTED_SIGNER_COMMIT}",
        "path: trusted-signer",
        "trusted-signer/scripts/verify_trusted_script_set.py",
        "trusted-signer/scripts/evidence_artifact.py",
        "trusted-signer/scripts/sign_candidate_evidence.py",
        "--kind unsigned --expected-digest $env:UNSIGNED_EVIDENCE_SHA256",
        "--private-key $key --signer-identity $env:VISION_SUPPLIER_SIGNER_IDENTITY --openssl $env:TRUSTED_OPENSSL",
        "VISION_SUPPLIER_PRIVATE_KEY_PEM",
        "kind signed",
        "Upload only direct supplier-signed evidence",
    ):
        _require(fragment in direct_sign, f"publisher_direct_sign_policy:{fragment}")
    key_step = re.search(r"(?ms)^      - name: Sign only verified evidence.*?(?=^      - name:)", direct_sign)
    _require(key_step is not None, "publisher_direct_sign_key_step_missing")
    assert key_step is not None
    _require("VISION_SUPPLIER_PRIVATE_KEY_PEM" not in direct_sign[: key_step.start()], "publisher_direct_sign_key_exposed_early")
    for fragment in (
        "$primaryFailure = $null",
        "$cleanupFailure = $null",
        "} finally {",
        "Remove-Item -LiteralPath $key -Force -ErrorAction Stop",
        "Test-Path -LiteralPath $key -PathType Leaf",
        "throw $primaryFailure",
        "throw $cleanupFailure",
        "Write-Warning \"supplier key cleanup failed after signing failure:",
    ):
        _require(fragment in key_step.group(0), f"publisher_direct_sign_cleanup:{fragment}")
    _require("SilentlyContinue" not in key_step.group(0), "publisher_direct_sign_cleanup_silent")
    _require("Write-Error" not in key_step.group(0), "publisher_direct_sign_cleanup_overrides_primary")
    _require(
        key_step.group(0).index("throw $primaryFailure") < key_step.group(0).index("throw $cleanupFailure"),
        "publisher_direct_sign_cleanup_primary_precedes_secondary",
    )

    direct_steps = _job_steps(_workflow_jobs(publisher_source, "publisher").get("sign_evidence"), "publisher_direct_sign")
    identities = _step_env(direct_steps.get("Validate direct signing identities without shell interpolation"), "publisher_direct_sign_identity")
    _require(
        identities.get("UNSIGNED_EVIDENCE_ARTIFACT_NAME") == "${{ needs.trusted_signer.outputs.unsigned_evidence_artifact_name }}",
        "publisher_direct_sign_unsigned_artifact_binding",
    )
    _require(
        identities.get("UNSIGNED_EVIDENCE_SHA256") == "${{ needs.trusted_signer.outputs.unsigned_evidence_sha256 }}",
        "publisher_direct_sign_unsigned_digest_binding",
    )
    checkout = direct_steps.get("Checkout immutable trusted signing implementation")
    _require(isinstance(checkout, dict), "publisher_direct_sign_checkout_missing")
    checkout_with = checkout.get("with")
    _require(isinstance(checkout_with, dict), "publisher_direct_sign_checkout_inputs")
    _require(checkout_with.get("repository") == TRUSTED_REPOSITORY, "publisher_direct_sign_checkout_repository")
    _require(checkout_with.get("ref") == TRUSTED_SIGNER_COMMIT, "publisher_direct_sign_checkout_ref")
    _require(checkout_with.get("path") == "trusted-signer", "publisher_direct_sign_checkout_path")

    publish = _job_block(publisher_source, "publish")
    _require("needs: [verify, sign_evidence]" in publish, "publisher_requires_direct_sign")
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
        "--repo hbhjt/vending-vision",
        "--target $env:RELEASE_TARGET",
        "--verify-tag",
    ):
        _require(fragment in publish, f"publisher_release_authority:{fragment}")
    publish_steps = _job_steps(
        _workflow_jobs(publisher_source, "publisher").get("publish"),
        "publisher_publish",
    )
    publish_step = publish_steps.get("Publish immutable prerelease only after trusted signing")
    _require(isinstance(publish_step, dict), "publisher_release_step_missing")
    release_run = publish_step.get("run")
    _require(isinstance(release_run, str), "publisher_release_step_run_missing")
    release_commands = [
        line.strip()
        for line in release_run.splitlines()
        if line.strip().startswith("gh release create ")
    ]
    _require(len(release_commands) == 1, "publisher_release_cli_count")
    _require(
        release_commands[0].startswith(
            "gh release create $env:RELEASE_TAG release/* --repo hbhjt/vending-vision "
            "--target $env:RELEASE_TARGET --prerelease --verify-tag "
        ),
        "publisher_release_cli_exact_repository_context",
    )
    _require(publish.count("actions/checkout@v4") == 2, "publisher_trusted_policy_checkout")
    for forbidden in (
        "generate_candidate_evidence.py", "sign_candidate_evidence.py",
    ):
        _require(forbidden not in publish, f"publisher_forbidden_capability:{forbidden}")
    _require("rulesets?targets=tag" not in publish, "publisher_unavailable_rulesets_api")
    _require(publish.count("environment: experimental-candidate") == 1, "publisher_environment_authority")
    _assert_gh_attestation_flags_parse(repository_root)


def _assert_exact4_publisher(
    *, builder: Path, publisher: Path, repository_root: Path
) -> None:
    """Fail closed unless the publisher consumes and releases exactly builder exact4."""
    builder_source = builder.read_text("utf-8")
    publisher_source = publisher.read_text("utf-8")
    _assert_files_are_immutable(
        commit=TRUSTED_BUILDER_COMMIT,
        paths={TRUSTED_BUILDER_PATH: builder},
        repository_root=repository_root,
        label="trusted_builder",
    )
    _require(
        _workflow_call_inputs(builder_source) == BUILDER_INPUTS,
        "trusted_builder_input_allowlist",
    )
    for required in (
        "Run the full packaged candidate verifier before signing",
        "--require-ai-worker",
        "actions/attest-build-provenance@v4",
        "Record trusted builder evidence",
        "trusted-builder-evidence.json",
        "id-token: write",
        "attestations: write",
    ):
        _require(required in builder_source, f"trusted_builder_evidence_policy:{required}")
    _require(
        builder_source.index("--require-ai-worker")
        < builder_source.index("actions/attest-build-provenance@v4")
        < builder_source.index("Record trusted builder evidence"),
        "trusted_builder_full_verifier_order",
    )

    publisher_jobs = _workflow_jobs(publisher_source, "publisher")
    _require(set(publisher_jobs) == {"trusted_builder", "publish"}, "publisher_jobs_exact4")
    trusted_builder = publisher_jobs.get("trusted_builder")
    _require(isinstance(trusted_builder, dict), "publisher_builder_job_missing")
    trusted_call = f"uses: {TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}@{TRUSTED_BUILDER_COMMIT}"
    _require(publisher_source.count(trusted_call) == 1, "publisher_literal_builder_pin")
    _require(trusted_builder.get("uses") == f"{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}@{TRUSTED_BUILDER_COMMIT}", "publisher_builder_job_pin")
    caller_inputs = trusted_builder.get("with")
    _require(isinstance(caller_inputs, dict), "publisher_builder_inputs_missing")
    _require(set(caller_inputs) == BUILDER_INPUTS, "publisher_builder_input_allowlist")

    publish = publisher_jobs.get("publish")
    _require(isinstance(publish, dict), "publisher_publish_job_missing")
    _require(publish.get("needs") == "trusted_builder", "publisher_requires_builder")
    _require(publish.get("environment") == "production", "publisher_production_environment")
    _require(publisher_source.count("environment:") == 1, "publisher_environment_exactly_one")
    _require(publish.get("permissions") == {"contents": "write", "attestations": "read"}, "publisher_permissions_exact4")

    for forbidden in (
        "trusted-ai-candidate-signer.yml",
        "sign_evidence",
        "verify_evidence",
        "VISION_SUPPLIER_",
        "signed-evidence",
        "path: source",
        "refs/tags/",
        "refs/heads/main",
        "release/*",
        "approve_candidate_source.py",
        "verify_hosted_release_authority.py",
        "trusted-policy",
        "hosted-authority",
    ):
        _require(forbidden not in publisher_source, "publisher_forbidden_capability")
    _require(
        re.search(r"(?i)\bsecrets\s*(?:\.|\[)", publisher_source) is None,
        "publisher_forbidden_capability",
    )

    steps = _job_steps(publish, "publisher_publish")
    _require(set(steps) == {
        "Checkout immutable exact4 verifier",
        "Download exactly the trusted builder artifact",
        "Reverify exact4 trusted candidate inputs in production",
        "Publish exactly the reverified builder assets",
    }, "publisher_steps_exact4")
    checkout = steps["Checkout immutable exact4 verifier"].get("with")
    _require(isinstance(checkout, dict), "publisher_verifier_checkout_inputs")
    _require(checkout == {
        "repository": TRUSTED_REPOSITORY,
        "ref": TRUSTED_BUILDER_COMMIT,
        "path": "trusted-builder",
        "persist-credentials": "false",
    }, "publisher_verifier_checkout_pin")
    download = steps["Download exactly the trusted builder artifact"].get("with")
    _require(isinstance(download, dict), "publisher_download_inputs")
    _require(download == {
        "name": "${{ needs.trusted_builder.outputs.artifact_name }}",
        "path": "candidate-input",
    }, "publisher_download_builder_artifact")
    _require(publisher_source.count("actions/download-artifact@v4") == 1, "publisher_download_count")

    reverify = steps["Reverify exact4 trusted candidate inputs in production"]
    env = _step_env(reverify, "publisher_reverify")
    _require(env == {
        "GH_TOKEN": "${{ github.token }}",
        "ARTIFACT_FILE": "${{ needs.trusted_builder.outputs.artifact_file }}",
        "SUBJECT_SHA256": "${{ needs.trusted_builder.outputs.subject_sha256 }}",
        "MANIFEST_SHA256": "${{ needs.trusted_builder.outputs.manifest_sha256 }}",
        "ATTESTATION_BUNDLE_SHA256": "${{ needs.trusted_builder.outputs.attestation_bundle_sha256 }}",
        "CANDIDATE_COMMIT": "${{ github.sha }}",
        "CANDIDATE_TAG": "${{ github.ref_name }}",
    }, "publisher_output_binding")
    reverify_run = reverify.get("run")
    _require(isinstance(reverify_run, str), "publisher_reverify_run")
    for fragment in (
        "gh attestation verify $artifact",
        f'--repo "{TRUSTED_REPOSITORY}"',
        f'--signer-workflow "{TRUSTED_REPOSITORY}/{TRUSTED_BUILDER_PATH}"',
        f'--signer-digest "{TRUSTED_BUILDER_COMMIT}"',
        "--deny-self-hosted-runners",
        "trusted-builder/scripts/verify_trusted_candidate_inputs.py",
        "--artifact $artifact",
        "--candidate-manifest (Join-Path $inputRoot \"candidate-manifest.json\")",
        "--github-attestation $bundle",
        "--trusted-builder-evidence (Join-Path $inputRoot \"trusted-builder-evidence.json\")",
        "--subject-sha256 $env:SUBJECT_SHA256",
        "--manifest-sha256 $env:MANIFEST_SHA256",
        "--attestation-bundle-sha256 $env:ATTESTATION_BUNDLE_SHA256",
        "--source-commit $env:CANDIDATE_COMMIT",
    ):
        _require(fragment in reverify_run, "publisher_exact4_member")
    _require("--source-ref" not in reverify_run and "--source-digest" not in reverify_run, "publisher_source_authority_removed")
    _require(reverify_run.count("candidate-manifest.json") == 1, "publisher_exact4_member")
    _require(reverify_run.count("github-build-provenance.sigstore.json") == 1, "publisher_exact4_member")
    _require(reverify_run.count("trusted-builder-evidence.json") == 1, "publisher_exact4_member")

    release = steps["Publish exactly the reverified builder assets"]
    release_env = _step_env(release, "publisher_release")
    _require(release_env == {
        "GH_TOKEN": "${{ github.token }}",
        "CANDIDATE_TAG": "${{ github.ref_name }}",
        "ARTIFACT_FILE": "${{ needs.trusted_builder.outputs.artifact_file }}",
    }, "publisher_release_output_binding")
    release_run = release.get("run")
    _require(isinstance(release_run, str), "publisher_release_run")
    release_commands = [line.strip() for line in release_run.splitlines() if line.strip().startswith("gh release create ")]
    _require(len(release_commands) == 1, "publisher_release_cli_count")
    exact_release = (
        "gh release create $env:CANDIDATE_TAG \"candidate-input/$env:ARTIFACT_FILE\" "
        "\"candidate-input/candidate-manifest.json\" "
        "\"candidate-input/github-build-provenance.sigstore.json\" "
        "\"candidate-input/trusted-builder-evidence.json\" "
        "--repo hbhjt/vending-vision --prerelease --verify-tag "
    )
    _require(release_commands[0].startswith(exact_release), "publisher_release_assets_exact4")
    _require("*" not in release_commands[0], "publisher_release_assets_exact4")
    _require(release_run.count("candidate-manifest.json") == 1, "publisher_release_assets_exact4")
    _require(release_run.count("github-build-provenance.sigstore.json") == 1, "publisher_release_assets_exact4")
    _require(release_run.count("trusted-builder-evidence.json") == 1, "publisher_release_assets_exact4")
    _assert_no_untrusted_run_expressions(builder_source, "trusted_builder")
    _assert_no_untrusted_run_expressions(publisher_source, "publisher")
    _assert_gh_attestation_flags_parse(repository_root)


def check_trusted_candidate_workflows(
    *, builder: Path, publisher: Path, repository_root: Path
) -> None:
    _assert_exact4_publisher(
        builder=builder, publisher=publisher, repository_root=repository_root
    )


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
    except (OSError, StopIteration, TrustPolicyError, WorkflowYamlError) as exc:
        print(f"TRUSTED_CANDIDATE_WORKFLOW_POLICY=FAIL:{exc}")
        return 1
    print("TRUSTED_CANDIDATE_WORKFLOW_POLICY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
