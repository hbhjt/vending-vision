from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from scripts.approve_candidate_source import approve_source
from scripts.verify_trusted_builder_closure import ClosureError, verify as verify_closure
from scripts.verify_release_tag_ruleset import verify_rulesets
from scripts.workflow_yaml import load_workflow_yaml, workflow_run_scalars


ROOT = Path(__file__).parents[1]
TRUSTED_PROOF = (
    ROOT / ".github" / "workflows" / "trusted-precutover-companion-proof.yml"
)
COMPANION_BUILDER_COMMIT = "d8c93b50cac005a371d06badcc398638fd8acabb"
BUILDER_CLOSURE = ROOT / "trusted-precutover-companion-builder-closure.json"
BUILDER_CLOSURE_VERIFIER = ROOT / "scripts/verify_trusted_builder_closure.py"
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
MODEL_PART_INPUTS = {
    f"model_pack_part_{index:02d}_{field}"
    for index in range(1, 4)
    for field in ("url", "sha256", "bytes")
}
INPUTS |= MODEL_PART_INPUTS


def test_trusted_windows_companion_proof_workflow_exists():
    assert TRUSTED_PROOF.is_file(), (
        "no trusted Windows workflow executes the frozen companion and attests "
        "its canonical proof"
    )


def test_trusted_builder_requires_exact_canonical_source_closure_before_download():
    builder = (
        ROOT / ".github/workflows/trusted-precutover-companion-builder.yml"
    ).read_text("utf-8")
    assert BUILDER_CLOSURE.is_file(), "trusted builder has no exact source closure"
    assert BUILDER_CLOSURE_VERIFIER.is_file(), "trusted builder has no closure verifier"
    verify = "trusted-companion/scripts/verify_trusted_builder_closure.py"
    assert verify in builder
    assert builder.index(verify) < builder.index("download_verified_archive.py")
    assert "archive_extractor_worker.py" in BUILDER_CLOSURE.read_text("utf-8")


def _copy_builder_closure(tmp_path: Path) -> tuple[Path, Path, dict]:
    manifest = json.loads(BUILDER_CLOSURE.read_text("utf-8"))
    root = tmp_path / "source"
    for item in manifest["files"]:
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["path"], target)
    manifest_path = root / BUILDER_CLOSURE.name
    manifest_path.write_bytes(BUILDER_CLOSURE.read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root, manifest_path, manifest


@pytest.mark.parametrize(
    "path",
    json.loads(BUILDER_CLOSURE.read_text("utf-8"))["files"],
    ids=lambda item: item["path"],
)
def test_trusted_builder_closure_rejects_every_direct_file_mutation(tmp_path, path):
    root, manifest_path, _manifest = _copy_builder_closure(tmp_path)
    candidate = root / path["path"]
    candidate.write_bytes(candidate.read_bytes() + b"\nmutation")
    with pytest.raises(ClosureError, match="closure_digest"):
        verify_closure(root, manifest_path)


@pytest.mark.parametrize("mutation", ["missing-worker", "extra", "reorder", "pretty"])
def test_trusted_builder_closure_rejects_noncanonical_or_changed_file_set(
    tmp_path, mutation
):
    root, manifest_path, manifest = _copy_builder_closure(tmp_path)
    if mutation == "missing-worker":
        manifest["files"] = [
            item
            for item in manifest["files"]
            if item["path"] != "scripts/archive_extractor_worker.py"
        ]
    elif mutation == "extra":
        manifest["files"].append({"path": "extra.txt", "sha256": "0" * 64})
    elif mutation == "reorder":
        manifest["files"][0], manifest["files"][1] = (
            manifest["files"][1],
            manifest["files"][0],
        )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
        with pytest.raises(ClosureError, match="noncanonical"):
            verify_closure(root, manifest_path)
        return
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
    )
    with pytest.raises(ClosureError, match="file_set"):
        verify_closure(root, manifest_path)


def test_trusted_builder_closure_rejects_new_spec_hidden_import(tmp_path):
    root, manifest_path, manifest = _copy_builder_closure(tmp_path)
    spec = root / "vending_vision_precutover_verifier.spec"
    spec.write_text(spec.read_text("utf-8") + '\nhiddenimports += ["vision.hidden"]\n', "utf-8")
    for item in manifest["files"]:
        if item["path"] == spec.name:
            item["sha256"] = __import__("hashlib").sha256(spec.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
    )
    with pytest.raises(ClosureError, match="spec_(?:ast|hiddenimports)"):
        verify_closure(root, manifest_path)


def test_builder_closure_rejects_new_tracked_local_import(tmp_path):
    export, manifest_path, manifest = _copy_builder_closure(tmp_path)
    unlisted = export / "vision/unlisted_local.py"
    unlisted.write_text("VALUE = 1\n", "utf-8")
    ai_model = export / "vision/ai_model_pack.py"
    ai_model.write_text(
        ai_model.read_text("utf-8") + "\nimport vision.unlisted_local\n", "utf-8"
    )
    for item in manifest["files"]:
        if item["path"] == "vision/ai_model_pack.py":
            item["sha256"] = __import__("hashlib").sha256(ai_model.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
    )
    subprocess.run(["git", "add", "-A"], cwd=export, check=True)
    with pytest.raises(ClosureError, match="dependency:vision/unlisted_local.py"):
        verify_closure(export, manifest_path)


@pytest.mark.parametrize(
    ("target", "appendix"),
    [
        ("vision/ai_model_pack.py", "\nfrom vision import unlisted_local\n"),
        ("vision/ai_model_pack.py", "\nimport importlib\nimportlib.import_module('vision.unlisted_local')\n"),
        ("vision/ai_model_pack.py", "\nimport importlib\nimportlib.import_module('vision.' + 'unlisted_local')\n"),
        ("vision/ai_model_pack.py", "\nfrom importlib import import_module\nimport_module('vision.unlisted_local')\n"),
        ("vision/ai_model_pack.py", "\nloader = __import__\nloader('vision.unlisted_local')\n"),
        ("vision/ai_model_pack.py", "\n__import__('vision.' + 'unlisted_local')\n"),
        ("vending_vision_precutover_verifier.spec", "\ndatas.append(('vision/unlisted_local.py','vision'))\n"),
        ("vending_vision_precutover_verifier.spec", "\nhiddenimports.append('vision.' + 'unlisted_local')\n"),
        ("vending_vision_precutover_verifier.spec", "\nbinaries.append(('tool.exe','.'))\n"),
        ("vending_vision_precutover_verifier.spec", "\nruntime_hooks.append('vision/unlisted_local.py')\n"),
        ("vending_vision_precutover_verifier.spec", "\nextra_datas = datas\nextra_datas.append(('local.dat','.'))\n"),
        ("vending_vision_precutover_verifier.spec", "\nextra_hidden = hiddenimports\nextra_hidden.append('vision.unlisted_local')\n"),
        ("vending_vision_precutover_verifier.spec", "\nextra_bins = binaries\nextra_bins.append(('tool.exe','.'))\n"),
        ("vending_vision_precutover_verifier.spec", "\nextra_hooks = runtime_hooks\nextra_hooks.append('vision/unlisted_local.py')\n"),
        ("vending_vision_precutover_verifier.spec", "\nextra_paths = hookspath\nextra_paths.append('local-hooks')\n"),
    ],
)
def test_builder_closure_rejects_import_and_spec_mutation_bypasses(
    tmp_path, target, appendix
):
    root, manifest_path, manifest = _copy_builder_closure(tmp_path)
    unlisted = root / "vision/unlisted_local.py"
    unlisted.write_text("VALUE = 1\n", "utf-8")
    candidate = root / target
    candidate.write_text(candidate.read_text("utf-8") + appendix, "utf-8")
    for item in manifest["files"]:
        if item["path"] == target:
            item["sha256"] = __import__("hashlib").sha256(candidate.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", "utf-8"
    )
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    with pytest.raises(ClosureError):
        verify_closure(root, manifest_path)


def test_trusted_proof_has_closed_https_inputs_and_pins_companion_builder():
    source = TRUSTED_PROOF.read_text("utf-8")
    match = re.search(
        r"(?ms)^  workflow_call:\n    inputs:\n(?P<body>.*?)(?=^    outputs:)", source
    )
    assert match is not None
    assert set(re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", match.group("body"))) == INPUTS
    workflow = load_workflow_yaml(source)
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert inputs["model_pack_url"]["required"] == "false"
    assert inputs["model_pack_sha256"]["required"] == "true"
    assert inputs["model_pack_bytes"]["required"] == "true"
    for name in MODEL_PART_INPUTS:
        assert inputs[name]["required"] == "false"
    assert (
        "uses: hbhjt/vending-vision/.github/workflows/"
        f"trusted-precutover-companion-builder.yml@{COMPANION_BUILDER_COMMIT}"
    ) in source
    assert "runs-on: windows-latest" in source
    assert "secrets:" not in source
    for forbidden in ("path_input", "command", "predicate", "worker_path", "artifact_path"):
        assert forbidden not in source


def test_candidate_execution_job_has_no_oidc_or_attestation_write_capability():
    workflow = load_workflow_yaml(TRUSTED_PROOF.read_text("utf-8"))
    jobs = workflow["jobs"]
    execution = [
        (name, job)
        for name, job in jobs.items()
        if isinstance(job, dict)
        and any(
            "vending-vision-precutover-verifier.exe" in step.get("run", "")
            for step in job.get("steps", [])
            if isinstance(step, dict)
        )
    ]
    assert len(execution) == 1
    name, job = execution[0]
    assert name == "execute"
    permissions = job["permissions"]
    assert "id-token" not in permissions
    assert permissions.get("attestations") != "write"

    signing = jobs["sign"]
    assert signing["needs"] == ["companion_builder", "execute"]
    assert signing["permissions"]["id-token"] == "write"
    assert signing["permissions"]["attestations"] == "write"


def test_proof_jobs_and_downloads_have_fixed_total_deadlines():
    source = TRUSTED_PROOF.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    for name, expected in {"execute": 180, "sign": 180, "verify": 30}.items():
        timeout = workflow["jobs"][name].get("timeout-minutes")
        assert timeout == str(expected), name
        assert f"    timeout-minutes: {expected}\n" in source

    download_commands = [
        line.strip()
        for run in workflow_run_scalars(source)
        for line in run.splitlines()
        if "$downloader --url" in line
    ]
    assert len(download_commands) == 2
    assert all("--total-timeout-seconds 1800" in line for line in download_commands)


def test_proof_reconstructs_only_the_exact_three_model_parts_in_both_fresh_jobs():
    source = TRUSTED_PROOF.read_text("utf-8")
    execute = source[source.index("  execute:\n") : source.index("  sign:\n")]
    sign = source[source.index("  sign:\n") : source.index("  verify:\n")]
    for job, root in ((execute, "proof-input"), (sign, "signer-proof-input")):
        assert "$env:MODEL_PACK_URL -match '\\S' -and $partValueCount -eq 0" in job
        assert "$env:MODEL_PACK_URL -notmatch '\\S' -and $partValueCount -eq 9" in job
        assert "model pack input must be one complete archive URL or exactly the three ordered part identities" in job
        assert f"--parts-root {root}/model-parts" in job
        assert f"--destination {root}/model/official-model-pack.zip" in job
        for index in range(1, 4):
            part = f"official-model-pack.part{index:02d}"
            env = f"MODEL_PACK_PART_{index:02d}"
            assert f"$env:{env}_URL" in job
            assert f"$env:{env}_SHA256" in job
            assert f"$env:{env}_BYTES" in job
            assert f"{root}/model-parts/{part}" in job
            assert (
                f"--part-name {part} --part-sha256 $env:{env}_SHA256 "
                f"--part-bytes $env:{env}_BYTES"
            ) in job
        assert job.count("assemble-model-pack --parts-root") == 1


def _download_collection_shape(job: str, *, multipart: bool) -> list[list[str]]:
    start = job.index("          $downloads = @(")
    end = job.index("          foreach ($download in $downloads) {")
    script = job[start:end].replace(
        "            New-Item -ItemType Directory -Path ", "            # model-part directory: "
    )
    environment = {
        "CANDIDATE_ARCHIVE_URL": "https://example.invalid/candidate.zip",
        "CANDIDATE_ARCHIVE_SHA256": "a" * 64,
        "CANDIDATE_ARCHIVE_BYTES": "101",
        "CANDIDATE_MANIFEST_URL": "https://example.invalid/candidate-manifest.json",
        "CANDIDATE_MANIFEST_SHA256": "b" * 64,
        "CANDIDATE_MANIFEST_BYTES": "102",
        "CANDIDATE_ATTESTATION_URL": "https://example.invalid/candidate.sigstore.json",
        "CANDIDATE_ATTESTATION_SHA256": "c" * 64,
        "CANDIDATE_ATTESTATION_BYTES": "103",
        "CANDIDATE_EVIDENCE_URL": "https://example.invalid/candidate-evidence.json",
        "CANDIDATE_EVIDENCE_SHA256": "d" * 64,
        "CANDIDATE_EVIDENCE_BYTES": "104",
        "MODEL_PACK_SHA256": "e" * 64,
        "MODEL_PACK_BYTES": "105",
    }
    for index in range(1, 4):
        prefix = f"MODEL_PACK_PART_{index:02d}"
        environment[f"{prefix}_URL"] = (
            f"https://example.invalid/official-model-pack.part{index:02d}"
        ) if multipart else ""
        environment[f"{prefix}_SHA256"] = (chr(101 + index) * 64) if multipart else ""
        environment[f"{prefix}_BYTES"] = str(105 + index) if multipart else ""
    environment["MODEL_PACK_URL"] = "" if multipart else "https://example.invalid/model.zip"
    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            textwrap.dedent(script)
            + "\n$downloads | ConvertTo-Json -Depth 3 -Compress\n",
        ],
        cwd=ROOT,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("job_name", "multipart", "expected_count"),
    (("execute", False, 5), ("execute", True, 7), ("sign", False, 5), ("sign", True, 7)),
)
def test_proof_downloads_keep_every_input_as_one_four_field_tuple(
    job_name: str, multipart: bool, expected_count: int
):
    source = TRUSTED_PROOF.read_text("utf-8")
    boundaries = {
        "execute": ("  execute:\n", "  sign:\n"),
        "sign": ("  sign:\n", "  verify:\n"),
    }
    start, end = boundaries[job_name]
    job = source[source.index(start) : source.index(end)]

    downloads = _download_collection_shape(job, multipart=multipart)

    assert len(downloads) == expected_count
    assert all(len(download) == 4 for download in downloads)


def test_proof_and_fresh_verify_jobs_use_only_immutable_trusted_code_and_safe_env():
    source = TRUSTED_PROOF.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    jobs = workflow["jobs"]

    assert set(jobs) == {"companion_builder", "execute", "sign", "verify"}
    assert jobs["execute"]["runs-on"] == "windows-latest"
    assert jobs["sign"]["runs-on"] == "windows-latest"
    assert jobs["verify"]["runs-on"] == "windows-latest"
    assert source.count("repository: ${{ job.workflow_repository }}") == 3
    assert source.count("ref: ${{ job.workflow_sha }}") == 3
    assert "path: source" not in source
    assert "ref: ${{ github.sha }}" not in source
    assert "actions/checkout@v4" in source
    assert all("${{" not in run for run in workflow_run_scalars(source))

    execute = source[source.index("  execute:\n") : source.index("  sign:\n")]
    sign = source[source.index("  sign:\n") : source.index("  verify:\n")]
    verify = source[source.index("  verify:\n") :]
    assert "vending-vision-precutover-verifier.exe" in execute
    assert "$identity.modelPack.descriptorSha256" in execute
    assert "proof-input/candidate/trusted-builder-evidence.json" in execute
    assert "proof-input/model/official-model-pack.zip" in execute
    assert "bind-execution-proof --source companion-report.json" in execute
    assert "actions/attest-build-provenance@v4" not in execute
    assert "vending-vision-precutover-verifier.exe" not in sign
    assert "verify-execution-handoff" in sign
    assert "actions/attest-build-provenance@v4" in sign
    assert sign.index("verify-execution-handoff") < sign.index(
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
        r"(?ms)^      - name: Attest only the freshly revalidated canonical proof\n"
        r".*?(?=^      - name: Seal exact signed proof handoff)",
        trusted,
    )
    assert attest_match is not None
    attest_step = attest_match.group(0)
    attest_early = trusted.replace(attest_step, "").replace(
        "      - name: Revalidate exact canonical execution proof before attestation\n",
        attest_step
        + "      - name: Revalidate exact canonical execution proof before attestation\n",
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
        "missing-signed-identity-binder": trusted.replace(
            "& $env:TRUSTED_PYTHON $proofTool bind-execution-proof",
            "# removed signed identity binder",
            1,
        ),
        "untrusted-companion-source-binding": trusted.replace(
            '--companion-source-commit "d8c93b50cac005a371d06badcc398638fd8acabb"',
            "--companion-source-commit $env:CALLER_SHA",
            1,
        ),
        "multipart-allows-mixed-inputs": trusted.replace(
            "$env:MODEL_PACK_URL -match '\\S' -and $partValueCount -eq 0",
            "$env:MODEL_PACK_URL -match '\\S' -and $partValueCount -ge 0",
            1,
        ),
        "single-model-download-flattens-tuple": trusted.replace(
            "$downloads += ,@($env:MODEL_PACK_URL",
            "$downloads += @(@($env:MODEL_PACK_URL",
            1,
        ),
        "multipart-part-order-drift": trusted.replace(
            "--part-name official-model-pack.part01 --part-sha256 $env:MODEL_PACK_PART_01_SHA256 --part-bytes $env:MODEL_PACK_PART_01_BYTES",
            "--part-name official-model-pack.part02 --part-sha256 $env:MODEL_PACK_PART_01_SHA256 --part-bytes $env:MODEL_PACK_PART_01_BYTES",
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


def test_trusted_proof_policy_rejects_retired_builder_pin(tmp_path):
    trusted = TRUSTED_PROOF.read_text("utf-8")
    retired = "154dfd47b55ba13a5a968447b9f175d45f9ab990"
    candidate = tmp_path / "retired-builder.yml"
    candidate.write_text(trusted.replace(COMPANION_BUILDER_COMMIT, retired), "utf-8")
    completed = _check_policy(candidate)
    assert completed.returncode != 0


def test_trusted_proof_policy_rejects_privilege_and_cross_job_trust_regressions(
    tmp_path,
):
    trusted = TRUSTED_PROOF.read_text("utf-8")
    execute_permission = """  execute:
    needs: companion_builder
    runs-on: windows-latest
    timeout-minutes: 180
    permissions:
      contents: read
      attestations: read
"""
    execute_with_oidc = execute_permission.replace(
        "      attestations: read\n",
        "      attestations: read\n      id-token: write\n",
    )
    sign_execution_marker = (
        "      - name: Untrusted candidate execution mutation\n"
        "        shell: pwsh\n"
        "        run: |\n"
        "          & signer-verified-companion/vending-vision-precutover-verifier.exe --candidate-artifact signer-proof-input/candidate/candidate.zip\n"
        "      - name: Download fixed execution handoff\n"
    )
    arbitrary_python_marker = (
        "      - name: Untrusted Python mutation\n"
        "        shell: pwsh\n"
        "        run: |\n"
        "          $candidateScript = (Resolve-Path -LiteralPath signer-proof-input/candidate/untrusted.py).Path\n"
        "          & $env:TRUSTED_PYTHON $candidateScript\n"
        "      - name: Download fixed execution handoff\n"
    )
    raw_handoff = trusted.replace(
        "& $env:TRUSTED_PYTHON $proofTool verify-execution-handoff --directory execution-handoff --identity signer-proof-input-identity.json",
        "Get-Content -LiteralPath execution-handoff/precutover-ai-proof.json | Out-Null",
        1,
    )
    sign_start = trusted.index("  sign:\n")
    verify_start = trusted.index("  verify:\n")
    merged = trusted[:sign_start] + trusted[verify_start:].replace(
        "      - companion_builder\n      - sign\n",
        "      - companion_builder\n      - execute\n",
        1,
    )
    mutations = {
        "candidate-execution-job-gains-oidc": trusted.replace(
            execute_permission, execute_with_oidc, 1
        ),
        "signing-job-executes-candidate": trusted.replace(
            "      - name: Download fixed execution handoff\n", sign_execution_marker, 1
        ),
        "signing-job-executes-arbitrary-python": trusted.replace(
            "      - name: Download fixed execution handoff\n",
            arbitrary_python_marker,
            1,
        ),
        "signing-job-trusts-raw-handoff": raw_handoff,
        "execution-and-signing-jobs-merged": merged,
    }
    for name, source in mutations.items():
        candidate = tmp_path / f"{name}.yml"
        candidate.write_text(source, "utf-8")
        completed = _check_policy(candidate)
        assert completed.returncode != 0, name


def test_trusted_proof_policy_rejects_missing_mutable_or_zero_deadlines(tmp_path):
    trusted = TRUSTED_PROOF.read_text("utf-8")
    mutations = {
        "missing-execute-job-timeout": trusted.replace(
            "    timeout-minutes: 180\n", "", 1
        ),
        "zero-job-timeout": trusted.replace(
            "    timeout-minutes: 180\n", "    timeout-minutes: 0\n", 1
        ),
        "expression-verify-job-timeout": trusted.replace(
            "    timeout-minutes: 30\n",
            "    timeout-minutes: ${{ inputs.model_pack_bytes }}\n",
            1,
        ),
        "missing-download-total-timeout": trusted.replace(
            " --total-timeout-seconds 1800", "", 1
        ),
        "zero-download-total-timeout": trusted.replace(
            "--total-timeout-seconds 1800", "--total-timeout-seconds 0", 1
        ),
        "expression-download-total-timeout": trusted.replace(
            "--total-timeout-seconds 1800",
            "--total-timeout-seconds ${{ inputs.model_pack_bytes }}",
            1,
        ),
    }
    for name, source in mutations.items():
        assert source != trusted, name
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
