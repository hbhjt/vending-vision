from pathlib import Path
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def materialize_regional_evaluator_resources(internal: Path) -> None:
    descriptor = json.loads((ROOT / "regional-evaluator-descriptor.json").read_text("utf-8"))
    (internal / "regional-evaluator-descriptor.json").write_text(
        json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        "utf-8",
    )
    for source in descriptor["sources"]:
        destination = internal / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source["path"]).read_bytes())
    marker = internal / "vision" / "_build_version.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes((ROOT / "vision" / "_build_version.py").read_bytes())


def test_packaged_archive_guard_rejects_retired_modules_and_resources():
    from scripts.verify_packaged_exe import retired_packaged_entries

    retired_module = "PYZ.pyz:" + ".".join(
        ("vision", "".join(("try", "_on", "_session")))
    )
    retired_resource = "resource:" + "/".join(
        ("contracts", "".join(("vem", "_vision", "_v1")), "manifest.json")
    )
    entries = {
        retired_module,
        retired_resource,
        "PYZ.pyz:vision.fast_attempt",
        "resource:contracts/vem_vision_v2/manifest.json",
    }

    assert retired_packaged_entries(entries) == sorted(
        [retired_module, retired_resource]
    )


def test_frozen_spec_keeps_the_generated_v2_bundle_and_static_boundary_import():
    spec = (ROOT / "vending_vision.spec").read_text("utf-8")

    assert "CONTRACT_DATA_FILES" in spec
    assert "OFFICIAL_AI_SOURCE_DATA_FILES" in spec
    assert "REGIONAL_EVALUATOR_SOURCE_DATA_FILES" in spec
    assert '(str(ROOT / "vision" / "_build_version.py"), "vision")' in spec
    assert '(CONTRACT_ROOT / "manifest.json", "contracts/vem_vision_v2")' in spec
    assert '(CONTRACT_ROOT / "__init__.py", "contracts/vem_vision_v2")' in spec
    assert '(CONTRACT_ROOT / "python" / "__init__.py", "contracts/vem_vision_v2/python")' in spec
    assert '(CONTRACT_ROOT / "python" / "vision_v2_models.py", "contracts/vem_vision_v2/python")' in spec
    assert '(CONTRACT_ROOT / "vision-v2.client.schema.json", "contracts/vem_vision_v2")' in spec
    assert '(CONTRACT_ROOT / "vision-v2.server.schema.json", "contracts/vem_vision_v2")' in spec
    assert '(str(ROOT / "contracts"), "contracts")' not in spec
    assert '"vision.v2_contract_bundle"' in spec
    assert '"vision.worker_self_check"' in spec
    assert '"vision.ai_model_pack"' in spec
    assert '"vision.ai_attempt_worker"' in spec
    assert '"vision.ai_attempt_process"' in spec
    assert '"vision.catvton_preprocess"' in spec
    assert "official-ai-source-descriptor.json" in spec
    assert "regional-evaluator-descriptor.json" in spec
    assert '"contracts.vem_vision_v2.python.vision_v2_models"' in spec


def test_ai_runtime_packaging_includes_worker_code_but_excludes_official_weights():
    spec = (ROOT / "vending_vision.spec").read_text("utf-8")
    packaging = (ROOT / "docs" / "PACKAGING.md").read_text("utf-8")
    launcher = (ROOT / "run_vision_server.py").read_text("utf-8")

    assert '"vision.ai_attempt_worker"' in spec
    assert '"vision.ai_attempt_process"' in spec
    assert '"vision.catvton_pose_masks"' in spec
    assert '"vision.catvton_preprocess"' in spec
    assert "vending_vision_ai_worker.spec" in spec
    assert '"--ai-attempt-worker"' in launcher
    assert '"--verify-ai-worker-boundary"' in launcher
    assert "AI runtime worker contract probe passed" in launcher
    assert "vending-vision-ai-models.zip" in packaging
    assert "VEM_AI_MODEL_PACK" in packaging
    assert "不加载完整模型或推理" in packaging
    assert "顾客启动禁止下载" in packaging
    assert "requirements-ai.txt" in packaging
    assert "requirements-ai.lock.json" in packaging
    assert "local_files_only=True" in packaging
    assert "safetensors" not in spec.lower()
    assert "model.safetensors" not in spec

    worker_spec = (ROOT / "vending_vision_ai_worker.spec").read_text("utf-8")
    assert 'name="vending-vision-ai-worker"' in worker_spec
    assert "OFFICIAL_AI_SOURCE_DATA_FILES" in worker_spec
    assert "REGIONAL_EVALUATOR_SOURCE_DATA_FILES" in worker_spec
    assert '(str(ROOT / "vision" / "_build_version.py"), "vision")' in worker_spec
    assert "collect_submodules(\"vision.vendor.catvton\")" in worker_spec
    assert "official-ai-model-pack-descriptor.json" in worker_spec
    assert "official-ai-source-descriptor.json" in worker_spec
    assert "regional-evaluator-descriptor.json" in worker_spec
    assert "model.safetensors" not in worker_spec


def test_windows_pre_cutover_companion_is_a_separate_self_contained_build_artifact():
    from scripts.workflow_yaml import load_workflow_yaml, workflow_run_scalars

    spec_path = ROOT / "vending_vision_precutover_verifier.spec"
    entrypoint = ROOT / "run_precutover_verifier.py"
    build = (ROOT / "scripts" / "build_exe.ps1").read_text("utf-8")
    builder = (
        ROOT / ".github/workflows/trusted-precutover-companion-builder.yml"
    ).read_text("utf-8")

    assert spec_path.is_file(), "missing independent pre-cutover verifier spec"
    assert entrypoint.is_file(), "missing frozen pre-cutover verifier entrypoint"
    spec = spec_path.read_text("utf-8")
    assert 'name="vending-vision-precutover-verifier"' in spec
    assert 'str(ROOT / "run_precutover_verifier.py")' in spec
    assert "collect_submodules(\"torch\")" not in spec
    assert "vending_vision_precutover_verifier.spec" in build
    assert '(Join-Path $ToolRoot "vending_vision_precutover_verifier.spec")' in build
    assert "git -C $ToolRoot rev-parse HEAD" in build
    assert "vending-vision-precutover-verifier" in build
    assert "precutover-companion-descriptor.json" in build
    assert "Invoke-Checked $CompanionExe --help" in build
    assert '"scripts\\precutover_companion_descriptor.py") verify' in build
    assert "vending-vision-precutover-verifier" in builder
    assert "    timeout-minutes: 180\n" in builder
    assert "precutover-companion-provenance.sigstore.json" in builder
    assert "archive_sha256" in builder
    assert "descriptor_sha256" in builder
    assert "attestation_bundle_sha256" in builder
    assert "actions/attest-build-provenance@v4" in builder
    assert "repository: ${{ job.workflow_repository }}" in builder
    assert "ref: ${{ job.workflow_sha }}" in builder
    assert "path: source" not in builder
    assert "candidate-manifest" not in builder
    assert "vending-vision-ai-worker" not in builder
    assert "--require-hashes" in builder
    assert "--total-timeout-seconds 1800" in builder
    workflow = load_workflow_yaml(builder)
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "core_wheelhouse_url",
        "core_wheelhouse_sha256",
        "core_wheelhouse_bytes",
    }
    assert all("${{" not in command for command in workflow_run_scalars(builder))
    assert "$CorePython -m PyInstaller" in build
    assert "Copy-Item -LiteralPath $CompanionRoot" not in build
    companion_cli = (ROOT / "vision/precutover_companion.py").read_text("utf-8")
    assert 'parser.add_argument("--python"' not in companion_cli


def test_frozen_specs_materialize_source_descriptor_python_files_for_probe_hashing():
    descriptor = json.loads((ROOT / "official-ai-source-descriptor.json").read_text("utf-8"))
    source_paths = {entry["path"] for entry in descriptor["sources"] if entry["path"].endswith(".py")}
    assert "vision/process_supervisor.py" in source_paths
    assert "vision/ai_runtime_descriptor.py" in source_paths

    for spec_name in ("vending_vision.spec", "vending_vision_ai_worker.spec"):
        spec = (ROOT / spec_name).read_text("utf-8")
        assert "OFFICIAL_AI_SOURCE_DATA_FILES" in spec
        for path in source_paths:
            if path.startswith("vision/"):
                assert path in spec or "OFFICIAL_AI_SOURCE_DESCRIPTOR_PATH" in spec


def test_official_ai_runtime_dependencies_are_separate_from_core_archive_lock():
    requirements = (ROOT / "requirements-ai.txt").read_text("utf-8")
    core_requirements = (ROOT / "requirements.txt").read_text("utf-8")
    runtime_descriptor = (ROOT / "ai-runtime-descriptor.json").read_text("utf-8")

    assert "torch==2.8.0+cpu" in requirements
    assert "torchvision==0.23.0+cpu" in requirements
    assert "diffusers==0.29.2" in requirements
    assert "transformers==4.53.3" in requirements
    assert "accelerate==0.31.0" in requirements
    assert ">=" not in requirements
    assert "opencv-python-headless" in requirements
    assert "torch==2.8.0" not in core_requirements
    assert "diffusers==0.29.2" not in core_requirements

    lock = (ROOT / "requirements-ai.lock.json").read_text("utf-8")
    assert '"schemaVersion":"vem-ai-worker-wheelhouse-release/v1"' in lock
    assert '"target":"windows-x86_64"' in lock
    assert '"wheels":[' in lock
    assert '"torch-2.8.0+cpu-cp311-cp311-win_amd64.whl"' in lock
    assert '"schemaVersion":"vem-ai-runtime-descriptor/v1"' in runtime_descriptor
    assert '"python":"3.11.9"' in runtime_descriptor
    assert '"target":"windows-x86_64"' in runtime_descriptor
    assert '"workerExecutable":"vending-vision-ai-worker/vending-vision-ai-worker.exe"' in runtime_descriptor


def test_vendored_catvton_closure_is_pinned_local_only_and_weight_free():
    vendor = ROOT / "vision" / "vendor" / "catvton"
    provenance = (vendor / "PROVENANCE.md").read_text("utf-8")
    source = "\n".join(path.read_text("utf-8") for path in vendor.rglob("*.py"))

    assert "3b795364a4d2f3b5adb365f39cdea376d20bc53c" in provenance
    assert (vendor / "model" / "pipeline.py").is_file()
    assert (vendor / "model" / "attn_processor.py").is_file()
    assert (vendor / "model" / "utils.py").is_file()
    assert (vendor / "model" / "SCHP" / "networks" / "AugmentCE2P.py").is_file()
    assert "from model." not in source
    assert "import model." not in source
    assert "snapshot_download" not in source
    assert "local_files_only=True" in source
    assert "stabilityai/sd-vae-ft-mse" not in (vendor / "model" / "pipeline.py").read_text("utf-8")
    assert not list(vendor.rglob("*.safetensors"))
    assert not list(vendor.rglob("*.pth"))
    assert not list(vendor.rglob("*.bin"))


def test_packaged_verifier_executes_the_frozen_bundle_positive_negative_probe():
    verifier = (ROOT / "scripts" / "verify_packaged_exe.py").read_text("utf-8")
    launcher = (ROOT / "run_vision_server.py").read_text("utf-8")

    assert '"--verify-v2-contract-bundle"' in verifier
    assert '"--verify-v2-try-on-workers"' in verifier
    assert '"V2 contract bundle probe passed"' in verifier
    assert '"V2 try-on worker probe passed"' in verifier
    assert '"contracts" / "vem_vision_v2"' in verifier
    assert "expected_contract_resources" in verifier
    assert '"__init__.py"' in verifier
    assert '"python/__init__.py"' in verifier
    assert '"python/vision_v2_models.py"' in verifier
    assert "must not contain Python bytecode" in verifier
    assert "assert_v2_contract_resources(contract_root)" in verifier
    assert "verify_result_query_is_not_logged" in verifier
    assert "assert_result_query_not_logged" in verifier
    assert "_safe_process_log" in verifier
    assert "packaged_archive_entries" in verifier
    assert '"--verify-ai-worker-boundary"' in verifier
    assert '"--probe-runtime"' in verifier
    assert '"official-catvton-worker-runtime"' in verifier
    assert '"--probe-runtime"' in launcher
    assert "missing-pack" not in launcher
    assert '"AI runtime worker contract probe passed"' in verifier
    assert "--require-ai-worker" in verifier
    assert "--trusted-subject-sha256" in verifier
    assert "--expected-embedded-manifest-sha256" in verifier
    assert "--expected-source-commit" in verifier
    assert "--extract-root" in verifier
    assert "verify_candidate_archive" in verifier
    assert "--candidate-manifest" not in verifier
    assert "--expected-candidate-manifest-sha256" not in verifier
    assert "PACKAGED_EXE_VERIFICATION=CORE_ONLY" in verifier
    assert "PACKAGED_EXE_VERIFICATION=PASS" in verifier
    assert "assert_ai_worker_layout" in verifier
    assert "assert_hard_cutover_archive_absence(exe_path)" in verifier
    assert "retired modules remain in packaged archive" in verifier
    assert "retired_try_on_route" in verifier
    assert "assert_no_worker_resource_leak_output" in verifier


def test_packaged_verifier_require_ai_worker_rejects_missing_layout(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout

    exe = tmp_path / "vending-vision" / "vending-vision.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"main")

    assert assert_ai_worker_layout(exe, required=False) is None
    try:
        assert_ai_worker_layout(exe, required=True)
    except AssertionError as exc:
        assert "missing packaged AI worker" in str(exc)
    else:
        raise AssertionError("missing worker must fail when required")


def test_packaged_verifier_ai_worker_layout_binds_descriptor_resources(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    worker = tmp_path / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}"
    internal = worker.parent / "_internal"
    exe.parent.mkdir()
    internal.mkdir(parents=True)
    exe.write_bytes(b"main")
    worker.write_bytes(b"worker")
    for name in (
        "official-ai-model-pack-descriptor.json",
        "ai-runtime-descriptor.json",
        "requirements-ai.lock.json",
        "official-ai-source-descriptor.json",
    ):
        (internal / name).write_text("{}", "utf-8")
    materialize_regional_evaluator_resources(internal)

    result = assert_ai_worker_layout(exe, required=True)

    assert result["path"] == worker
    assert len(result["sha256"]) == 64


def test_packaged_verifier_rejects_missing_regional_evaluator_source(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    worker = tmp_path / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}"
    internal = worker.parent / "_internal"
    exe.parent.mkdir()
    internal.mkdir(parents=True)
    exe.write_bytes(b"main")
    worker.write_bytes(b"worker")
    for name in (
        "official-ai-model-pack-descriptor.json",
        "ai-runtime-descriptor.json",
        "requirements-ai.lock.json",
        "official-ai-source-descriptor.json",
    ):
        (internal / name).write_text("{}", "utf-8")
    materialize_regional_evaluator_resources(internal)
    (internal / "vision" / "config.py").unlink()

    with pytest.raises(AssertionError, match="regional evaluator"):
        assert_ai_worker_layout(exe, required=True)


def test_packaged_verifier_rejects_missing_release_version_runtime_marker(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    worker = tmp_path / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}"
    internal = worker.parent / "_internal"
    exe.parent.mkdir()
    internal.mkdir(parents=True)
    exe.write_bytes(b"main")
    worker.write_bytes(b"worker")
    for name in (
        "official-ai-model-pack-descriptor.json",
        "ai-runtime-descriptor.json",
        "requirements-ai.lock.json",
        "official-ai-source-descriptor.json",
    ):
        (internal / name).write_text("{}", "utf-8")
    materialize_regional_evaluator_resources(internal)
    (internal / "vision" / "_build_version.py").unlink()

    with pytest.raises(AssertionError, match="release version runtime marker"):
        assert_ai_worker_layout(exe, required=True)


def test_packaged_verifier_rejects_noncanonical_release_version_runtime_marker(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    worker = tmp_path / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}"
    internal = worker.parent / "_internal"
    exe.parent.mkdir()
    internal.mkdir(parents=True)
    exe.write_bytes(b"main")
    worker.write_bytes(b"worker")
    for name in (
        "official-ai-model-pack-descriptor.json",
        "ai-runtime-descriptor.json",
        "requirements-ai.lock.json",
        "official-ai-source-descriptor.json",
    ):
        (internal / name).write_text("{}", "utf-8")
    materialize_regional_evaluator_resources(internal)
    (internal / "vision" / "_build_version.py").write_text(
        'APP_VERSION = "untrusted"\n', "utf-8"
    )

    with pytest.raises(AssertionError, match="invalid release version runtime marker"):
        assert_ai_worker_layout(exe, required=True)


def test_candidate_archive_rejects_self_manifested_exact_json_fake_without_external_trust(tmp_path):
    from scripts.candidate_artifact_manifest import verify_candidate_archive, write_candidate_archive

    dist = tmp_path / "dist"
    main = dist / "vending-vision" / "vending-vision.exe"
    worker = dist / "vending-vision-ai-worker" / "vending-vision-ai-worker.exe"
    internal = worker.parent / "_internal"
    main.parent.mkdir(parents=True)
    internal.mkdir(parents=True)
    main.write_bytes(b"main")
    worker.write_bytes(b"real-worker")
    for name in (
        "ai-runtime-descriptor.json", "requirements-ai.lock.json",
        "official-ai-source-descriptor.json", "official-ai-model-pack-descriptor.json",
    ):
        (internal / name).write_bytes(name.encode("ascii"))
    artifact = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "candidate.manifest.json"
    trusted = write_candidate_archive(dist, artifact, manifest_path, source_commit="a" * 40)
    repeated_artifact = tmp_path / "candidate-repeat.zip"
    repeated_manifest = tmp_path / "candidate-repeat.manifest.json"
    repeated = write_candidate_archive(
        dist, repeated_artifact, repeated_manifest, source_commit="a" * 40
    )
    assert repeated == trusted
    assert repeated_artifact.read_bytes() == artifact.read_bytes()

    verified = verify_candidate_archive(
        artifact,
        tmp_path / "verified-real",
        expected_subject_sha256=trusted["subjectSha256"],
        expected_manifest_sha256=trusted["embeddedManifestSha256"],
        expected_source_commit="a" * 40,
    )
    assert verified["workerExecutable"].read_bytes() == b"real-worker"

    worker.write_text('{"probe":"official-catvton-worker-runtime","torch":"2.8.0+cpu"}\n', "utf-8")
    fake_artifact = tmp_path / "candidate-fake.zip"
    fake_manifest = tmp_path / "candidate-fake.manifest.json"
    fake = write_candidate_archive(dist, fake_artifact, fake_manifest, source_commit="a" * 40)

    with pytest.raises(AssertionError, match="trusted subject digest mismatch"):
        verify_candidate_archive(
            fake_artifact,
            tmp_path / "fake-subject",
            expected_subject_sha256=trusted["subjectSha256"],
            expected_manifest_sha256=trusted["embeddedManifestSha256"],
            expected_source_commit="a" * 40,
        )
    with pytest.raises(AssertionError, match="embedded manifest digest mismatch"):
        verify_candidate_archive(
            fake_artifact,
            tmp_path / "fake-manifest",
            expected_subject_sha256=fake["subjectSha256"],
            expected_manifest_sha256=trusted["embeddedManifestSha256"],
            expected_source_commit="a" * 40,
        )

    not_zip = tmp_path / "not.zip"
    not_zip.write_bytes(b"not-a-zip")
    with pytest.raises(AssertionError, match="candidate artifact is not a ZIP"):
        verify_candidate_archive(
            not_zip,
            tmp_path / "not-zip",
            expected_subject_sha256=hashlib.sha256(not_zip.read_bytes()).hexdigest(),
            expected_manifest_sha256=trusted["embeddedManifestSha256"],
            expected_source_commit="a" * 40,
        )

    tampered = tmp_path / "candidate-payload-tampered.zip"
    with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(tampered, "w") as output:
        for info in source.infolist():
            payload = source.read(info)
            if info.filename.endswith("vending-vision-ai-worker.exe"):
                payload = b"evil-worker"
            output.writestr(info, payload)
    with pytest.raises(AssertionError, match="payload digest mismatch"):
        verify_candidate_archive(
            tampered,
            tmp_path / "tampered-payload",
            expected_subject_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
            expected_manifest_sha256=trusted["embeddedManifestSha256"],
            expected_source_commit="a" * 40,
        )


@pytest.mark.parametrize("case", ["traversal", "symlink", "special", "collision", "compressed"])
def test_candidate_archive_safe_extract_rejects_unsafe_zip_entries(tmp_path, case):
    from scripts.candidate_artifact_manifest import verify_candidate_archive

    artifact = tmp_path / f"{case}.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        if case == "traversal":
            archive.writestr("../escape.exe", b"bad")
        elif case in {"symlink", "special"}:
            info = zipfile.ZipInfo("unsafe")
            info.create_system = 3
            mode = stat.S_IFLNK if case == "symlink" else stat.S_IFIFO
            info.external_attr = (mode | 0o777) << 16
            archive.writestr(info, b"target")
        elif case == "collision":
            archive.writestr("Demo.exe", b"one")
            archive.writestr("demo.exe", b"two")
        else:
            archive.writestr("compressed.bin", b"zip-bomb-shape", compress_type=zipfile.ZIP_DEFLATED)

    with pytest.raises(AssertionError, match="candidate archive"):
        verify_candidate_archive(
            artifact,
            tmp_path / "extracted",
            expected_subject_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            expected_manifest_sha256="0" * 64,
            expected_source_commit="a" * 40,
        )
    assert not (tmp_path / "extracted").exists()


def test_packaged_verifier_rejects_worker_that_does_not_emit_runtime_probe_json(tmp_path):
    from scripts.verify_packaged_exe import assert_ai_worker_layout, verify_ai_worker_runtime_probe

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    worker = tmp_path / "vending-vision-ai-worker" / f"vending-vision-ai-worker{suffix}"
    internal = worker.parent / "_internal"
    exe.parent.mkdir()
    internal.mkdir(parents=True)
    if sys.platform == "win32":
        # A Windows fixture must be executable; a POSIX shebang is WinError 216.
        shutil.copyfile(sys.executable, worker)
        runtime_dll = Path(sys.base_prefix) / (
            f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        )
        shutil.copyfile(runtime_dll, worker.parent / runtime_dll.name)
        exe.write_bytes(b"MZ-test-entry")
    else:
        exe.write_text("#!/usr/bin/env python3\nprint('not-json')\n", "utf-8")
        worker.write_text("#!/usr/bin/env python3\nprint('not-json')\n", "utf-8")
        worker.chmod(0o755)
    for name in (
        "official-ai-model-pack-descriptor.json",
        "ai-runtime-descriptor.json",
        "requirements-ai.lock.json",
        "official-ai-source-descriptor.json",
    ):
        (internal / name).write_text("{}", "utf-8")
    materialize_regional_evaluator_resources(internal)

    layout = assert_ai_worker_layout(exe, required=True)
    try:
        verify_ai_worker_runtime_probe(layout)
    except AssertionError as exc:
        assert "AI worker runtime probe failed" in str(exc)
    else:
        raise AssertionError("worker without runtime probe JSON must fail")


def test_build_and_publish_candidate_require_ai_wheelhouse_and_dual_specs():
    build = (ROOT / "scripts" / "build_exe.ps1").read_text("utf-8")
    builder = (ROOT / ".github" / "workflows" / "trusted-ai-candidate-builder.yml").read_text("utf-8")
    signer = (ROOT / ".github" / "workflows" / "trusted-ai-candidate-signer.yml").read_text("utf-8")
    publisher = (ROOT / ".github" / "workflows" / "publish-candidate.yml").read_text("utf-8")

    assert "AiWheelhouseDescriptor" in build
    assert '".venv-packaging-core"' in build
    assert '".venv-packaging-ai"' in build
    assert "bootstrap_build_envs.py" in build
    assert "render_ai_build_requirements.py" in build
    assert "requirements-ai-build-tools.txt" in build
    assert "verify_ai_wheelhouse.py" in build
    assert "requirements-ai-release.txt" in build
    assert "--requirements-output" in build
    assert "--python $CorePython --target-sys-platform win32" in build
    assert "Invoke-Checked $AiPython" in build
    assert "Invoke-Checked $CorePython" in build
    assert "$AiPython -m pip install" in build
    assert "--require-hashes --no-deps -r $AiBuildRequirements" in build
    assert "run_ai_attempt_worker.py" in build
    assert '"--probe-runtime"' in build
    assert "vending_vision.spec" in build
    assert "vending_vision_ai_worker.spec" in build
    assert "vending-vision-ai-worker" in build
    assert "--require-ai-worker" not in build
    assert "$CoreDist" in build
    assert "$AiDist" in build
    assert "Copy-Item -LiteralPath (Join-Path $CoreDist \"vending-vision\")" in build
    assert "Copy-Item -LiteralPath (Join-Path $AiDist \"vending-vision-ai-worker\")" in build
    assert "pip download" not in build

    assert "ai-wheelhouse" in builder
    assert "materialize_ai_wheelhouse.py" in builder
    assert "requirements-ai.lock.json" in builder
    assert "CORE_WHEELHOUSE_URL" in builder
    assert "CORE_WHEELHOUSE_SHA256" in builder
    assert "CORE_WHEELHOUSE_BYTES" in builder
    assert "--expected-bytes $env:CORE_WHEELHOUSE_BYTES" in builder
    assert "download_verified_archive.py" in builder
    assert "requirements-ai-release.txt" in builder
    assert "--requirements-output build/requirements-ai-release.txt" in builder
    assert "pip download" not in builder
    assert builder.count("scripts/build_exe.ps1") == 1
    assert "-SourceRoot $PWD" in builder
    assert "candidate_artifact_manifest.py" in builder
    assert "--manifest-output" in builder
    assert "--source-commit" in builder
    assert "Compress-Archive" not in builder
    assert "actions/attest-build-provenance@v4" in builder
    assert builder.index("--require-ai-worker") < builder.index("actions/attest-build-provenance@v4")
    assert "subject-path:" in builder
    assert "id-token: write" in builder
    assert "attestations: write" in builder
    assert "secrets:" not in builder

    assert "trusted-ai-candidate-builder.yml@3fe9e00c98d9df59c71ce9be5b980a713ddd3110" in publisher
    assert "trusted-ai-candidate-signer.yml@af9f7bb766e8a467e8c9a24396a76b616fd68188" in publisher
    assert "scripts/build_exe.ps1" not in publisher
    assert "actions/attest-build-provenance" not in publisher
    assert "needs: trusted_builder" in publisher
    assert "needs: verify" in publisher
    assert publisher.count("runs-on: windows-latest") == 2
    assert "actions/download-artifact@v4" in publisher
    assert "gh attestation verify" in publisher
    assert "--signer-repo" not in publisher
    assert "--signer-workflow \"hbhjt/vending-vision/.github/workflows/trusted-ai-candidate-builder.yml\"" in publisher
    assert "--signer-digest \"3fe9e00c98d9df59c71ce9be5b980a713ddd3110\"" in publisher
    assert "--source-ref" in publisher
    assert "--source-digest" in publisher
    assert "--deny-self-hosted-runners" in publisher
    assert "--trusted-subject-sha256" in publisher
    assert "--expected-embedded-manifest-sha256" in publisher
    assert "--expected-source-commit" in publisher
    assert "--extract-root" in publisher
    assert "--require-ai-worker" in publisher
    assert "environment: experimental-candidate" in signer
    assert "--trusted-builder-evidence" in signer
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" in signer
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in publisher
    assert "--expected-candidate-manifest-sha256" not in publisher


def test_windows_ci_runs_tests_and_digest_bound_packaging_in_parallel_before_publish():
    """Hosted publishing must join independent Windows test and packaging gates."""
    from scripts.workflow_yaml import load_workflow_yaml

    workflow = load_workflow_yaml(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    )
    jobs = workflow["jobs"]
    test_job = jobs["windows-test"]
    package_job = jobs["windows-package"]
    publish_job = jobs["publish-main-artifacts"]
    assert "needs" not in test_job
    assert "needs" not in package_job
    assert publish_job["needs"] == [
        "test",
        "windows-test",
        "regional-evidence-contract",
        "windows-package",
    ]
    assert publish_job["if"] == "github.ref == 'refs/heads/main'"

    test_runs = "\n".join(
        step.get("run", "") for step in test_job["steps"] if isinstance(step, dict)
    )
    assert "python -m pytest -q" in test_runs
    assert "python -m py_compile app.py run_vision_server.py vision/*.py scripts/*.py" in test_runs
    assert "verify_model_manifest" in test_runs
    assert "scripts/build_exe.ps1" not in test_runs

    steps = package_job["steps"]
    named_steps = [step for step in steps if isinstance(step, dict) and "name" in step]
    materialize_index, materialize_step = next(
        (index, step)
        for index, step in enumerate(named_steps)
        if step["name"] == "物化精确 AI wheel 闭包"
    )
    build_index, build_step = next(
        (index, step)
        for index, step in enumerate(named_steps)
        if step["name"] == "构建并验证 Windows runtime 与录播交付包"
    )
    materialize_run = " ".join(materialize_step["run"].split())
    build_run = " ".join(build_step["run"].split())
    lock = ROOT / "requirements-ai.lock.json"
    runtime_descriptor = json.loads(
        (ROOT / "ai-runtime-descriptor.json").read_text("utf-8")
    )

    materialize = (
        "python scripts/materialize_ai_wheelhouse.py "
        "--descriptor requirements-ai.lock.json "
        "--runtime-descriptor ai-runtime-descriptor.json "
        "--destination ai-wheelhouse"
    )
    build = (
        "./scripts/build_exe.ps1 "
        "-Wheelhouse (Join-Path $PWD \"wheelhouse\") "
        "-AiWheelhouse (Join-Path $PWD \"ai-wheelhouse\") "
        "-AiWheelhouseDescriptor (Join-Path $PWD \"requirements-ai.lock.json\")"
    )

    assert materialize in materialize_run
    assert build in build_run
    assert (
        '& (Join-Path $PWD ".venv-packaging-core\\Scripts\\python.exe") '
        "scripts/verify_packaged_exe.py"
    ) in build_run
    assert materialize_index < build_index
    assert build_run.count("scripts/build_exe.ps1") == 1
    assert build_run.count("scripts/verify_packaged_exe.py") == 1
    assert build_run.count("scripts/package_main_artifacts.ps1") == 1
    assert '-Commit "${{ github.sha }}"' in build_run

    staging_upload = next(
        step for step in steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert staging_upload["with"] == {
        "name": "vending-vision-main-staging-${{ github.sha }}",
        "path": "main-artifacts/*",
        "if-no-files-found": "error",
    }

    publish_steps = publish_job["steps"]
    staging_download = next(
        step for step in publish_steps if step.get("uses") == "actions/download-artifact@v4"
    )
    final_upload = next(
        step for step in publish_steps if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert staging_download["with"] == {
        "name": "vending-vision-main-staging-${{ github.sha }}",
        "path": "main-artifacts",
    }
    assert final_upload["with"] == {
        "name": "vending-vision-main-${{ github.sha }}",
        "path": "main-artifacts/*",
        "if-no-files-found": "error",
    }
    assert runtime_descriptor["requirementsAiLockSha256"] == hashlib.sha256(
        lock.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("missing", [None, "server-valid.json"])
def test_main_artifact_runtime_allows_only_the_exact_v2_contract_fixtures(tmp_path, missing):
    """The frozen V2 bundle needs its four contract fixtures, not test video data."""
    root = tmp_path / "package-root"
    scripts = root / "scripts"
    runtime = root / "dist" / "vending-vision"
    recorded = root / "fixtures" / "recorded-video"
    scripts.mkdir(parents=True)
    runtime.mkdir(parents=True)
    recorded.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "package_main_artifacts.ps1", scripts)
    (runtime / "vending-vision.exe").write_bytes(b"MZ")
    (recorded / "expected-results.json").write_text("{}", "utf-8")
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")
    for name in (
        "client-invalid.json",
        "client-valid.json",
        "server-invalid.json",
        "server-valid.json",
    ):
        path = runtime / "_internal" / "contracts" / "vem_vision_v2" / "fixtures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", "utf-8")
    if missing is not None:
        (
            runtime
            / "_internal"
            / "contracts"
            / "vem_vision_v2"
            / "fixtures"
            / missing
        ).unlink()

    output = root / "output"
    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(scripts / "package_main_artifacts.ps1"),
            "-Commit",
            "a" * 40,
            "-OutputDirectory",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = f"{completed.stdout}{completed.stderr}"
    if missing is not None:
        assert completed.returncode != 0
        assert "Runtime archive is missing V2 contract fixtures" in combined
        return
    assert completed.returncode == 0, combined
    with zipfile.ZipFile(output / "vending-vision-windows-x86_64.zip") as archive:
        entries = {entry.replace("\\", "/") for entry in archive.namelist()}
    assert {
        "_internal/contracts/vem_vision_v2/fixtures/client-invalid.json",
        "_internal/contracts/vem_vision_v2/fixtures/client-valid.json",
        "_internal/contracts/vem_vision_v2/fixtures/server-invalid.json",
        "_internal/contracts/vem_vision_v2/fixtures/server-valid.json",
    } <= entries


def test_main_artifact_runtime_rejects_noncontract_fixture_paths(tmp_path):
    root = tmp_path / "package-root"
    scripts = root / "scripts"
    runtime = root / "dist" / "vending-vision"
    recorded = root / "fixtures" / "recorded-video"
    scripts.mkdir(parents=True)
    runtime.mkdir(parents=True)
    recorded.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "package_main_artifacts.ps1", scripts)
    (runtime / "vending-vision.exe").write_bytes(b"MZ")
    (runtime / "_internal" / "fixtures" / "unexpected.json").parent.mkdir(parents=True)
    (runtime / "_internal" / "fixtures" / "unexpected.json").write_text("{}", "utf-8")
    for name in (
        "client-invalid.json",
        "client-valid.json",
        "server-invalid.json",
        "server-valid.json",
    ):
        path = runtime / "_internal" / "contracts" / "vem_vision_v2" / "fixtures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", "utf-8")
    (recorded / "expected-results.json").write_text("{}", "utf-8")
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")

    completed = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(scripts / "package_main_artifacts.ps1"),
            "-Commit",
            "a" * 40,
            "-OutputDirectory",
            str(root / "output"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert completed.returncode != 0
    assert "Runtime archive includes recorded-video fixtures" in (
        completed.stdout + completed.stderr
    )


def _write_runtime_contract_fixtures(runtime, contract_root):
    for name in (
        "client-invalid.json",
        "client-valid.json",
        "server-invalid.json",
        "server-valid.json",
    ):
        path = runtime / "_internal" / contract_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", "utf-8")


def _run_main_artifact_package(root):
    return subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(root / "scripts" / "package_main_artifacts.ps1"),
            "-Commit",
            "a" * 40,
            "-OutputDirectory",
            str(root / "output"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _main_artifact_harness_root(tmp_path):
    root = tmp_path / "package-root"
    (root / "scripts").mkdir(parents=True)
    (root / "dist" / "vending-vision").mkdir(parents=True)
    recorded = root / "fixtures" / "recorded-video"
    recorded.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "package_main_artifacts.ps1", root / "scripts")
    (root / "dist" / "vending-vision" / "vending-vision.exe").write_bytes(b"MZ")
    (recorded / "expected-results.json").write_text("{}", "utf-8")
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")
    return root


def _run_main_artifact_archive_guard(root, runtime_entries):
    """Exercise the PowerShell ZIP guard with legacy entry separators directly."""
    script_path = root / "scripts" / "package_main_artifacts.ps1"
    script = script_path.read_text("utf-8")
    script = script.replace(
        "Remove-Item -LiteralPath $RuntimeArchive, $FixtureArchive -Force -ErrorAction SilentlyContinue",
        "$null = 1",
    ).replace(
        "Compress-Archive -Path (Join-Path $RuntimeStage \"*\") -DestinationPath $RuntimeArchive -CompressionLevel Optimal",
        "$null = 1",
    ).replace(
        "Compress-Archive -Path (Join-Path $FixtureStage \"*\") -DestinationPath $FixtureArchive -CompressionLevel Optimal",
        "$null = 1",
    )
    script_path.write_text(script, "utf-8")
    output = root / "output"
    output.mkdir()
    manifest = json.dumps(
        {
            "schemaVersion": "vending-vision-main-artifacts/v1",
            "commit": "a" * 40,
            "runtimeArchive": "vending-vision-windows-x86_64.zip",
            "fixtureArchive": "vending-vision-test-fixtures.zip",
        }
    ).encode()
    with zipfile.ZipFile(output / "vending-vision-windows-x86_64.zip", "w") as archive:
        for entry in runtime_entries:
            archive.writestr(entry, manifest if entry == "vision-artifact.json" else b"x")
    with zipfile.ZipFile(output / "vending-vision-test-fixtures.zip", "w") as archive:
        for entry in (
            "recorded-video/expected-results.json",
            "recorded-video/top.mp4",
            "recorded-video/front.mp4",
        ):
            archive.writestr(entry, b"x")
        archive.writestr("vision-artifact.json", manifest)
    return _run_main_artifact_package(root)


def _runtime_contract_archive_entries(separator="/"):
    prefix = separator.join(
        ("_internal", "contracts", "vem_vision_v2", "fixtures")
    )
    return [
        "vending-vision.exe",
        "vision-artifact.json",
        *(f"{prefix}{separator}{name}" for name in (
            "client-invalid.json",
            "client-valid.json",
            "server-invalid.json",
            "server-valid.json",
        )),
    ]


def test_main_artifact_runtime_normalizes_legacy_backslash_contract_entries(tmp_path):
    root = _main_artifact_harness_root(tmp_path)

    completed = _run_main_artifact_archive_guard(
        root, _runtime_contract_archive_entries("\\")
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("fixtures/unexpected.json", "Runtime archive includes recorded-video fixtures"),
        ("recorded-video/top.mp4", "Runtime archive includes recorded-video fixtures"),
        ("unexpected.mp4", "Runtime archive includes recorded-video fixtures"),
        ("top.mp4", "Runtime archive includes recorded-video fixtures"),
        ("front.mp4", "Runtime archive includes recorded-video fixtures"),
        ("expected-results.json", "Runtime archive includes recorded-video fixtures"),
    ],
)
def test_main_artifact_runtime_rejects_forward_fixture_and_video_entries(tmp_path, path, error):
    root = _main_artifact_harness_root(tmp_path)
    runtime = root / "dist" / "vending-vision"
    _write_runtime_contract_fixtures(runtime, "contracts/vem_vision_v2/fixtures")
    extra = runtime / "_internal" / path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unexpected")

    completed = _run_main_artifact_package(root)

    assert completed.returncode != 0
    assert error in completed.stdout + completed.stderr


def test_main_artifact_runtime_rejects_backslash_fixture_and_casefold_collisions(tmp_path):
    root = _main_artifact_harness_root(tmp_path)

    entries = _runtime_contract_archive_entries()
    entries.append("_internal\\contracts\\vem_vision_v2\\Fixtures\\client-valid.json")
    entries.append("_internal\\fixtures\\unexpected.json")
    completed = _run_main_artifact_archive_guard(root, entries)

    assert completed.returncode != 0
    assert "Runtime archive has case-folding collision" in completed.stdout + completed.stderr


def test_main_artifact_runtime_rejects_slash_backslash_normalized_collisions(tmp_path):
    root = _main_artifact_harness_root(tmp_path)

    completed = _run_main_artifact_archive_guard(
        root,
        _runtime_contract_archive_entries()
        + _runtime_contract_archive_entries("\\")[2:],
    )

    assert completed.returncode != 0
    assert "Runtime archive has duplicate normalized entry" in completed.stdout + completed.stderr


def test_worker_probe_executes_production_observation_and_render_ipc():
    probe = subprocess.run(
        [sys.executable, str(ROOT / "run_vision_server.py"), "--verify-v2-try-on-workers"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    combined = f"{probe.stdout}{probe.stderr}"
    assert probe.returncode == 0, combined
    assert "production acquisition observation: none" in probe.stdout
    assert "production render response:" in probe.stdout
    assert "resource_tracker" not in combined
    assert "leaked shared_memory" not in combined


def test_frozen_runtime_keeps_the_spawn_safe_render_worker_entry():
    spec = (ROOT / "vending_vision.spec").read_text("utf-8")
    launcher = (ROOT / "run_vision_server.py").read_text("utf-8")
    target = (ROOT / "vision" / "render_worker_target.py").read_text("utf-8")
    acquisition = (ROOT / "vision" / "acquisition_observer.py").read_text("utf-8")

    assert '"vision.render_worker_target"' in spec
    assert '"vision.acquisition_observer"' in spec
    assert '"vision.worker_self_check"' in spec
    assert "multiprocessing.freeze_support()" in launcher
    assert launcher.index("multiprocessing.freeze_support()") < launcher.index(
        "from app import app as fastapi_app"
    )
    assert "import app" not in target
    assert "import app" not in acquisition
    assert "vision.pose_estimator" not in target
    assert "vision.model" not in target


def test_every_production_uvicorn_entry_disables_access_logs():
    """A result capability query must never become an HTTP access log line."""
    sources = [
        path
        for path in ROOT.rglob("*")
        if path.suffix in {".py", ".bat"} and "tests" not in path.parts
    ]
    uvicorn_entries = []
    for path in sources:
        source = path.read_text("utf-8")
        if "uvicorn.run(" in source:
            uvicorn_entries.append(path)
            assert re.search(r"uvicorn\.run\([\s\S]*?access_log\s*=\s*False", source)
        if "-m uvicorn" in source:
            uvicorn_entries.append(path)
            assert "--no-access-log" in source

    assert {path.name for path in uvicorn_entries} >= {
        "run_vision_server.py",
        "start_server.bat",
    }


def test_packaged_production_modules_cannot_select_test_pose_fixtures():
    production_modules = [
        ROOT / "run_vision_server.py",
        ROOT / "vision" / "attempt_worker.py",
        ROOT / "vision" / "render_worker_target.py",
    ]
    source = "\n".join(path.read_text("utf-8").lower() for path in production_modules)
    worker_target = (ROOT / "vision" / "render_worker_target.py").read_text("utf-8").lower()
    spec = (ROOT / "vending_vision.spec").read_text("utf-8").lower()

    assert "test_pose" not in source
    assert "fake" not in source
    assert "fixture" not in worker_target
    assert "test_pose" not in worker_target
    assert "fake" not in worker_target
    assert '"tests"' not in spec
    assert "test_pose" not in spec
    assert "fake" not in spec
