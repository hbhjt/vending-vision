from pathlib import Path
import hashlib
import json
import re
import stat
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


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
    assert "collect_submodules(\"vision.vendor.catvton\")" in worker_spec
    assert "official-ai-model-pack-descriptor.json" in worker_spec
    assert "official-ai-source-descriptor.json" in worker_spec
    assert "model.safetensors" not in worker_spec


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

    result = assert_ai_worker_layout(exe, required=True)

    assert result["path"] == worker
    assert len(result["sha256"]) == 64


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

    assert "trusted-ai-candidate-builder.yml@fbb97d16f42b2c20a04831750c639fda6db1a3e9" in publisher
    assert "trusted-ai-candidate-signer.yml@14e97b96b57acf3e3f23442e0d80904a55565a59" in publisher
    assert "scripts/build_exe.ps1" not in publisher
    assert "actions/attest-build-provenance" not in publisher
    assert "needs: trusted_builder" in publisher
    assert "needs: verify" in publisher
    assert publisher.count("runs-on: windows-latest") == 2
    assert "actions/download-artifact@v4" in publisher
    assert "gh attestation verify" in publisher
    assert "--signer-repo" not in publisher
    assert "--signer-workflow \"hbhjt/vending-vision/.github/workflows/trusted-ai-candidate-builder.yml\"" in publisher
    assert "--signer-digest \"fbb97d16f42b2c20a04831750c639fda6db1a3e9\"" in publisher
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
