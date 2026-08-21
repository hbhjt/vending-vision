from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def _init_clean_repository(root):
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Packaging Tests"], cwd=root, check=True)
    (root / ".gitignore").write_text("dist/\nignored-cache/\n", "utf-8")
    (root / ".python-version").write_text("3.11.13\n", "utf-8")
    (root / "tracked.txt").write_text("packaging source\n", "utf-8")
    subprocess.run(["git", "add", ".gitignore", ".python-version", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "packaging source"], cwd=root, check=True)


def test_ci_publishes_only_main_runtime_fixture_delivery_pair():
    """The public delivery seam has no retired RC artifact or generator."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    retired = "can" + "didate"
    generator = retired + "_artifact_manifest.py"

    assert not (ROOT / "scripts" / generator).exists()
    assert not (ROOT / "hard-cutover-binary-allowlist.json").exists()
    assert retired + "-artifacts" not in ci
    assert generator not in ci
    assert "vending-vision-" + retired + "-" not in ci


def test_packaged_archive_guard_rejects_retired_modules_and_resources():
    from scripts.hard_cutover_policy import retired_packaged_entries

    retired_module = "PYZ.pyz:" + ".".join(
        ("vision", "".join(("try", "_on", "_session")))
    )
    retired_resource = "resource:" + "/".join(
        ("contracts", "".join(("vem", "_vision", "_v1")), "manifest.json")
    )
    retired_worker = "PYZ.pyz:vision." + "".join(("a", "i", "_attempt_worker"))
    retired_vendor = "resource:vision/vendor/" + "".join(("cat", "vton")) + "/model.py"
    retired_quick_attempt = "PYZ.pyz:vision." + "".join(("fa", "st", "_attempt"))
    retired_ai_entries = {
        "resource:" + "".join(("official-", "a", "i", "-source-descriptor.json")),
        "resource:" + "".join(("requirements-", "a", "i", ".txt")),
        "resource:" + "".join(("a", "i", "-runtime-descriptor.json")),
        "resource:" + "".join(("a", "i", "-model-manifest.json")),
        "resource:" + "".join(("a", "i", "-wheelhouse/package.whl")),
        "resource:" + "".join((".venv-packaging-", "a", "i", "/python.exe")),
        "PYZ.pyz:vision." + "".join(("a", "i", "_acceptance_evidence")),
        "PYZ.pyz:scripts." + "".join(("materialize_", "a", "i", "_wheelhouse")),
        "PYZ.pyz:scripts." + "".join(("verify_", "a", "i", "_wheelhouse")),
        "PYZ.pyz:scripts." + "".join(("render_", "a", "i", "_build_requirements")),
        "PYZ.pyz:vision." + "".join(("source", "_provenance")),
        "PYZ.pyz:vision." + "".join(("process", "_supervisor")),
        "resource:" + "".join(("regional", "-evaluator-descriptor.json")),
    }
    entries = {
        retired_module,
        retired_resource,
        retired_quick_attempt,
        retired_worker,
        retired_vendor,
        *retired_ai_entries,
        "resource:contracts/vem_vision_v2/manifest.json",
    }

    assert retired_packaged_entries(entries) == sorted(
        [
            retired_module,
            retired_resource,
            retired_quick_attempt,
            retired_worker,
            retired_vendor,
            *retired_ai_entries,
        ]
    )


@pytest.mark.parametrize(
    "entry",
    (
        "vision/" + "/".join(("a" + "i", "attempt_worker.py")),
        ".".join(("vision", "a" + "i", "attempt_worker")),
        "VISION\\" + "A" + "I" + "\\ATTEMPT-WORKER.PY",
        "resource:pkg/deep/" + "a" + "i" + "_source_provenance.py",
        "base_library.zip:vendor/model/weights." + "safe" + "tensors",
        "runtime.zip:layer.whl:vision.vendor." + "cat" + "vton" + ".pipeline",
    ),
)
def test_packaged_archive_guard_normalizes_retired_entry_variants(entry):
    from scripts.hard_cutover_policy import retired_packaged_entries

    assert retired_packaged_entries([entry]) == [entry]


def test_packaged_archive_guard_allows_normal_stdlib_base_library_entries():
    from scripts.hard_cutover_policy import retired_packaged_entries

    entries = {
        "resource:_internal/base_library.zip",
        "base_library.zip:asyncio.base_events",
        "base_library.zip:importlib._bootstrap",
        "base_library.zip:email.message",
    }

    assert retired_packaged_entries(entries) == []


def test_frozen_spec_keeps_the_generated_v2_bundle_and_static_boundary_import():
    spec = (ROOT / "vending_vision.spec").read_text("utf-8")

    assert "CONTRACT_DATA_FILES" in spec
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
    assert '"contracts.vem_vision_v2.python.vision_v2_models"' in spec
    for retired in (
        "".join(("a", "i", "_attempt")),
        "".join(("a", "i", "_model_pack")),
        "".join(("cat", "vton")),
        "".join(("requirements-", "a", "i")),
        "".join(("regional", "_evaluator")),
    ):
        assert retired not in spec.lower()


def test_packaged_verifier_executes_the_frozen_bundle_positive_negative_probe():
    verifier = (ROOT / "scripts" / "verify_packaged_exe.py").read_text("utf-8")

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
    assert "PACKAGED_EXE_VERIFICATION=PASS" in verifier
    assert "assert_release_version_runtime_marker(exe_path)" in verifier
    assert "assert_hard_cutover_archive_absence(exe_path)" in verifier
    assert "assert_packaged_model_manifest(exe_path)" in verifier
    assert "retired modules remain in packaged archive" in verifier
    assert "retired_try_on_route" in verifier
    assert "assert_no_worker_resource_leak_output" in verifier


def test_packaged_verifier_checks_declared_model_roles_and_digests(tmp_path):
    from scripts.verify_packaged_exe import assert_packaged_model_manifest
    from vision.model_manifest import verify_model_manifest

    exe = tmp_path / "vending-vision" / "vending-vision.exe"
    models = exe.parent / "_internal" / "models"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    entries = []
    for role, relative in (
        ("person_detection", "models/person_detection/person.onnx"),
        ("face_detection", "models/face_detection/face.onnx"),
        ("age_network_definition", "models/age_gender/age.prototxt"),
        ("age_network_weights", "models/age_gender/age.caffemodel"),
        ("gender_network_definition", "models/age_gender/gender.prototxt"),
        ("gender_network_weights", "models/age_gender/gender.caffemodel"),
    ):
        path = exe.parent / "_internal" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode("ascii"))
        entries.append({"role": role, "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (models / "model-manifest.json").write_text(
        json.dumps({"schemaVersion": "vending-vision-model-manifest/v1", "models": entries}),
        "utf-8",
    )

    assert verify_model_manifest(exe.parent / "_internal")["ok"]
    assert_packaged_model_manifest(exe)
    entries[0]["sha256"] = "0" * 64
    (models / "model-manifest.json").write_text(
        json.dumps({"schemaVersion": "vending-vision-model-manifest/v1", "models": entries}),
        "utf-8",
    )
    with pytest.raises(AssertionError, match="production model manifest"):
        assert_packaged_model_manifest(exe)


def test_packaged_verifier_keeps_simple_ai_name_check_without_deep_container_audit(
    tmp_path, monkeypatch
):
    from scripts import verify_packaged_exe as verifier

    exe = tmp_path / "vending-vision" / "vending-vision.exe"
    internal = exe.parent / "_internal"
    (internal / "vendor").mkdir(parents=True)
    exe.write_bytes(b"MZ")
    (internal / "vendor" / "legal.tflite").write_bytes(b"tflite")
    (internal / "vendor" / "legal.gz").write_bytes(b"gzip")
    monkeypatch.setattr(verifier, "packaged_archive_entries", lambda _: {"PYZ.pyz:vision.runtime"})

    verifier.assert_hard_cutover_archive_absence(exe)
    retired_resource = "".join(("a", "i", "-runtime.json"))
    (internal / "vendor" / retired_resource).write_text("{}", "utf-8")
    with pytest.raises(AssertionError, match="retired modules"):
        verifier.assert_hard_cutover_archive_absence(exe)
    (internal / "vendor" / retired_resource).unlink()
    retired_module = "PYZ.pyz:vision." + "".join(("a", "i", "_attempt_worker"))
    monkeypatch.setattr(verifier, "packaged_archive_entries", lambda _: {retired_module})
    with pytest.raises(AssertionError, match="retired modules"):
        verifier.assert_hard_cutover_archive_absence(exe)


@pytest.mark.parametrize(
    ("retired_argv", "retired_option"),
    (
        (["--probe-runtime"], "--probe-runtime"),
        (["--probe"], "--probe"),
        (["--person", "sentinel-person.png"], "--person"),
        (["--garment", "sentinel-garment.png"], "--garment"),
        (["--output", "sentinel-output.png"], "--output"),
    ),
)
def test_main_runtime_cli_rejects_retired_child_worker_options(
    retired_argv, retired_option, capsys
):
    from run_vision_server import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(retired_argv)

    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "unrecognized arguments" in diagnostic
    assert retired_option in diagnostic


def test_packaged_verifier_rejects_missing_release_version_runtime_marker(tmp_path):
    from scripts.verify_packaged_exe import assert_release_version_runtime_marker

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"main")

    with pytest.raises(AssertionError, match="release version runtime marker"):
        assert_release_version_runtime_marker(exe)


@pytest.mark.parametrize(
    "marker",
    (
        'APP_VERSION = "1.2.3"\n',
        'APP_VERSION = "1.2.3-rc.12"\n',
        'APP_VERSION = "1.2.3-alpha-"\n',
    ),
)
def test_packaged_verifier_accepts_semver_release_version_runtime_marker(
    tmp_path, marker
):
    from scripts.verify_packaged_exe import assert_release_version_runtime_marker
    from vision.build_identity import write_packaged_build_identity

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    internal = exe.parent / "_internal"
    (internal / "vision").mkdir(parents=True)
    exe.write_bytes(b"main")
    version_marker = internal / "vision" / "_build_version.py"
    version_marker.write_text(marker, "utf-8")
    write_packaged_build_identity(
        version_marker,
        internal / "vision" / "_build_identity.json",
        "a" * 40,
    )

    assert_release_version_runtime_marker(exe)


@pytest.mark.parametrize(
    "marker",
    (
        'APP_VERSION = "1.2.3-.."\n',
        'APP_VERSION = "1.2.3-alpha..1"\n',
        'APP_VERSION = "1.2.3-01"\n',
        'APP_VERSION = "1.2.3+build..7"\n',
        'APP_VERSION = "1.2.3+build."\n',
        'APP_VERSION = "1.2.3+build_7"\n',
        'APP_VERSION = "1.2.3+build.7"\n',
        'APP_VERSION = "1.2.3-rc.12+build.007"\n',
    ),
)
def test_packaged_verifier_rejects_noncanonical_release_version_runtime_marker(
    tmp_path, marker
):
    from scripts.verify_packaged_exe import assert_release_version_runtime_marker

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    internal = exe.parent / "_internal"
    (internal / "vision").mkdir(parents=True)
    exe.write_bytes(b"main")
    (internal / "vision" / "_build_version.py").write_text(marker, "utf-8")
    (internal / "vision" / "_build_identity.json").write_text(
        json.dumps(
            {
                "appVersion": "1.2.3",
                "schemaVersion": "vending-vision-build-identity/v1",
                "sourceCommit": "a" * 40,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )

    with pytest.raises(AssertionError, match="invalid release version runtime marker"):
        assert_release_version_runtime_marker(exe)


def test_windows_build_preserves_the_single_core_supply_chain_gates():
    build = (ROOT / "scripts" / "build_exe.ps1").read_text("utf-8")

    for required in (
        'Get-Content (Join-Path $Root ".python-version")',
        "function Invoke-Checked",
        'A pre-validated offline core wheelhouse is required',
        'scripts\\bootstrap_build_envs.py',
        '--core-env $CoreVenv',
        '--core-wheelhouse $Wheelhouse',
        '--core-requirements (Join-Path $Root "requirements.txt")',
        'scripts\\dependency_lock.py',
        'from vision.model_manifest import verify_model_manifest',
        '-m PyInstaller --clean --noconfirm',
        'scripts\\verify_packaged_exe.py',
        'git -C $Root rev-parse HEAD',
        'git -C $Root status --porcelain --untracked-files=normal',
        'scripts\\write_packaged_build_identity.py',
        '_build_identity.json',
    ):
        assert required in build
    assert build.count("Invoke-Checked $CorePython") == 5
    assert '$CoreVenv = Join-Path $Root ".venv-packaging-core"' in build
    assert '$CoreDist = Join-Path $BuildDir "dist-core"' in build
    assert 'foreach ($Environment in @($CoreVenv))' in build
    assert 'foreach ($Output in @($CoreWork, $CoreDist, $FinalDist))' in build
    assert "worker" not in build.lower()


def test_windows_build_rejects_nonignored_untracked_and_allows_ignored_outputs(
    tmp_path,
):
    repository = tmp_path / "repository"
    _init_clean_repository(repository)
    untracked = repository / "runtime_extension.py"
    untracked.write_text("pass\n", "utf-8")
    command = [
        "pwsh",
        "-NoProfile",
        "-File",
        str(ROOT / "scripts" / "build_exe.ps1"),
        "-Wheelhouse",
        str(repository / "missing-wheelhouse"),
        "-SourceRoot",
        str(repository),
    ]

    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    rejected = subprocess.run(
        command, capture_output=True, text=True, env=environment
    )
    assert rejected.returncode != 0
    assert "Build source has tracked or non-ignored untracked changes" in (
        rejected.stdout + rejected.stderr
    )

    untracked.unlink()
    ignored = repository / "ignored-cache" / "runtime.cache"
    ignored.parent.mkdir()
    ignored.write_bytes(b"ignored build output")
    allowed = subprocess.run(command, capture_output=True, text=True, env=environment)
    combined = allowed.stdout + allowed.stderr
    assert allowed.returncode != 0
    assert "A pre-validated offline core wheelhouse is required" in combined
    assert "Build source has tracked or non-ignored untracked changes" not in combined


def test_windows_ci_runs_tests_and_digest_bound_packaging_in_parallel_before_publish():
    """Hosted publishing must join independent Windows test and packaging gates."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    test_job = ci.split("windows-test:", 1)[1].split("windows-package:", 1)[0]
    package_job = ci.split("windows-package:", 1)[1].split("publish-main-artifacts:", 1)[0]
    publish_job = ci.split("publish-main-artifacts:", 1)[1]

    assert re.search(
        r"needs:\n\s+- test\n\s+- windows-test\n\s+- windows-package",
        publish_job,
    )
    assert re.search(r"if: github\.ref == 'refs/heads/main'", publish_job)

    assert "python -m pytest -q" in test_job
    assert (
        "python -m py_compile app.py run_vision_server.py vision/*.py scripts/*.py"
        in test_job
    )
    assert "verify_model_manifest" in test_job
    assert "scripts/build_exe.ps1" not in test_job

    build_run = " ".join(
        package_job.split("构建并验证 Windows runtime 与录播交付包", 1)[1]
        .split("name: 暂存同提交 Vision artifacts", 1)[0]
        .split()
    )
    build = (
        "./scripts/build_exe.ps1 "
        "-Wheelhouse (Join-Path $PWD \"wheelhouse\") "
        "-SourceCommit \"${{ github.sha }}\""
    )

    assert build in build_run
    assert (
        '& (Join-Path $PWD ".venv-packaging-core\\Scripts\\python.exe") '
        "scripts/verify_packaged_exe.py"
    ) in build_run
    assert build_run.count("scripts/build_exe.ps1") == 1
    assert build_run.count("scripts/verify_packaged_exe.py") == 1
    assert build_run.count("scripts/package_main_artifacts.ps1") == 1
    assert '-Commit "${{ github.sha }}"' in build_run
    assert "worker-patch" not in package_job.lower()

    assert re.search(
        r"name: vending-vision-main-staging-\$\{\{ github\.sha \}\}\n"
        r"\s+path: main-artifacts/\*\n\s+if-no-files-found: error",
        package_job,
    )
    assert re.search(
        r"uses: actions/download-artifact@v4\n\s+with:\n"
        r"\s+name: vending-vision-main-staging-\$\{\{ github\.sha \}\}\n"
        r"\s+path: main-artifacts",
        publish_job,
    )
    assert re.search(
        r"uses: actions/upload-artifact@v4\n\s+with:\n"
        r"\s+name: vending-vision-main-\$\{\{ github\.sha \}\}\n"
        r"\s+path: main-artifacts/\*\n\s+if-no-files-found: error",
        publish_job,
    )


def test_linux_and_windows_ci_run_hash_pinned_focused_quality_gates_outside_runtime_closure():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    quality_lock = (ROOT / "requirements-quality.txt").read_text("utf-8")
    runtime_lock = (ROOT / "requirements.txt").read_text("utf-8")
    linux_job = ci.split("test:", 1)[1].split("windows-test:", 1)[0]
    windows_job = ci.split("windows-test:", 1)[1].split("windows-package:", 1)[0]
    package_job = ci.split("windows-package:", 1)[1].split(
        "publish-main-artifacts:", 1
    )[0]
    lint_command = (
        "python -m ruff check --select E4,E7,E9,F "
        "vision/build_identity.py vision/config.py vision/v2_contract_bundle.py "
        "scripts/hard_cutover_policy.py "
        "scripts/verify_packaged_exe.py scripts/write_packaged_build_identity.py"
    )
    type_command = (
        "python -m mypy --follow-imports=skip --ignore-missing-imports "
        "--check-untyped-defs "
        "vision/build_identity.py vision/config.py vision/v2_contract_bundle.py "
        "scripts/hard_cutover_policy.py "
        "scripts/verify_packaged_exe.py scripts/write_packaged_build_identity.py"
    )

    assert "ruff==0.12.11 \\\n" in quality_lock
    assert "mypy==1.17.1 \\\n" in quality_lock
    assert "mypy_extensions==1.1.0 \\\n" in quality_lock
    assert "pathspec==1.1.1 \\\n" in quality_lock
    assert "typing_extensions==4.16.0 \\\n" in quality_lock
    assert quality_lock.count("--hash=sha256:") == 7
    assert "ruff==" not in runtime_lock
    assert "mypy==" not in runtime_lock
    for job in (linux_job, windows_job):
        assert "requirements-quality.txt" in job
        assert "--require-hashes" in job
        assert lint_command in " ".join(job.split())
        assert type_command in " ".join(job.split())
    assert "requirements-quality.txt" not in package_job
    assert "ruff check" not in package_job
    assert "mypy" not in package_job


def _write_bound_geometry_fixture_manifest(recorded):
    source = ROOT / "fixtures" / "recorded-video" / "sources" / "person-man-front.png"
    source_target = recorded / "sources" / source.name
    source_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, source_target)
    (recorded / "generate-geometry-front.py").write_text("fixture generator", "utf-8")
    recordings = {}
    for key, filename in (
        ("geometryFar", "geometry-far.mp4"),
        ("geometryMid", "geometry-mid.mp4"),
        ("geometryNear", "geometry-near.mp4"),
    ):
        payload = filename.encode()
        (recorded / filename).write_bytes(payload)
        recordings[key] = {
            "file": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "loop": True,
            "source": "sources/person-man-front.png",
            "sourceSha256": hashlib.sha256(source_target.read_bytes()).hexdigest(),
            "generator": "generate-geometry-front.py",
        }
    (recorded / "expected-results.json").write_text(
        json.dumps(
            {
                "schemaVersion": "vending-vision-recorded-video-fixture/v1",
                "recordings": recordings,
            }
        ),
        "utf-8",
    )


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
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")
    _write_bound_geometry_fixture_manifest(recorded)
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
        artifact_manifest = archive.read("vision-artifact.json")
    delivery_manifest = (output / "vending-vision-main-artifacts.json").read_bytes()
    assert {
        "_internal/contracts/vem_vision_v2/fixtures/client-invalid.json",
        "_internal/contracts/vem_vision_v2/fixtures/client-valid.json",
        "_internal/contracts/vem_vision_v2/fixtures/server-invalid.json",
        "_internal/contracts/vem_vision_v2/fixtures/server-valid.json",
    } <= entries
    assert not artifact_manifest.startswith(b"\xef\xbb\xbf")
    assert not delivery_manifest.startswith(b"\xef\xbb\xbf")
    assert json.loads(artifact_manifest.decode("utf-8")) == {
        "schemaVersion": "vending-vision-main-artifacts/v1",
        "commit": "a" * 40,
        "runtimeArchive": "vending-vision-windows-x86_64.zip",
        "fixtureArchive": "vending-vision-test-fixtures.zip",
    }
    delivery = json.loads(delivery_manifest.decode("utf-8"))
    assert delivery["runtime"]["bytes"] == (output / "vending-vision-windows-x86_64.zip").stat().st_size
    assert delivery["fixtures"]["bytes"] == (output / "vending-vision-test-fixtures.zip").stat().st_size


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
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")
    _write_bound_geometry_fixture_manifest(recorded)

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


def test_main_artifact_fixture_archive_rejects_unbound_geometry_bytes(tmp_path):
    """Fixture archive acceptance requires its canonical recording manifest, not names alone."""
    root = _main_artifact_harness_root(tmp_path)
    runtime = root / "dist" / "vending-vision"
    _write_runtime_contract_fixtures(runtime, "contracts/vem_vision_v2/fixtures")
    (root / "fixtures" / "recorded-video" / "expected-results.json").write_text("{}", "utf-8")

    completed = _run_main_artifact_package(root)

    assert completed.returncode != 0
    assert "Fixture archive recording manifest is invalid" in completed.stdout + completed.stderr


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
    (recorded / "top.mp4").write_bytes(b"top")
    (recorded / "front.mp4").write_bytes(b"front")
    _write_bound_geometry_fixture_manifest(recorded)
    return root


@pytest.mark.parametrize("missing", ("geometry-far.mp4", "geometry-mid.mp4", "geometry-near.mp4"))
def test_main_artifact_fixture_archive_requires_every_geometry_recording(tmp_path, missing):
    """The Windows fixture artifact fails closed when a geometry journey clip is omitted."""
    root = _main_artifact_harness_root(tmp_path)
    runtime = root / "dist" / "vending-vision"
    _write_runtime_contract_fixtures(runtime, "contracts/vem_vision_v2/fixtures")
    (root / "fixtures" / "recorded-video" / missing).unlink()

    completed = _run_main_artifact_package(root)

    assert completed.returncode != 0
    assert "Fixture archive layout is incomplete" in completed.stdout + completed.stderr


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
    ).replace(
        "Assert-FixtureRecordingManifest $FixtureArchive",
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
            "recorded-video/geometry-far.mp4",
            "recorded-video/geometry-mid.mp4",
            "recorded-video/geometry-near.mp4",
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


def test_main_artifact_fixture_manifest_uses_canonical_zip_lookup_names(tmp_path):
    """A Windows-created backslash ZIP passes both layout and bound-byte checks."""
    root = _main_artifact_harness_root(tmp_path)
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
    artifact_manifest = json.dumps(
        {
            "schemaVersion": "vending-vision-main-artifacts/v1",
            "commit": "a" * 40,
            "runtimeArchive": "vending-vision-windows-x86_64.zip",
            "fixtureArchive": "vending-vision-test-fixtures.zip",
        }
    ).encode()
    with zipfile.ZipFile(output / "vending-vision-windows-x86_64.zip", "w") as archive:
        for entry in _runtime_contract_archive_entries("\\"):
            archive.writestr(
                entry,
                artifact_manifest if entry.replace("\\", "/") == "vision-artifact.json" else b"x",
            )
    recordings = {}
    source_bytes = (ROOT / "fixtures" / "recorded-video" / "sources" / "person-man-front.png").read_bytes()
    fixture_entries = {"sources/person-man-front.png": source_bytes, "generate-geometry-front.py": b"generator"}
    for key, filename in (("geometryFar", "geometry-far.mp4"), ("geometryMid", "geometry-mid.mp4"), ("geometryNear", "geometry-near.mp4")):
        payload = filename.encode()
        fixture_entries[filename] = payload
        recordings[key] = {
            "file": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "loop": True,
            "source": "sources/person-man-front.png",
            "sourceSha256": hashlib.sha256(source_bytes).hexdigest(),
            "generator": "generate-geometry-front.py",
        }
    fixture_entries["expected-results.json"] = json.dumps(
        {"schemaVersion": "vending-vision-recorded-video-fixture/v1", "recordings": recordings}
    ).encode()
    fixture_entries.update({"top.mp4": b"top", "front.mp4": b"front", "vision-artifact.json": artifact_manifest})
    with zipfile.ZipFile(output / "vending-vision-test-fixtures.zip", "w") as archive:
        for entry, payload in fixture_entries.items():
            archive.writestr(f"recorded-video\\{entry}" if entry != "vision-artifact.json" else entry, payload)

    completed = _run_main_artifact_package(root)

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
