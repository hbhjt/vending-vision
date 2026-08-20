from pathlib import Path
import json
import re
import shutil
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
    assert "retired modules remain in packaged archive" in verifier
    assert "retired_try_on_route" in verifier
    assert "assert_no_worker_resource_leak_output" in verifier


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
        'APP_VERSION = "1.2.3+build.7"\n',
        'APP_VERSION = "1.2.3-rc.12+build.007"\n',
    ),
)
def test_packaged_verifier_accepts_semver_release_version_runtime_marker(
    tmp_path, marker
):
    from scripts.verify_packaged_exe import assert_release_version_runtime_marker

    suffix = ".exe" if sys.platform == "win32" else ""
    exe = tmp_path / "vending-vision" / f"vending-vision{suffix}"
    internal = exe.parent / "_internal"
    (internal / "vision").mkdir(parents=True)
    exe.write_bytes(b"main")
    (internal / "vision" / "_build_version.py").write_text(marker, "utf-8")

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

    with pytest.raises(AssertionError, match="invalid release version runtime marker"):
        assert_release_version_runtime_marker(exe)


def test_candidate_archive_write_is_deterministic_and_binds_source_commit(tmp_path):
    from scripts.candidate_artifact_manifest import (
        EMBEDDED_MANIFEST,
        write_candidate_archive,
    )

    dist = tmp_path / "dist"
    main = dist / "vending-vision" / "vending-vision.exe"
    main.parent.mkdir(parents=True)
    main.write_bytes(b"main")

    artifact = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "candidate-manifest.json"
    first = write_candidate_archive(
        dist, artifact, manifest_path, source_commit="a" * 40
    )
    repeated_artifact = tmp_path / "candidate-repeat.zip"
    repeated_manifest = tmp_path / "candidate-repeat.manifest.json"
    second = write_candidate_archive(
        dist, repeated_artifact, repeated_manifest, source_commit="a" * 40
    )

    assert first == second
    assert artifact.read_bytes() == repeated_artifact.read_bytes()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["schemaVersion"] == "vending-vision-candidate-artifact/v3"
    assert manifest["sourceCommit"] == "a" * 40
    assert (
        manifest["bindings"]["mainExecutable"]["path"]
        == "vending-vision/vending-vision.exe"
    )
    assert set(manifest["bindings"]) == {"mainExecutable"}
    assert set(manifest["layout"]) == {"mainExecutable", "mainOnedir"}
    with zipfile.ZipFile(artifact) as archive:
        assert EMBEDDED_MANIFEST in archive.namelist()
        assert "vending-vision/vending-vision.exe" in archive.namelist()
        assert not any("worker" in name.lower() for name in archive.namelist())

    retired = dist / "vending-vision-worker" / "worker.exe"
    retired.parent.mkdir()
    retired.write_bytes(b"worker")
    with pytest.raises(RuntimeError, match="candidate_payload_layout"):
        write_candidate_archive(
            dist,
            tmp_path / "candidate-extra.zip",
            tmp_path / "candidate-extra.manifest.json",
            source_commit="a" * 40,
        )

    shutil.rmtree(retired.parent)
    retired_inside_main = dist / "vending-vision" / "".join(
        ("a", "i", "-runtime-descriptor.json")
    )
    retired_inside_main.write_bytes(b"retired")
    with pytest.raises(RuntimeError, match="candidate_payload_retired"):
        write_candidate_archive(
            dist,
            tmp_path / "candidate-retired.zip",
            tmp_path / "candidate-retired.manifest.json",
            source_commit="a" * 40,
        )


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
    ):
        assert required in build
    assert build.count("Invoke-Checked $CorePython") == 4
    assert '$CoreVenv = Join-Path $Root ".venv-packaging-core"' in build
    assert '$CoreDist = Join-Path $BuildDir "dist-core"' in build
    assert 'foreach ($Environment in @($CoreVenv))' in build
    assert 'foreach ($Output in @($CoreWork, $CoreDist, $FinalDist))' in build
    assert "worker" not in build.lower()


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
        "-Wheelhouse (Join-Path $PWD \"wheelhouse\")"
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
