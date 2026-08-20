from pathlib import Path
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
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
    from scripts.candidate_artifact_manifest import retired_packaged_entries

    assert retired_packaged_entries([entry]) == [entry]


def test_packaged_archive_guard_allows_normal_stdlib_base_library_entries():
    from scripts.candidate_artifact_manifest import retired_packaged_entries

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


def _init_candidate_repository(root):
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Candidate Tests"], cwd=root, check=True)
    (root / ".gitignore").write_text("dist/\nignored-cache/\n", "utf-8")
    (root / ".python-version").write_text("3.11.13\n", "utf-8")
    (root / "tracked.txt").write_text("candidate source\n", "utf-8")
    source_model = root / "models" / "current" / "model.onnx"
    source_model.parent.mkdir(parents=True)
    source_model.write_bytes(b"current production model")
    source_manifest = root / "models" / "model-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": "vending-vision-model-manifest/v1",
                "models": [
                    {
                        "role": "current_model",
                        "path": "models/current/model.onnx",
                        "sha256": hashlib.sha256(source_model.read_bytes()).hexdigest(),
                    }
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        "utf-8",
    )
    subprocess.run(
        ["git", "add", ".gitignore", ".python-version", "tracked.txt", "models"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "commit", "-qm", "candidate source"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _copy_candidate_models(repository, dist):
    shutil.copytree(
        repository / "models",
        dist / "vending-vision" / "_internal" / "models",
    )


def _write_packaged_build_marker(dist, source_commit):
    from vision.build_identity import write_packaged_build_identity

    runtime_root = dist / "vending-vision" / "_internal" / "vision"
    marker = runtime_root / "_build_version.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('APP_VERSION = "1.2.3"\n', "utf-8")
    write_packaged_build_identity(
        marker, runtime_root / "_build_identity.json", source_commit
    )


def test_packaged_commit_identity_keeps_production_config_version_parseable(
    tmp_path, monkeypatch
):
    from vision import config as production_config

    dist = tmp_path / "dist"
    _write_packaged_build_marker(dist, "a" * 40)
    packaged_vision = dist / "vending-vision" / "_internal" / "vision"
    monkeypatch.setattr(production_config, "__file__", str(packaged_vision / "config.py"))

    assert production_config._load_build_app_version() == "1.2.3"
    assert production_config.load_runtime_build_identity().source_commit == "a" * 40


def test_candidate_archive_write_is_deterministic_and_binds_clean_head_and_build_marker(
    tmp_path,
):
    from scripts.candidate_artifact_manifest import (
        EMBEDDED_MANIFEST,
        write_candidate_archive,
    )

    repository = tmp_path / "repository"
    source_commit = _init_candidate_repository(repository)
    dist = repository / "dist"
    main = dist / "vending-vision" / "vending-vision.exe"
    main.parent.mkdir(parents=True)
    main.write_bytes(b"main")
    _write_packaged_build_marker(dist, source_commit)
    _copy_candidate_models(repository, dist)

    artifact = tmp_path / "candidate.zip"
    manifest_path = tmp_path / "candidate-manifest.json"
    first = write_candidate_archive(
        dist,
        artifact,
        manifest_path,
        source_commit=source_commit,
        repository_root=repository,
    )
    repeated_artifact = tmp_path / "candidate-repeat.zip"
    repeated_manifest = tmp_path / "candidate-repeat.manifest.json"
    second = write_candidate_archive(
        dist,
        repeated_artifact,
        repeated_manifest,
        source_commit=source_commit,
        repository_root=repository,
    )

    assert first == second
    assert artifact.read_bytes() == repeated_artifact.read_bytes()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["schemaVersion"] == "vending-vision-candidate-artifact/v3"
    assert manifest["sourceCommit"] == source_commit
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
            source_commit=source_commit,
            repository_root=repository,
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
            source_commit=source_commit,
            repository_root=repository,
        )


def test_candidate_archive_rejects_commit_marker_mismatch_and_dirty_head(tmp_path):
    from scripts.candidate_artifact_manifest import write_candidate_archive

    repository = tmp_path / "repository"
    source_commit = _init_candidate_repository(repository)
    dist = repository / "dist"
    executable = dist / "vending-vision" / "vending-vision.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"main")
    _write_packaged_build_marker(dist, "b" * 40)

    with pytest.raises(RuntimeError, match="candidate_build_commit"):
        write_candidate_archive(
            dist,
            tmp_path / "marker.zip",
            tmp_path / "marker.json",
            source_commit=source_commit,
            repository_root=repository,
        )

    _write_packaged_build_marker(dist, source_commit)
    (repository / "tracked.txt").write_text("dirty\n", "utf-8")
    with pytest.raises(RuntimeError, match="candidate_source_dirty"):
        write_candidate_archive(
            dist,
            tmp_path / "dirty.zip",
            tmp_path / "dirty.json",
            source_commit=source_commit,
            repository_root=repository,
        )


def test_candidate_archive_rejects_nonignored_untracked_and_allows_ignored_outputs(
    tmp_path,
):
    from scripts.candidate_artifact_manifest import write_candidate_archive

    repository = tmp_path / "repository"
    source_commit = _init_candidate_repository(repository)
    dist = repository / "dist"
    executable = dist / "vending-vision" / "vending-vision.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"main")
    _write_packaged_build_marker(dist, source_commit)
    _copy_candidate_models(repository, dist)

    untracked = repository / "runtime_extension.py"
    untracked.write_text("pass\n", "utf-8")
    with pytest.raises(RuntimeError, match="candidate_source_dirty"):
        write_candidate_archive(
            dist,
            tmp_path / "untracked.zip",
            tmp_path / "untracked.json",
            source_commit=source_commit,
            repository_root=repository,
        )

    untracked.unlink()
    ignored = repository / "ignored-cache" / "runtime.cache"
    ignored.parent.mkdir()
    ignored.write_bytes(b"ignored build output")
    write_candidate_archive(
        dist,
        tmp_path / "ignored.zip",
        tmp_path / "ignored.json",
        source_commit=source_commit,
        repository_root=repository,
    )


def test_candidate_archive_rejects_package_without_bound_model_manifest(tmp_path):
    from scripts.candidate_artifact_manifest import write_candidate_archive

    repository = tmp_path / "repository"
    source_commit = _init_candidate_repository(repository)
    dist = repository / "dist"
    executable = dist / "vending-vision" / "vending-vision.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"main")
    _write_packaged_build_marker(dist, source_commit)

    with pytest.raises(RuntimeError, match="candidate_model_manifest"):
        write_candidate_archive(
            dist,
            tmp_path / "missing-models.zip",
            tmp_path / "missing-models.json",
            source_commit=source_commit,
            repository_root=repository,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("path", "models/hidden/model.onnx"),
        ("role", "hidden_generative_model"),
        ("sha256", "0" * 64),
    ),
)
def test_candidate_archive_binds_model_path_role_and_digest_to_clean_source_manifest(
    tmp_path, field, value
):
    from scripts.candidate_artifact_manifest import write_candidate_archive

    repository = tmp_path / "repository"
    source_commit = _init_candidate_repository(repository)
    dist = repository / "dist"
    executable = dist / "vending-vision" / "vending-vision.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"main")
    _write_packaged_build_marker(dist, source_commit)
    _copy_candidate_models(repository, dist)
    packaged_manifest = (
        dist
        / "vending-vision"
        / "_internal"
        / "models"
        / "model-manifest.json"
    )
    payload = json.loads(packaged_manifest.read_text("utf-8"))
    payload["models"][0][field] = value
    packaged_manifest.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        "utf-8",
    )

    with pytest.raises(RuntimeError, match="candidate_model_manifest"):
        write_candidate_archive(
            dist,
            tmp_path / f"changed-{field}.zip",
            tmp_path / f"changed-{field}.json",
            source_commit=source_commit,
            repository_root=repository,
        )


def _zip_bytes(entries, *, compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return output.getvalue()


def _tar_bytes():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo("legacy-worker.py")
        payload = b"pass\n"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _mark_zip_encrypted(payload):
    encrypted = bytearray(payload)
    local = encrypted.find(b"PK\x03\x04")
    central = encrypted.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    encrypted[local + 6 : local + 8] = (
        int.from_bytes(encrypted[local + 6 : local + 8], "little") | 1
    ).to_bytes(2, "little")
    encrypted[central + 8 : central + 10] = (
        int.from_bytes(encrypted[central + 8 : central + 10], "little") | 1
    ).to_bytes(2, "little")
    return bytes(encrypted)


@pytest.mark.parametrize(
    ("archive_payload", "error"),
    (
        (
            _zip_bytes(
                [
                    (
                        "nested/runtime.whl",
                        _zip_bytes(
                            [("vision/" + "a" + "i" + "/attempt_worker.py", b"pass\n")]
                        ),
                    )
                ]
            ),
            "candidate_archive_retired",
        ),
        (_zip_bytes([("../escape.py", b"pass\n")]), "candidate_archive_unsafe_path"),
        (_mark_zip_encrypted(_zip_bytes([("safe.py", b"pass\n")])), "candidate_archive_encrypted"),
        (
            _zip_bytes([("zeros.bin", b"0" * (2 * 1024 * 1024))], compression=zipfile.ZIP_DEFLATED),
            "candidate_archive_ratio",
        ),
        (b"not a zip", "candidate_archive_uninspectable"),
    ),
)
def test_candidate_archive_recursively_rejects_unsafe_or_uninspectable_archives(
    tmp_path, archive_payload, error
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    archive = tmp_path / "base_library.zip"
    archive.write_bytes(archive_payload)

    with pytest.raises(RuntimeError, match=error):
        audit_packaged_archives([("vending-vision/_internal/base_library.zip", archive)])


def test_candidate_archive_accepts_bounded_stdlib_base_library(tmp_path):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    archive = tmp_path / "base_library.zip"
    archive.write_bytes(
        _zip_bytes(
            [
                ("asyncio/base_events.pyc", b"stdlib"),
                ("importlib/_bootstrap.pyc", b"stdlib"),
                ("email/message.pyc", b"stdlib"),
            ]
        )
    )

    audit_packaged_archives([("vending-vision/_internal/base_library.zip", archive)])


@pytest.mark.parametrize(
    ("name", "payload"),
    (
        ("runtime.dat", _tar_bytes()),
        ("runtime.dat", gzip.compress(b"legacy worker")),
        ("runtime.dat", b"7z\xbc\xaf\x27\x1c" + b"legacy worker"),
        ("runtime.tar", b"not really a tar archive"),
        ("runtime.gz", b"not really a gzip archive"),
        ("runtime.7z", b"not really a 7z archive"),
    ),
    ids=(
        "renamed-tar",
        "renamed-gzip",
        "renamed-7z",
        "tar-suffix",
        "gzip-suffix",
        "7z-suffix",
    ),
)
def test_candidate_archive_rejects_non_zip_container_suffixes_and_magic(
    tmp_path, name, payload
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    disguised = tmp_path / name
    disguised.write_bytes(payload)

    with pytest.raises(RuntimeError, match="candidate_archive_container"):
        audit_packaged_archives([(f"vending-vision/_internal/{name}", disguised)])


@pytest.mark.parametrize(
    "nested_payload",
    (
        _tar_bytes(),
        gzip.compress(b"legacy worker"),
        b"7z\xbc\xaf\x27\x1clegacy worker",
    ),
    ids=("tar", "gzip", "7z"),
)
def test_candidate_archive_rejects_non_zip_container_nested_in_zip(
    tmp_path, nested_payload
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    archive = tmp_path / "base_library.zip"
    archive.write_bytes(_zip_bytes([("vendor/runtime.dat", nested_payload)]))

    with pytest.raises(RuntimeError, match="candidate_archive_container"):
        audit_packaged_archives(
            [("vending-vision/_internal/base_library.zip", archive)]
        )


def test_candidate_archive_rejects_more_than_three_nested_archive_layers(tmp_path):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    payload = _zip_bytes([("stdlib.pyc", b"stdlib")])
    for depth in range(4):
        payload = _zip_bytes([(f"layer-{depth}.zip", payload)])
    archive = tmp_path / "base_library.zip"
    archive.write_bytes(payload)

    with pytest.raises(RuntimeError, match="candidate_archive_depth"):
        audit_packaged_archives([("vending-vision/_internal/base_library.zip", archive)])


@pytest.mark.parametrize("prefix", (b"", b"MZ-STUB-PREFIX"))
def test_candidate_archive_detects_parseable_zip_behind_nonarchive_name(
    tmp_path, prefix
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    disguised = tmp_path / "runtime.dat"
    disguised.write_bytes(
        prefix
        + _zip_bytes(
            [("vision/" + "a" + "i" + "/attempt_worker.py", b"pass\n")]
        )
    )

    with pytest.raises(RuntimeError, match="candidate_archive_retired"):
        audit_packaged_archives([("vending-vision/_internal/runtime.dat", disguised)])


def test_candidate_archive_rejects_model_suffix_inside_nested_zip(tmp_path):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    nested = tmp_path / "base_library.zip"
    nested.write_bytes(
        _zip_bytes([("vendor/generative/weights.ckpt", b"hidden model")])
    )

    with pytest.raises(RuntimeError, match="candidate_model_set"):
        audit_packaged_archives(
            [("vending-vision/_internal/base_library.zip", nested)]
        )


@pytest.mark.parametrize(
    "runtime_dependency",
    (
        "".join(("tor", "ch")),
        "".join(("torch", "vision")),
        "".join(("diff", "users")),
        "".join(("transform", "ers")),
        "".join(("acceler", "ate")),
        "".join(("safe", "tensors")),
        "".join(("huggingface", "_hub")),
    ),
)
def test_candidate_archive_rejects_each_retired_runtime_dependency_in_nested_wheel(
    tmp_path, runtime_dependency
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    wheel = _zip_bytes([(runtime_dependency + "/__init__.py", b"pass\n")])
    archive = tmp_path / "base_library.zip"
    archive.write_bytes(_zip_bytes([("nested/runtime.whl", wheel)]))

    with pytest.raises(RuntimeError, match="candidate_archive_retired"):
        audit_packaged_archives([("vending-vision/_internal/base_library.zip", archive)])


@pytest.mark.parametrize(
    "runtime_distribution",
    (
        "".join(("tor", "ch")),
        "".join(("torch", "vision")),
        "".join(("diff", "users")),
        "".join(("transform", "ers")),
        "".join(("acceler", "ate")),
        "".join(("safe", "tensors")),
        "".join(("huggingface", "_hub")),
    ),
)
def test_candidate_archive_rejects_each_retired_runtime_distribution_metadata(
    tmp_path, runtime_distribution
):
    from scripts.candidate_artifact_manifest import audit_packaged_archives

    wheel = _zip_bytes(
        [(runtime_distribution + "-1.0.dist-info/METADATA", b"Metadata-Version: 2.1\n")]
    )
    archive = tmp_path / "base_library.zip"
    archive.write_bytes(_zip_bytes([("nested/runtime.whl", wheel)]))

    with pytest.raises(RuntimeError, match="candidate_archive_retired"):
        audit_packaged_archives([("vending-vision/_internal/base_library.zip", archive)])


def test_candidate_models_reject_weight_not_declared_by_packaged_manifest(tmp_path):
    from scripts.candidate_artifact_manifest import audit_packaged_model_files

    declared = tmp_path / "declared.onnx"
    declared.write_bytes(b"declared")
    declared_digest = hashlib.sha256(declared.read_bytes()).hexdigest()
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": "vending-vision-model-manifest/v1",
                "models": [
                    {
                        "role": "person_detection",
                        "path": "models/person_detection/declared.onnx",
                        "sha256": declared_digest,
                    }
                ],
            }
        ),
        "utf-8",
    )
    undeclared = tmp_path / "undeclared.onnx"
    undeclared.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="candidate_model_set"):
        audit_packaged_model_files(
            [
                ("vending-vision/_internal/models/model-manifest.json", manifest),
                (
                    "vending-vision/_internal/models/person_detection/declared.onnx",
                    declared,
                ),
                ("vending-vision/_internal/models/hidden/undeclared.onnx", undeclared),
            ]
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        "vending-vision/_internal/assets/hidden-generative.onnx",
        "vending-vision/_internal/vendor/generative/weights.ckpt",
        "vending-vision/_internal/assets/hidden-generative.pb",
        "vending-vision/_internal/vendor/generative/weights.tflite",
    ),
)
def test_candidate_models_reject_model_suffix_outside_canonical_directory(
    tmp_path, relative_path
):
    from scripts.candidate_artifact_manifest import audit_packaged_model_files

    model_root = ROOT / "models"
    payload = [
        (
            "vending-vision/_internal/models/"
            + path.relative_to(model_root).as_posix(),
            path,
        )
        for path in model_root.rglob("*")
        if path.is_file()
    ]
    hidden_model = tmp_path / Path(relative_path).name
    hidden_model.write_bytes(b"undeclared generative model")

    with pytest.raises(RuntimeError, match="candidate_model_set"):
        audit_packaged_model_files([*payload, (relative_path, hidden_model)])


def test_candidate_models_accept_exact_current_production_manifest():
    from scripts.candidate_artifact_manifest import audit_packaged_model_files

    model_root = ROOT / "models"
    payload = [
        (
            "vending-vision/_internal/models/" + path.relative_to(model_root).as_posix(),
            path,
        )
        for path in model_root.rglob("*")
        if path.is_file()
    ]

    audit_packaged_model_files(payload)


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
    _init_candidate_repository(repository)
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
        "scripts/candidate_artifact_manifest.py scripts/hard_cutover_policy.py "
        "scripts/verify_packaged_exe.py scripts/write_packaged_build_identity.py"
    )
    type_command = (
        "python -m mypy --follow-imports=skip --ignore-missing-imports "
        "--check-untyped-defs "
        "vision/build_identity.py vision/config.py vision/v2_contract_bundle.py "
        "scripts/candidate_artifact_manifest.py scripts/hard_cutover_policy.py "
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
