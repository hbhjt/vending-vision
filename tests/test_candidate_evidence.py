import base64
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import generate_candidate_evidence
from scripts.dependency_lock import (
    LICENSE_OVERRIDES,
    active_hash_locked_requirements,
    installed_distributions,
    read_hash_locked_requirements,
    resolved_license,
    selected_wheels,
)
from scripts.sign_candidate_evidence import DOCUMENTS, canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
SIGNER = "spki-sha256:" + "a" * 64


def _ai_evidence_args(tmp_path):
    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    if not bundle.exists():
        bundle.write_bytes(b"candidate-bundle")
    worker = tmp_path / "vending-vision-ai-worker.exe"
    worker.write_bytes(b"stage23-worker")
    runtime = ROOT / "ai-runtime-descriptor.json"
    lock = ROOT / "requirements-ai.lock.json"
    source = ROOT / "official-ai-source-descriptor.json"
    model = ROOT / "official-ai-model-pack-descriptor.json"
    bindings = {
        name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in {
            "workerExecutable": worker,
            "runtimeDescriptor": runtime,
            "aiLock": lock,
            "sourceDescriptor": source,
            "modelPackDescriptor": model,
        }.items()
    }
    manifest = tmp_path / "candidate.manifest.json"
    manifest.write_text(
        json.dumps(
            {"schemaVersion": "vending-vision-candidate-artifact/v3", "bindings": bindings},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "utf-8",
    )
    github_attestation = tmp_path / "github-attestation.json"
    github_attestation.write_text('{"verifiedBy":"stage23-test"}', "utf-8")
    trusted_builder_evidence = tmp_path / "trusted-builder-evidence.json"
    trusted_builder_evidence.write_text(
        json.dumps(
            {
                "schemaVersion": "vending-vision-trusted-builder-evidence/v1",
                "builderRepository": "hbhjt/vending-vision",
                "builderWorkflow": ".github/workflows/trusted-ai-candidate-builder.yml",
                "builderWorkflowSha": "691b5056e8b9bf2667bc527b2170780b05863946",
                "sourceCommit": "b" * 40,
                "subjectSha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "embeddedManifestSha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "attestationBundleSha256": hashlib.sha256(github_attestation.read_bytes()).hexdigest(),
            },
            separators=(",", ":"),
        ),
        "utf-8",
    )
    return [
        "--candidate-manifest", str(manifest),
        "--github-attestation", str(github_attestation),
        "--trusted-builder-evidence", str(trusted_builder_evidence),
        "--ai-requirements-lock", str(lock),
        "--ai-runtime-descriptor", str(runtime),
        "--source-descriptor", str(source),
        "--model-pack-descriptor", str(model),
        "--worker-executable", str(worker),
    ]


def test_release_lock_contains_hashes_for_full_runtime_and_packaging_closure():
    packages = read_hash_locked_requirements(ROOT / "requirements.txt")

    assert packages["opencv-contrib-python"]["version"] == "4.10.0.84"
    assert packages["pyinstaller"]["version"] == "6.16.0"
    assert packages["anyio"]["version"]
    assert packages["cv2-enumerate-cameras"]["hashes"]
    assert all(package["hashes"] for package in packages.values())


def test_candidate_cli_bootstraps_repo_imports_for_help_and_full_generation(tmp_path):
    script = ROOT / "scripts" / "generate_candidate_evidence.py"
    help_result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--wheelhouse" in help_result.stdout

    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    bundle.write_bytes(b"candidate-bundle")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    pip_version = importlib.metadata.version("pip")
    wheel = wheelhouse / f"pip-{pip_version}-py3-none-any.whl"
    wheel.write_bytes(b"candidate wheel fixture")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        f"pip=={pip_version} --hash=sha256:{wheel_hash}\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidate"
    generation = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bundle",
            str(bundle),
            "--version",
            "0.2.1-rc.1",
            "--commit",
            "b" * 40,
            "--repository",
            "hbhjt/vending-vision",
            "--signer-identity",
            SIGNER,
            "--requirements-lock",
            str(lock),
            "--wheelhouse",
            str(wheelhouse),
            "--python",
            sys.executable,
            *_ai_evidence_args(tmp_path),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generation.returncode == 0, generation.stderr
    descriptor = json.loads((output / "vision-release-descriptor.json").read_text("utf-8"))
    assert descriptor["protocol"]["version"] == "vem.vision.v2"


def test_release_lock_includes_the_windows_click_console_dependency_in_the_win32_closure():
    packages = read_hash_locked_requirements(ROOT / "requirements.txt")

    assert packages["colorama"]["version"] == "0.4.6"
    assert packages["colorama"]["marker"] == 'sys_platform == "win32"'
    assert packages["colorama"]["hashes"]
    assert "colorama" in active_hash_locked_requirements(packages, {"sys_platform": "win32"})
    assert "colorama" not in active_hash_locked_requirements(packages, {"sys_platform": "linux"})


def test_win32_release_closure_has_reviewed_spdx_for_every_locked_package():
    windows_lock = active_hash_locked_requirements(
        read_hash_locked_requirements(ROOT / "requirements.txt"), {"sys_platform": "win32"}
    )

    assert len(windows_lock) == 62
    assert set(windows_lock) == set(LICENSE_OVERRIDES)
    assert LICENSE_OVERRIDES["colorama"] == "BSD-3-Clause"


def test_candidate_sbom_covers_the_actual_win32_locked_closure(tmp_path, monkeypatch):
    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    bundle.write_bytes(b"candidate-bundle")
    windows_lock = active_hash_locked_requirements(
        read_hash_locked_requirements(ROOT / "requirements.txt"), {"sys_platform": "win32"}
    )
    dependencies = [
        {
            **entry,
            "license": LICENSE_OVERRIDES[normalized],
            "wheel": {
                "filename": f"{normalized}-{entry['version']}.whl",
                "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            },
        }
        for normalized, entry in sorted(windows_lock.items())
    ]
    monkeypatch.setattr(generate_candidate_evidence, "verify_dependency_closure", lambda *_: dependencies)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_candidate_evidence.py", "--bundle", str(bundle), "--version", "0.2.1-rc.1",
            "--commit", "b" * 40, "--repository", "hbhjt/vending-vision",
            "--signer-identity", SIGNER, "--wheelhouse", str(tmp_path), "--output", str(tmp_path / "candidate"),
            *_ai_evidence_args(tmp_path),
        ],
    )

    generate_candidate_evidence.main()

    all_packages = json.loads((tmp_path / "candidate" / "vision-sbom.spdx.json").read_text(encoding="utf-8"))["packages"]
    packages = [item for item in all_packages if item.get("comment") == "scope=core-runtime"]
    ai_packages = [item for item in all_packages if item.get("comment") == "scope=ai-worker-runtime"]
    sbom_licenses = {item["name"].lower(): item["licenseDeclared"] for item in packages}
    assert len(packages) == 62
    assert sbom_licenses == {normalized: LICENSE_OVERRIDES[normalized] for normalized in windows_lock}
    assert len(ai_packages) == 34
    assert {item["name"] for item in ai_packages} >= {"torch", "torchvision", "diffusers", "transformers"}
    descriptor = json.loads((tmp_path / "candidate" / "vision-release-descriptor.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "contracts" / "vem_vision_v2" / "manifest.json").read_text(encoding="utf-8"))
    assert descriptor["protocol"] == {
        "version": manifest["protocol"],
        "schemaVersion": manifest["schemaVersion"],
        "bundleVersion": manifest["bundleVersion"],
        "contractDigest": manifest["bundleDigest"],
        "webSocketPath": "/ws",
    }
    assert descriptor["candidateManifest"]["schemaVersion"] == "vending-vision-candidate-artifact/v3"
    assert descriptor["githubArtifactAttestation"]["format"] == "sigstore-bundle"
    assert descriptor["trustedBuilderEvidence"]["schemaVersion"] == "vending-vision-trusted-builder-evidence/v1"
    assert set(descriptor["aiRuntime"]) == {
        "requirementsLock", "runtimeDescriptor", "sourceDescriptor",
        "modelPackDescriptor", "workerExecutable",
    }
    supplier_attestation = json.loads(
        (tmp_path / "candidate" / "vision-artifact-attestation.json").read_text("utf-8")
    )
    assert supplier_attestation["candidateManifestDigest"] == descriptor["candidateManifest"]["digest"]
    assert supplier_attestation["githubArtifactAttestationDigest"] == descriptor["githubArtifactAttestation"]["digest"]
    assert supplier_attestation["trustedBuilderEvidenceDigest"] == descriptor["trustedBuilderEvidence"]["digest"]
    assert supplier_attestation["attestedSubjectDigest"] == descriptor["bundle"]["digest"]
    assert supplier_attestation["workerExecutableDigest"] == descriptor["aiRuntime"]["workerExecutable"]["digest"]
    provenance = json.loads((tmp_path / "candidate" / "vision-provenance.json").read_text("utf-8"))
    dependency_uris = {
        item["uri"]
        for item in provenance["predicate"]["buildDefinition"]["resolvedDependencies"]
    }
    assert any(uri.startswith("github-attestation:") for uri in dependency_uris)
    assert any(uri.startswith("ai-worker:") for uri in dependency_uris)


def test_win32_marked_lock_selects_a_complete_offline_wheel_closure(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "colorama-0.4.6-py2.py3-none-any.whl"
    wheel.write_bytes(b"Windows marker closure fixture")
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text(
        'colorama==0.4.6 ; sys_platform == "win32" \\\n'
        f'    --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n',
        encoding="utf-8",
    )

    windows_lock = active_hash_locked_requirements(
        read_hash_locked_requirements(lock_path), {"sys_platform": "win32"}
    )

    assert selected_wheels(windows_lock, wheelhouse) == {
        "colorama": {"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    }


def test_windows_ci_and_candidate_publish_force_the_win32_offline_closure():
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    from scripts.workflow_yaml import load_workflow_yaml

    ci_workflow = load_workflow_yaml(ci)
    publish = (ROOT / ".github/workflows/publish-candidate.yml").read_text(encoding="utf-8")
    builder = (ROOT / ".github/workflows/trusted-ai-candidate-builder.yml").read_text(encoding="utf-8")

    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.9"
    for job_name in ("windows-test", "windows-package"):
        runs = "\n".join(
            step.get("run", "")
            for step in ci_workflow["jobs"][job_name]["steps"]
            if isinstance(step, dict)
        )
        assert "--target-sys-platform win32" in runs
        assert (
            "python -m pip install --no-index --find-links wheelhouse "
            "--require-hashes -r requirements.txt"
        ) in runs
    assert "dependency_lock.py" not in publish
    assert "python -m pip install" not in publish
    assert "scripts/build_exe.ps1" not in publish
    assert builder.count("scripts/build_exe.ps1") == 1
    assert ".venv-packaging-core" in builder


def test_reviewed_license_facts_match_the_locked_cffi_and_pillow_wheel_metadata():
    installed = installed_distributions(sys.executable)

    assert resolved_license("cffi", installed["cffi"]) == "MIT-0"
    assert resolved_license("pillow", installed["pillow"]) == "MIT-CMU"


def test_candidate_sbom_uses_hash_locked_installed_wheels_and_real_gpl_license(tmp_path, monkeypatch):
    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    bundle.write_bytes(b"candidate-bundle")
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "cv2_enumerate_cameras-1.1.16-py3-none-any.whl"
    wheel.write_bytes(b"selected wheel bytes")
    wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "cv2-enumerate-cameras==1.1.16 \\\n"
        f"    --hash=sha256:{wheel_hash}\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_candidate_evidence.py",
            "--bundle",
            str(bundle),
            "--version",
            "0.2.1-rc.1",
            "--commit",
            "b" * 40,
            "--repository",
            "hbhjt/vending-vision",
            "--signer-identity",
            SIGNER,
            "--requirements-lock",
            str(lock),
            "--wheelhouse",
            str(wheelhouse),
            "--python",
            sys.executable,
            *_ai_evidence_args(tmp_path),
            "--output",
            str(output),
        ],
    )

    generate_candidate_evidence.main()

    sbom = json.loads((output / "vision-sbom.spdx.json").read_text(encoding="utf-8"))
    package = next(
        item for item in sbom["packages"]
        if item["name"] == "cv2-enumerate-cameras" and item.get("comment") == "scope=core-runtime"
    )
    assert package["licenseDeclared"] == "GPL-3.0-or-later"
    assert package["licenseConcluded"] == "GPL-3.0-or-later"
    assert package["checksums"] == [{"algorithm": "SHA256", "checksumValue": wheel_hash}]
    assert package["downloadLocation"] != "NOASSERTION"


def test_candidate_sbom_declares_and_concludes_the_reviewed_cffi_and_pillow_facts(tmp_path, monkeypatch):
    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    bundle.write_bytes(b"candidate-bundle")
    dependencies = [
        {
            "name": name,
            "version": version,
            "license": license_expression,
            "wheel": {"filename": f"{name}-{version}.whl", "sha256": hashlib.sha256(name.encode()).hexdigest()},
        }
        for name, version, license_expression in (
            ("cffi", "2.1.0", "MIT-0"),
            ("Pillow", "12.3.0", "MIT-CMU"),
        )
    ]
    monkeypatch.setattr(generate_candidate_evidence, "verify_dependency_closure", lambda *_: dependencies)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_candidate_evidence.py", "--bundle", str(bundle), "--version", "0.2.1-rc.1",
            "--commit", "b" * 40, "--repository", "hbhjt/vending-vision",
            "--signer-identity", SIGNER, "--wheelhouse", str(tmp_path), "--output", str(tmp_path / "candidate"),
            *_ai_evidence_args(tmp_path),
        ],
    )

    generate_candidate_evidence.main()

    packages = [
        item
        for item in json.loads((tmp_path / "candidate" / "vision-sbom.spdx.json").read_text(encoding="utf-8"))["packages"]
        if item.get("comment") == "scope=core-runtime"
    ]
    facts = {item["name"].lower(): (item["licenseDeclared"], item["licenseConcluded"]) for item in packages}
    assert facts == {"cffi": ("MIT-0", "MIT-0"), "pillow": ("MIT-CMU", "MIT-CMU")}


def test_packaged_smoke_managed_fixture_uses_plain_maintenance_contract(tmp_path):
    from scripts.verify_packaged_exe import create_managed_maintenance_fixture

    fixture_root = tmp_path / "managed-production"
    config_path = create_managed_maintenance_fixture(fixture_root, port=17893)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert fixture_root.is_dir()
    assert config["schemaVersion"] == "vending-vision-site-config/v1"
    assert "mock_scenario" not in config
    assert not {"maintenance_capability_keyring_path", "maintenance_session_path", "maintenance_replay_path"} & set(config)


def test_packaged_resource_verifier_rejects_retired_maintenance_v1_schema(tmp_path):
    from scripts.verify_packaged_exe import assert_bundled_resources

    internal = tmp_path / "_internal"
    for path in [
        internal / "config.json",
        internal / "dashboard" / "profile_dashboard.html",
        internal / "config" / "vending-vision-camera-maintenance-v2.schema.json",
        internal / "config" / "vending-vision-camera-maintenance-v2.requests.schema.json",
        internal / "config" / "vending-vision-camera-maintenance-v2.responses.schema.json",
        internal / "models" / "person_detection" / "person_yolov8n.onnx",
        internal / "models" / "face_detection" / "face_detection_yunet_2023mar.onnx",
        internal / "models" / "age_gender" / "age_net.caffemodel",
        internal / "models" / "age_gender" / "gender_net.caffemodel",
        internal / "cv2_enumerate_cameras" / "_windows_backend_test.pyd",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"packaged")
    for relative_path in [
        "manifest.json",
        "__init__.py",
        "python/__init__.py",
        "python/vision_v2_models.py",
        "vision-v2.client.schema.json",
        "vision-v2.server.schema.json",
        "fixtures/client-valid.json",
        "fixtures/client-invalid.json",
        "fixtures/server-valid.json",
        "fixtures/server-invalid.json",
    ]:
        source = ROOT / "contracts" / "vem_vision_v2" / relative_path
        destination = internal / "contracts" / "vem_vision_v2" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (internal / "config" / "vending-vision-camera-maintenance-v1.schema.json").write_bytes(b"retired")

    with pytest.raises(AssertionError, match="retired packaged resources"):
        assert_bundled_resources(tmp_path / "vending-vision.exe")


def test_candidate_signatures_match_vem_role_digest_contract(tmp_path):
    private_key = tmp_path / "supplier.pem"
    public_key = tmp_path / "supplier-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
        check=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
        check=True,
    )
    public_der = subprocess.run(
        ["openssl", "pkey", "-pubin", "-in", str(public_key), "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout
    identity = "spki-sha256:" + hashlib.sha256(public_der).hexdigest()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for role, file_name in DOCUMENTS.items():
        (candidate / file_name).write_bytes(canonical_bytes({"roleFixture": role}) + b"\n")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sign_candidate_evidence.py"),
            "--candidate-dir",
            str(candidate),
            "--private-key",
            str(private_key),
            "--signer-identity",
            identity,
            "--openssl",
            shutil.which("openssl"),
        ],
        check=True,
    )

    for role, file_name in DOCUMENTS.items():
        document = candidate / file_name
        envelope = json.loads((candidate / f"{file_name}.sig.json").read_text(encoding="utf-8"))
        assert envelope["signer"] == {
            "identity": identity,
            "publicKey": base64.b64encode(public_der).decode("ascii"),
        }
        statement = tmp_path / f"{role}.statement"
        signature = tmp_path / f"{role}.signature"
        digest = "sha256:" + hashlib.sha256(document.read_bytes()).hexdigest()
        statement.write_bytes(canonical_bytes({"role": role, "digest": digest}))
        signature.write_bytes(base64.b64decode(envelope["signature"]))
        subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public_key),
                "-rawin",
                "-in",
                str(statement),
                "-sigfile",
                str(signature),
            ],
            check=True,
        )
