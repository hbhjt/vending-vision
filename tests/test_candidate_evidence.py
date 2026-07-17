import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

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


def test_release_lock_contains_hashes_for_full_runtime_and_packaging_closure():
    packages = read_hash_locked_requirements(ROOT / "requirements.txt")

    assert packages["opencv-contrib-python"]["version"] == "4.10.0.84"
    assert packages["pyinstaller"]["version"] == "6.16.0"
    assert packages["anyio"]["version"]
    assert packages["cv2-enumerate-cameras"]["hashes"]
    assert all(package["hashes"] for package in packages.values())


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
        ],
    )

    generate_candidate_evidence.main()

    packages = json.loads((tmp_path / "candidate" / "vision-sbom.spdx.json").read_text(encoding="utf-8"))["packages"]
    sbom_licenses = {item["name"].lower(): item["licenseDeclared"] for item in packages}
    assert len(packages) == 62
    assert sbom_licenses == {normalized: LICENSE_OVERRIDES[normalized] for normalized in windows_lock}


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
    publish = (ROOT / ".github/workflows/publish-candidate.yml").read_text(encoding="utf-8")

    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.9"
    assert ci.count("--target-sys-platform win32") == 1
    assert "python -m pip install --no-index --find-links wheelhouse --require-hashes -r requirements.txt" in ci
    assert "python scripts/dependency_lock.py --wheelhouse wheelhouse --python python --target-sys-platform win32" in publish
    assert publish.index("--target-sys-platform win32") < publish.index("./scripts/build_exe.ps1")
    assert "python -m pip install --no-index --find-links wheelhouse --require-hashes -r requirements.txt" in publish


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
            "--output",
            str(output),
        ],
    )

    generate_candidate_evidence.main()

    sbom = json.loads((output / "vision-sbom.spdx.json").read_text(encoding="utf-8"))
    package = next(item for item in sbom["packages"] if item["name"] == "cv2-enumerate-cameras")
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
        ],
    )

    generate_candidate_evidence.main()

    packages = json.loads((tmp_path / "candidate" / "vision-sbom.spdx.json").read_text(encoding="utf-8"))["packages"]
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
