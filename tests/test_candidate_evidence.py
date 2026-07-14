import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts import generate_candidate_evidence
from scripts.dependency_lock import read_hash_locked_requirements
from scripts.sign_candidate_evidence import DOCUMENTS, canonical_bytes
from vision.camera_binding import DurableReplayStore, MaintenanceCapabilityVerifier


ROOT = Path(__file__).resolve().parents[1]
SIGNER = "spki-sha256:" + "a" * 64


def test_release_lock_contains_hashes_for_full_runtime_and_packaging_closure():
    packages = read_hash_locked_requirements(ROOT / "requirements.txt")

    assert packages["opencv-contrib-python"]["version"] == "4.10.0.84"
    assert packages["pyinstaller"]["version"] == "6.16.0"
    assert packages["anyio"]["version"]
    assert packages["cv2-enumerate-cameras"]["hashes"]
    assert all(package["hashes"] for package in packages.values())


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


def test_packaged_smoke_managed_fixture_mints_exact_endpoint_capabilities(tmp_path):
    from scripts.verify_packaged_exe import create_managed_maintenance_fixture

    now = int(time.time())
    config_path, mint_capability = create_managed_maintenance_fixture(tmp_path, port=17893, now=now)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["schemaVersion"] == "vending-vision-site-config/v1"
    assert "mock_scenario" not in config
    assert config["maintenance_replay_path"].endswith("camera-maintenance-replay.sqlite")

    verifier = MaintenanceCapabilityVerifier(
        config["maintenance_capability_keyring_path"],
        config["maintenance_session_path"],
        DurableReplayStore(config["maintenance_replay_path"]),
        clock=lambda: now,
    )
    read = mint_capability("camera.read")
    assert verifier.verify(read, "camera.read")["scope"] == "camera.read"
    refresh = mint_capability("camera.refresh")
    assert verifier.verify(refresh, "camera.refresh")["scope"] == "camera.refresh"


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
