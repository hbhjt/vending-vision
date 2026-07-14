import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts import generate_candidate_evidence
from scripts.sign_candidate_evidence import DOCUMENTS, canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
SIGNER = "spki-sha256:" + "a" * 64


def test_candidate_sbom_expands_shared_runtime_and_packaging_pins(tmp_path, monkeypatch):
    bundle = tmp_path / "vending-vision-0.2.1-rc.1-windows-x86_64.zip"
    bundle.write_bytes(b"candidate-bundle")
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
            "--output",
            str(output),
        ],
    )

    generate_candidate_evidence.main()

    sbom = json.loads((output / "vision-sbom.spdx.json").read_text(encoding="utf-8"))
    packages = {package["name"]: package["versionInfo"] for package in sbom["packages"]}
    assert packages["opencv-contrib-python"] == "4.10.0.84"
    assert packages["pyinstaller"] == "6.16.0"
    assert "-r requirements.txt" not in packages


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
