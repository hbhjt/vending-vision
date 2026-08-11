from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


DOCUMENTS = {
    "descriptor": "vision-release-descriptor.json",
    "attestation": "vision-artifact-attestation.json",
    "sbom": "vision-sbom.spdx.json",
    "provenance": "vision-provenance.json",
}
SIGNER_IDENTITY = re.compile(r"^spki-sha256:[a-f0-9]{64}$")


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def signer_public_key(private_key, openssl):
    return subprocess.run(
        [str(openssl), "pkey", "-in", str(private_key), "-pubout", "-outform", "DER"],
        check=True,
        capture_output=True,
    ).stdout


def signer_identity(public_key_der):
    return "spki-sha256:" + hashlib.sha256(public_key_der).hexdigest()


def sign_document(role, document, private_key, expected_identity, public_key_der, openssl):
    document_digest = "sha256:" + hashlib.sha256(document.read_bytes()).hexdigest()
    statement = canonical_bytes({"role": role, "digest": document_digest})
    with tempfile.TemporaryDirectory() as temporary_directory:
        statement_path = Path(temporary_directory) / "statement.json"
        statement_path.write_bytes(statement)
        signature = subprocess.run(
            [
                str(openssl),
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key),
                "-in",
                str(statement_path),
            ],
            check=True,
            capture_output=True,
        ).stdout
    return {
        "signer": {
            "identity": expected_identity,
            "publicKey": base64.b64encode(public_key_der).decode("ascii"),
        },
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def write_json(path, value):
    path.write_bytes(canonical_bytes(value) + b"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--signer-identity", required=True)
    parser.add_argument("--openssl", required=True)
    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir).resolve()
    private_key = Path(args.private_key).resolve()
    openssl = Path(args.openssl).resolve()
    if not openssl.is_file():
        raise SystemExit("absolute OpenSSL executable is required")
    if not SIGNER_IDENTITY.fullmatch(args.signer_identity):
        raise SystemExit("signer identity must be spki-sha256:<64 lowercase hex>")
    public_key_der = signer_public_key(private_key, openssl)
    actual_identity = signer_identity(public_key_der)
    if actual_identity != args.signer_identity:
        raise SystemExit("supplier key identity mismatch")

    for role, file_name in DOCUMENTS.items():
        document = candidate_dir / file_name
        if not document.is_file():
            raise SystemExit(f"candidate document missing: {file_name}")
        envelope = sign_document(
            role,
            document,
            private_key,
            actual_identity,
            public_key_der,
            openssl,
        )
        write_json(candidate_dir / f"{file_name}.sig.json", envelope)


if __name__ == "__main__":
    main()
