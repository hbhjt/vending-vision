"""Admit and canonicalize the two untrusted pre-cutover proof envelopes."""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit


PROOF_KEYS = (
    "candidate_archive_url",
    "candidate_archive_sha256",
    "candidate_archive_bytes",
    "candidate_manifest_url",
    "candidate_manifest_sha256",
    "candidate_manifest_bytes",
    "candidate_attestation_url",
    "candidate_attestation_sha256",
    "candidate_attestation_bytes",
    "candidate_evidence_url",
    "candidate_evidence_sha256",
    "candidate_evidence_bytes",
    "model_pack_url",
    "model_pack_sha256",
    "model_pack_bytes",
    "model_pack_part_01_url",
    "model_pack_part_01_sha256",
    "model_pack_part_01_bytes",
    "model_pack_part_02_url",
    "model_pack_part_02_sha256",
    "model_pack_part_02_bytes",
    "model_pack_part_03_url",
    "model_pack_part_03_sha256",
    "model_pack_part_03_bytes",
)
COMPANION_KEYS = (
    "artifact_name",
    "archive_file",
    "archive_sha256",
    "descriptor_file",
    "descriptor_sha256",
    "attestation_bundle_file",
    "attestation_bundle_sha256",
)
MAX_SAFE_INTEGER = (1 << 53) - 1


class AdmissionError(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AdmissionError(f"duplicate_key:{key}")
        value[key] = item
    return value


def _parse(raw: str, label: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise AdmissionError(f"{label}_type")
    try:
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=lambda value: (_ for _ in ()).throw(AdmissionError(f"{label}_constant:{value}")))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AdmissionError(f"{label}_json") from exc
    if not isinstance(value, dict):
        raise AdmissionError(f"{label}_shape")
    return value


def _canonical_url(value: object, *, optional: bool) -> bool:
    if not isinstance(value, str):
        return False
    if optional and value == "":
        return True
    if any(ord(character) < 0x20 for character in value):
        return False
    parsed = urlsplit(value)
    return bool(
        parsed.scheme == "https"
        and parsed.netloc
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
        and value == parsed.geturl()
    )


def _reject_unsafe_string(value: object, label: str) -> None:
    if not isinstance(value, str):
        return
    if any(character in value for character in "${}"):
        raise AdmissionError(f"placeholder:{label}")
    if any(ord(character) < 0x20 for character in value):
        raise AdmissionError(f"control:{label}")


def _validate_proof(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != set(PROOF_KEYS):
        raise AdmissionError("proof_inputs_key_set")
    for key in PROOF_KEYS:
        item = value[key]
        _reject_unsafe_string(item, f"proof_inputs:{key}")
        if key.endswith("_bytes"):
            optional = key.startswith("model_pack_part_")
            if type(item) is not int or item < (0 if optional else 1) or item > MAX_SAFE_INTEGER:
                raise AdmissionError(f"proof_inputs_type:{key}")
        elif key.endswith("_sha256"):
            optional = key.startswith("model_pack_part_")
            if not isinstance(item, str) or (
                item != "" and re.fullmatch(r"[a-f0-9]{64}", item) is None
            ) or (not optional and item == ""):
                raise AdmissionError(f"proof_inputs_identity:{key}")
        elif key.endswith("_url") and not _canonical_url(
            item, optional=key == "model_pack_url" or key.startswith("model_pack_part_")
        ):
            raise AdmissionError(f"proof_inputs_identity:{key}")

    whole = value["model_pack_url"] != ""
    part_keys = tuple(key for key in PROOF_KEYS if key.startswith("model_pack_part_"))
    parts = all(value[key] not in ("", 0) for key in part_keys)
    parts_empty = all(value[key] in ("", 0) for key in part_keys)
    if not ((whole and parts_empty) or (not whole and parts)):
        raise AdmissionError("model_mode")
    return {key: value[key] for key in PROOF_KEYS}


def _validate_companion(value: dict[str, Any]) -> dict[str, str]:
    if set(value) != set(COMPANION_KEYS):
        raise AdmissionError("companion_builder_outputs_key_set")
    result: dict[str, str] = {}
    for key in COMPANION_KEYS:
        item = value[key]
        _reject_unsafe_string(item, f"companion_builder_outputs:{key}")
        if not isinstance(item, str) or not item or any(ord(character) < 0x20 for character in item):
            raise AdmissionError(f"companion_builder_outputs_type:{key}")
        if key.endswith("_sha256") and re.fullmatch(r"[a-f0-9]{64}", item) is None:
            raise AdmissionError(f"companion_builder_outputs_identity:{key}")
        result[key] = item
    return result


def admit_inputs(proof_inputs: str, companion_builder_outputs: str) -> tuple[str, str]:
    proof = _validate_proof(_parse(proof_inputs, "proof_inputs"))
    companion = _validate_companion(
        _parse(companion_builder_outputs, "companion_builder_outputs")
    )
    options = {"ensure_ascii": True, "separators": (",", ":")}
    return json.dumps(proof, **options), json.dumps(companion, **options)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-inputs-env", required=True)
    parser.add_argument("--companion-builder-outputs-env", required=True)
    parser.add_argument("--github-output", required=True)
    args = parser.parse_args()
    try:
        proof, companion = admit_inputs(
            os.environ[args.proof_inputs_env],
            os.environ[args.companion_builder_outputs_env],
        )
        with open(args.github_output, "a", encoding="utf-8", newline="\n") as output:
            output.write(f"proof_inputs={proof}\n")
            output.write(f"companion_builder_outputs={companion}\n")
    except (AdmissionError, KeyError, OSError) as exc:
        print(f"TRUSTED_PRECUTOVER_INPUT_ADMISSION=FAIL:{exc}")
        return 1
    print("TRUSTED_PRECUTOVER_INPUT_ADMISSION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
