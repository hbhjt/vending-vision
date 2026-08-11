from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUNTIME_DESCRIPTOR_PATH = Path(__file__).resolve().parents[1] / "ai-runtime-descriptor.json"


class AiRuntimeDescriptorError(RuntimeError):
    pass


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_runtime_descriptor() -> str:
    return hashlib.sha256(RUNTIME_DESCRIPTOR_PATH.read_bytes()).hexdigest()


def load_ai_runtime_descriptor() -> dict:
    raw = RUNTIME_DESCRIPTOR_PATH.read_text("utf-8")
    descriptor = json.loads(raw)
    if canonical_json(descriptor) != raw.rstrip("\n"):
        raise AiRuntimeDescriptorError("ai_runtime_descriptor_noncanonical")
    if set(descriptor) != {
        "schemaVersion",
        "target",
        "python",
        "directRequirements",
        "requirementsAiSha256",
        "requirementsAiLockSha256",
        "workerLayout",
    }:
        raise AiRuntimeDescriptorError("ai_runtime_descriptor_shape")
    if descriptor["schemaVersion"] != "vem-ai-runtime-descriptor/v1":
        raise AiRuntimeDescriptorError("ai_runtime_descriptor_schema")
    return descriptor


def expected_dependency_versions() -> dict[str, str]:
    result = {}
    for requirement in load_ai_runtime_descriptor()["directRequirements"]:
        name, version = requirement.split("==", 1)
        result[name.lower()] = version
    return result
