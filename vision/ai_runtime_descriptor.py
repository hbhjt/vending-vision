from __future__ import annotations

import hashlib
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

RUNTIME_DESCRIPTOR_PATH = Path(__file__).resolve().parents[1] / "ai-runtime-descriptor.json"


class AiRuntimeDescriptorError(RuntimeError):
    pass


def canonical_json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_runtime_descriptor(path: Path = RUNTIME_DESCRIPTOR_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalise_requirement(value: str) -> str:
    requirement = Requirement(value)
    if requirement.url or requirement.extras or requirement.marker or not requirement.specifier:
        raise AiRuntimeDescriptorError("ai_runtime_requirement")
    return f"{canonicalize_name(requirement.name)}{requirement.specifier}"


def _read_direct_requirements(path: Path) -> list[str]:
    return [
        _normalise_requirement(line.strip())
        for line in path.read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_ai_runtime_descriptor(path: Path = RUNTIME_DESCRIPTOR_PATH) -> dict:
    raw = path.read_text("utf-8")
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
    requirements_path = path.with_name("requirements-ai.txt")
    lock_path = path.with_name("requirements-ai.lock.json")
    if hashlib.sha256(requirements_path.read_bytes()).hexdigest() != descriptor["requirementsAiSha256"]:
        raise AiRuntimeDescriptorError("ai_runtime_requirements_digest")
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != descriptor["requirementsAiLockSha256"]:
        raise AiRuntimeDescriptorError("ai_runtime_lock_digest")
    lock_raw = lock_path.read_text("utf-8")
    lock = json.loads(lock_raw)
    if canonical_json(lock) != lock_raw.rstrip("\n"):
        raise AiRuntimeDescriptorError("ai_runtime_lock_noncanonical")
    expected = sorted(_normalise_requirement(value) for value in descriptor["directRequirements"])
    if sorted(_read_direct_requirements(requirements_path)) != expected:
        raise AiRuntimeDescriptorError("ai_runtime_requirements_semantics")
    if sorted(_normalise_requirement(value) for value in lock.get("directRequirements", [])) != expected:
        raise AiRuntimeDescriptorError("ai_runtime_lock_semantics")
    return descriptor


def expected_dependency_versions() -> dict[str, str]:
    result = {}
    for requirement in load_ai_runtime_descriptor()["directRequirements"]:
        name, version = requirement.split("==", 1)
        result[name.lower()] = version
    return result


def expected_dependency_requirements() -> dict[str, str]:
    result = {}
    for value in load_ai_runtime_descriptor()["directRequirements"]:
        requirement = Requirement(value)
        result[canonicalize_name(requirement.name)] = value
    return result


def dependency_version_satisfies(requirement_text: str, actual_version: object) -> bool:
    if not isinstance(actual_version, str):
        return False
    requirement = Requirement(requirement_text)
    try:
        return requirement.specifier.contains(Version(actual_version), prereleases=True)
    except InvalidVersion:
        return False
