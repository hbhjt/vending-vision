"""V2 contract boundary loaded only from the vendored generated bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"


class V2ContractBundleUnavailable(RuntimeError):
    """The optional Fast contract is absent; core Vision must keep running."""


@dataclass(frozen=True)
class V2ContractIdentity:
    schema_version: str
    bundle_version: str
    contract_digest: str


def load_v2_contract_identity() -> V2ContractIdentity:
    """Read the generated manifest instead of maintaining a second digest."""
    try:
        manifest = json.loads((_BUNDLE_ROOT / "manifest.json").read_text("utf-8"))
        schema_version = manifest["schemaVersion"]
        bundle_version = manifest["bundleVersion"]
        contract_digest = manifest["bundleDigest"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise V2ContractBundleUnavailable("vision_v2_contract_bundle_unavailable") from error
    if not all(
        isinstance(value, str) and value
        for value in (schema_version, bundle_version, contract_digest)
    ):
        raise V2ContractBundleUnavailable("vision_v2_contract_bundle_unavailable")
    return V2ContractIdentity(schema_version, bundle_version, contract_digest)


def parse_v2_boundary_message(value: Any):
    """Parse one handshake message with the generated Python boundary model."""
    try:
        from contracts.vem_vision_v2.python.vision_v2_models import parse_message
    except (ImportError, OSError) as error:
        raise V2ContractBundleUnavailable("vision_v2_contract_bundle_unavailable") from error
    try:
        return parse_message(value)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid_v2_boundary_message") from error
