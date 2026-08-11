"""Official CatVTON attempt-child entrypoint.

It is a child-only module.  Network and Hugging Face download APIs are denied
before imports; the installed immutable pack is the only accepted source.
Issue 10/11 run the full official inference with the staged pack on Windows.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys

from vision.ai_model_pack import (
    OFFICIAL_CATVTON_REPOSITORY,
    OFFICIAL_CATVTON_REVISION,
    verify_ai_model_pack,
)


def _deny_downloads() -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    socket.socket = _blocked_socket  # type: ignore[assignment]


def _blocked_socket(*_args, **_kwargs):
    raise RuntimeError("customer_ai_attempt_network_forbidden")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pack", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--person")
    parser.add_argument("--garment")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    _deny_downloads()
    pack = verify_ai_model_pack(args.model_pack)
    if (
        pack.upstream_repository != OFFICIAL_CATVTON_REPOSITORY
        or pack.upstream_revision != OFFICIAL_CATVTON_REVISION
    ):
        raise RuntimeError("official_catvton_identity_required")
    # This deliberately stops at imports in ordinary startup/probe.  No model
    # is loaded by Vision startup and only the verified official worker can
    # claim official readiness.
    if args.probe:
        print("official-catvton-worker-configured")
        return 0
    if not (args.person and args.garment and args.output):
        raise RuntimeError("official_catvton_attempt_paths_required")
    raise RuntimeError("official_catvton_inference_requires_installed_vm_pack")


if __name__ == "__main__":
    raise SystemExit(main())
