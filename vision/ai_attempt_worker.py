"""Official CatVTON attempt-child entrypoint.

It is a child-only module.  Network and Hugging Face download APIs are denied
before imports; the installed immutable pack is the only accepted source.
Issue 10/11 run the full official inference with the staged pack on Windows.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from vision.ai_model_pack import (
    AiModelPackError,
    OFFICIAL_CATVTON_SOURCE_REVISION,
    verify_ai_model_pack,
)
from vision.ai_runtime_descriptor import dependency_version_satisfies, expected_dependency_requirements
from vision.regional_evaluator import RegionalEvaluatorError, run_regional_evaluator_attempt
from vision.regional_evaluator_provenance import verify_regional_evaluator_provenance
from vision.source_provenance import verify_official_source_provenance

_REGIONAL_EVIDENCE_POLICY = {
    "schemaVersion": "vem-ai-regional-evidence-policy/v1",
    "sha256": "141dc7bd9d8b03dc54dd5ec343e5090b9c673366d37c6250cf6135f717161654",
}

def _deny_downloads() -> None:
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    socket.socket = _blocked_socket  # type: ignore[assignment]


def _blocked_socket(*_args, **_kwargs):
    raise RuntimeError("customer_ai_attempt_network_forbidden")


class CatVTONWorkerError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _pack_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=False)
    if not path.is_file() or root.resolve() not in path.parents:
        raise CatVTONWorkerError(f"official_catvton_missing:{relative}")
    return path


def _probe_runtime_worker() -> dict[str, object]:
    import importlib.metadata

    import accelerate  # noqa: F401
    import cv2  # noqa: F401
    import diffusers  # noqa: F401
    import numpy  # noqa: F401
    import safetensors  # noqa: F401
    import scipy  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import tqdm  # noqa: F401
    import transformers  # noqa: F401
    from PIL import Image  # noqa: F401

    from vision.vendor.catvton.model.SCHP import SCHP  # noqa: F401
    from vision.vendor.catvton.model.pipeline import CatVTONPipeline  # noqa: F401

    provenance = Path(__file__).resolve().parent / "vendor" / "catvton" / "PROVENANCE.md"
    if OFFICIAL_CATVTON_SOURCE_REVISION not in provenance.read_text("utf-8"):
        raise CatVTONWorkerError("official_catvton_source_revision_mismatch")
    if not verify_official_source_provenance():
        raise CatVTONWorkerError("official_catvton_source_provenance_mismatch")
    if not verify_regional_evaluator_provenance():
        raise CatVTONWorkerError("official_catvton_regional_evaluator_provenance_mismatch")

    payload: dict[str, object] = {
        "probe": "official-catvton-worker-runtime",
        "catvtonSourceRevision": OFFICIAL_CATVTON_SOURCE_REVISION,
    }
    for name, requirement in expected_dependency_requirements().items():
        actual = importlib.metadata.version(name)
        if not dependency_version_satisfies(requirement, actual):
            raise CatVTONWorkerError(f"official_catvton_dependency_version:{name}")
        payload[name] = actual
    return payload


def _probe_official_worker(pack_root: Path) -> dict[str, object]:
    payload = _probe_runtime_worker()
    for relative in (
        "inpainting/scheduler/scheduler_config.json",
        "inpainting/unet/config.json",
        "vae/config.json",
    ):
        json.loads(_pack_path(pack_root, relative).read_text("utf-8"))
    payload["probe"] = "official-catvton-worker"
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pack")
    parser.add_argument("--probe-runtime", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--person")
    parser.add_argument("--garment")
    parser.add_argument("--template", choices=["tshirt_short_sleeve", "tshirt_long_sleeve"], default="tshirt_short_sleeve")
    parser.add_argument("--output")
    parser.add_argument("--regional-evidence-output")
    parser.add_argument("--captured-source")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    _deny_downloads()
    if args.probe_runtime:
        try:
            print(json.dumps(_probe_runtime_worker(), ensure_ascii=False, sort_keys=True))
            return 0
        except ModuleNotFoundError as exc:
            print(f"official_catvton_dependency_missing:{exc.name}", file=sys.stderr)
            return 2
        except CatVTONWorkerError as exc:
            print(exc.code, file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"official_catvton_runtime_probe_failed:{type(exc).__name__}", file=sys.stderr)
            return 2
    if not args.model_pack:
        raise RuntimeError("official_catvton_model_pack_required")
    pack_root = Path(args.model_pack)
    try:
        verify_ai_model_pack(pack_root)
    except AiModelPackError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.probe:
        try:
            probe = _probe_official_worker(pack_root)
        except ModuleNotFoundError as exc:
            print(f"official_catvton_dependency_missing:{exc.name}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"official_catvton_probe_failed:{type(exc).__name__}", file=sys.stderr)
            return 2
        print(json.dumps(probe, ensure_ascii=False, sort_keys=True))
        return 0
    if not (args.person and args.garment and args.output):
        raise RuntimeError("official_catvton_attempt_paths_required")
    if bool(args.regional_evidence_output) != bool(args.captured_source):
        raise RuntimeError("official_catvton_regional_evidence_paths_required")
    try:
        metrics = run_regional_evaluator_attempt(
            args, pack_root, policy=_REGIONAL_EVIDENCE_POLICY
        )
    except ModuleNotFoundError as exc:
        print(f"official_catvton_dependency_missing:{exc.name}", file=sys.stderr)
        return 2
    except (CatVTONWorkerError, RegionalEvaluatorError) as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"official_catvton_worker_failed:{type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
