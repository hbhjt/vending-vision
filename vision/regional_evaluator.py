"""Canonical regional-evidence evaluator used by the official AI worker.

This module deliberately carries no VEM policy digest.  Its independently
versioned descriptor can therefore bind the evidence algorithm without a
worker-policy-descriptor hash cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class RegionalEvaluatorError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _rle(mask) -> dict[str, object]:
    flat = (mask.reshape(-1) > 0).tolist()
    runs: list[list[int]] = []
    start = None
    for index, selected in enumerate([*flat, False]):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            runs.append([start, index - start])
            start = None
    if not runs:
        raise RegionalEvaluatorError("official_catvton_regional_mask_empty")
    return {"encoding": "rle-row-major/v1", "runs": runs}


def _measurement(original, result, mask) -> dict[str, int]:
    import numpy as np

    selected = mask > 0
    sampled = int(np.count_nonzero(selected))
    if sampled == 0:
        raise RegionalEvaluatorError("official_catvton_regional_mask_empty")
    delta = np.abs(result.astype(np.int16) - original.astype(np.int16))
    changed = int(np.count_nonzero(np.any(delta > 0, axis=2) & selected))
    return {
        "changedFractionBps": changed * 10_000 // sampled,
        "changedPixels": changed,
        "meanDelta": int(delta[selected].sum()) // (sampled * 3),
        "sampledPixels": sampled,
    }


def write_regional_evidence(
    *,
    output_path: Path,
    evidence_path: Path,
    person_path: Path,
    garment_path: Path,
    captured_source: dict[str, object],
    original,
    result,
    upper_body,
    protected,
    evaluator_source_descriptor_sha256: str,
    policy: dict[str, str],
) -> None:
    """Atomically write one canonical sidecar for the supplied evaluated output."""
    if (
        evidence_path.parent.resolve() != output_path.parent.resolve()
        or evidence_path == output_path
    ):
        raise RegionalEvaluatorError("official_catvton_regional_output_path_invalid")
    upper_measurement = _measurement(original, result, upper_body)
    protected_measurement = _measurement(original, result, protected)
    upper_measurement["verdict"] = (
        "changed" if upper_measurement["changedPixels"] > 0 else "insufficient_change"
    )
    protected_measurement["verdict"] = (
        "preserved" if protected_measurement["changedPixels"] == 0 else "changed"
    )
    verdict = (
        "passed"
        if upper_measurement["verdict"] == "changed"
        and protected_measurement["verdict"] == "preserved"
        else "regional_check_failed"
    )
    sidecar = {
        "attempt": {
            "acquisitionSource": "direct_recorded_frame",
            "decodedHeight": int(original.shape[0]),
            "decodedWidth": int(original.shape[1]),
            "garmentSha256": _sha256(garment_path),
            "inputSha256": _sha256(person_path),
            "recordedFixtureSha256": captured_source["fixtureSha256"],
            "resultSha256": _sha256(output_path),
            "sourceCamera": "front",
        },
        "evaluator": {
            "algorithm": "rgb-absolute-delta-rle/v1",
            "atr": "schp-atr",
            "lip": "schp-lip",
            "pose": "mediapipe-pose",
            "sourceDescriptorSha256": evaluator_source_descriptor_sha256,
        },
        "kind": "regional-evidence",
        "masks": {
            "height": int(original.shape[0]),
            "protectedRegion": _rle(protected),
            "upperBody": _rle(upper_body),
            "width": int(original.shape[1]),
        },
        "measurements": {
            "protectedRegion": protected_measurement,
            "upperBody": upper_measurement,
        },
        "policy": policy,
        "schemaVersion": "vem-ai-regional-evidence/v1",
        "verdict": verdict,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    temp = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(_canonical_json(sidecar))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, evidence_path)
    finally:
        temp.unlink(missing_ok=True)
