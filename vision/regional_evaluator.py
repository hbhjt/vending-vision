"""Canonical regional-evidence evaluator used by the official AI worker.

This module deliberately carries no VEM policy digest.  Its independently
versioned descriptor can therefore bind the evidence algorithm without a
worker-policy-descriptor hash cycle.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

from vision.regional_evaluator_provenance import (
    regional_evaluator_descriptor_sha256,
    verify_regional_evaluator_provenance,
)


_MAX_INPUT_BYTES = 20 * 1024 * 1024
_MAX_INPUT_PIXELS = 8192 * 8192
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class RegionalEvaluatorError(RuntimeError):
    """Typed evaluator failure that is safe to expose at the worker boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _captured_source(raw: str | None) -> dict[str, object]:
    try:
        value = json.loads(raw or "")
    except ValueError as exc:
        raise RegionalEvaluatorError("official_catvton_captured_source_invalid") from exc
    keys = {
        "adapter", "configSha256", "decodedFrameCount", "fixtureSha256",
        "frameIndex", "relabeled", "role", "synthetic",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value.get("adapter") != "recorded_video"
        or value.get("role") != "front"
        or value.get("synthetic") is not False
        or value.get("relabeled") is not False
        or not isinstance(value.get("decodedFrameCount"), int)
        or not isinstance(value.get("frameIndex"), int)
        or not 0 <= value["frameIndex"] < value["decodedFrameCount"]
        or not _DIGEST.fullmatch(value.get("configSha256", ""))
        or not _DIGEST.fullmatch(value.get("fixtureSha256", ""))
    ):
        raise RegionalEvaluatorError("official_catvton_captured_source_invalid")
    return value


def _pack_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=False)
    if not path.is_file() or root.resolve() not in path.parents:
        raise RegionalEvaluatorError(f"official_catvton_missing:{relative}")
    return path


def _read_png(path: Path, *, role: str):
    from PIL import Image, UnidentifiedImageError

    try:
        if not path.is_file() or path.stat().st_size > _MAX_INPUT_BYTES:
            raise RegionalEvaluatorError(f"official_catvton_invalid_{role}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise RegionalEvaluatorError(f"official_catvton_invalid_{role}")
            image.load()
            if (
                image.width < 1
                or image.height < 1
                or image.width * image.height > _MAX_INPUT_PIXELS
            ):
                raise RegionalEvaluatorError(f"official_catvton_invalid_{role}")
            return image.convert("RGBA")
    except RegionalEvaluatorError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise RegionalEvaluatorError(f"official_catvton_invalid_{role}") from exc


def _person_arrays(person_path: Path):
    import numpy as np

    person = _read_png(person_path, role="person").convert("RGB")
    return person, np.array(person, dtype=np.uint8)


def _garment_condition(garment_path: Path):
    import numpy as np

    garment = _read_png(garment_path, role="garment")
    rgba = np.array(garment, dtype=np.uint8)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = np.clip(
        rgba[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)
    return rgb, rgba


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
            "pose": "mediapipe-pose-or-frame-proportional",
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


def run_regional_evaluator_attempt(
    args: argparse.Namespace, pack_root: Path, *, policy: dict[str, str]
) -> dict[str, object]:
    """Run the full regional evaluation and publish its evidence sidecar.

    Checkpoint selection, parser calls, mask construction and the resulting
    protected-region calculation intentionally live together here.  The
    worker owns only process and model-pack boundaries.
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from vision.catvton_pose_masks import CatVTONPoseError, target_hands_sleeve_masks
    from vision.catvton_preprocess import (
        build_generation_mask,
        composite_to_original,
        harmonize_garment_color,
        include_color_matched_old_edges,
        letterbox_image,
        letterbox_mask,
        parsed_old_clothes_mask,
        protected_mask_to_original,
        refine_mask_with_generated_parse,
    )
    from vision.vendor.catvton.model.SCHP import SCHP
    from vision.vendor.catvton.model.pipeline import CatVTONPipeline

    if not verify_regional_evaluator_provenance():
        raise RegionalEvaluatorError("official_catvton_regional_evaluator_provenance_mismatch")
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(16, (os.cpu_count() or 8) - 2)))
    attention_file = _pack_path(
        pack_root, "CatVTON/mix-48k-1024/attention/model.safetensors"
    )
    lip_checkpoint = _pack_path(
        pack_root, "CatVTON/SCHP/exp-schp-201908261155-lip.pth"
    )
    atr_checkpoint = _pack_path(
        pack_root, "CatVTON/SCHP/exp-schp-201908301523-atr.pth"
    )
    base_model = _pack_path(
        pack_root, "inpainting/scheduler/scheduler_config.json"
    ).parents[1]
    vae_model = _pack_path(pack_root, "vae/config.json").parent

    person_image, original = _person_arrays(Path(args.person))
    garment_source, garment_rgba = _garment_condition(Path(args.garment))
    try:
        target_mask, hands_mask, sleeve_mask = target_hands_sleeve_masks(
            original, garment_rgba, template=args.template
        )
    except CatVTONPoseError as exc:
        raise RegionalEvaluatorError(exc.code) from exc

    parsing_started = time.perf_counter()
    lip_model = SCHP(str(lip_checkpoint), device="cpu")
    with torch.inference_mode():
        lip_parse = np.array(lip_model(person_image), dtype=np.uint8)
    del lip_model
    gc.collect()
    atr_model = SCHP(str(atr_checkpoint), device="cpu")
    with torch.inference_mode():
        atr_parse = np.array(atr_model(person_image), dtype=np.uint8)
    del atr_model
    gc.collect()
    parsing_seconds = time.perf_counter() - parsing_started

    original_clothes = parsed_old_clothes_mask(lip_parse, atr_parse)
    generation_mask = build_generation_mask(target_mask, lip_parse, atr_parse, hands_mask)
    person_fit, transform = letterbox_image(original, args.width, args.height)
    mask_fit = letterbox_mask(generation_mask, transform)
    original_clothes_fit = letterbox_mask(original_clothes, transform)
    sleeve_mask_fit = letterbox_mask(sleeve_mask, transform)
    garment_fit, _ = letterbox_image(garment_source, args.width, args.height)

    load_started = time.perf_counter()
    pipeline = CatVTONPipeline(
        base_ckpt=str(base_model),
        vae_ckpt=str(vae_model),
        attn_ckpt=str(attention_file.parents[2]),
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cpu",
        compile=False,
        skip_safety_check=True,
        use_tf32=False,
        local_files_only=True,
    )
    load_seconds = time.perf_counter() - load_started
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = pipeline(
            image=Image.fromarray(person_fit),
            condition_image=Image.fromarray(garment_fit),
            mask=Image.fromarray(mask_fit),
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            height=args.height,
            width=args.width,
            generator=generator,
        )[0]
    generation_seconds = time.perf_counter() - generation_started

    generated_rgb_raw = np.array(generated.convert("RGB"), dtype=np.uint8)
    del pipeline
    gc.collect()
    post_parse_started = time.perf_counter()
    post_parse_model = SCHP(str(lip_checkpoint), device="cpu")
    with torch.inference_mode():
        generated_lip_parse = np.array(
            post_parse_model(Image.fromarray(generated_rgb_raw)), dtype=np.uint8
        )
    del post_parse_model
    gc.collect()
    final_mask_fit = refine_mask_with_generated_parse(
        mask_fit, generated_lip_parse, original_clothes_fit, sleeve_mask_fit
    )
    final_mask_fit = include_color_matched_old_edges(
        final_mask_fit, mask_fit, person_fit, original_clothes_fit
    )
    post_parse_seconds = time.perf_counter() - post_parse_started
    generated_rgb = harmonize_garment_color(
        generated_rgb_raw, final_mask_fit, garment_fit
    )
    result = composite_to_original(original, generated_rgb, final_mask_fit, transform)
    upper_body, protected = protected_mask_to_original(mask_fit, final_mask_fit, transform)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    Image.fromarray(result).save(temp_output, format="PNG", compress_level=3)
    with Image.open(temp_output) as check:
        check.load()
        if (
            check.format != "PNG"
            or check.width != original.shape[1]
            or check.height != original.shape[0]
        ):
            raise RegionalEvaluatorError("official_catvton_invalid_output")
    os.replace(temp_output, output)
    if args.regional_evidence_output:
        write_regional_evidence(
            output_path=output,
            evidence_path=Path(args.regional_evidence_output),
            person_path=Path(args.person),
            garment_path=Path(args.garment),
            captured_source=_captured_source(args.captured_source),
            original=original,
            result=result,
            upper_body=upper_body,
            protected=protected,
            evaluator_source_descriptor_sha256=regional_evaluator_descriptor_sha256(),
            policy=policy,
        )
    return {
        "output": str(output),
        "parsing_seconds": round(parsing_seconds, 2),
        "post_parse_seconds": round(post_parse_seconds, 2),
        "load_seconds": round(load_seconds, 2),
        "generation_seconds": round(generation_seconds, 2),
        "worker_seconds": round(time.perf_counter() - started, 2),
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
    }
