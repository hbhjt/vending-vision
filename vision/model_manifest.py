from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vision.config import runtime_path


MANIFEST_SCHEMA = "vending-vision-model-manifest/v1"
REQUIRED_ROLES = {
    "person_detection",
    "face_detection",
    "age_network_definition",
    "age_network_weights",
    "gender_network_definition",
    "gender_network_weights",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_manifest(runtime_root: Path | None = None) -> dict:
    """Verify declared production models under the configured or explicit runtime root."""
    manifest_path = (
        Path(runtime_path("models/model-manifest.json"))
        if runtime_root is None
        else Path(runtime_root) / "models" / "model-manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "message": f"model manifest unavailable: {exc}", "models": []}

    models = manifest.get("models")
    if manifest.get("schemaVersion") != MANIFEST_SCHEMA or not isinstance(models, list):
        return {"ok": False, "message": "model manifest contract is invalid", "models": []}
    roles = {item.get("role") for item in models if isinstance(item, dict)}
    if roles != REQUIRED_ROLES:
        return {"ok": False, "message": "model manifest roles are incomplete", "models": []}

    results = []
    for item in models:
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            return {"ok": False, "message": "model manifest entry is invalid", "models": results}
        model_path = (
            Path(runtime_path(relative))
            if runtime_root is None
            else Path(runtime_root) / relative
        )
        actual = sha256_file(model_path) if model_path.is_file() else None
        ok = actual == expected
        results.append({"role": item["role"], "path": relative, "ok": ok})
        if not ok:
            return {
                "ok": False,
                "message": f"required model digest mismatch: {relative}",
                "models": results,
            }
    return {"ok": True, "message": "all declared production models verified", "models": results}
