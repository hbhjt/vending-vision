import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vision.regional_evaluator_provenance import (
    REGIONAL_EVALUATOR_DESCRIPTOR_PATH,
    REGIONAL_EVALUATOR_SOURCE_PATHS,
    build_regional_evaluator_descriptor,
    canonical_regional_evaluator_descriptor_json,
    regional_evaluator_descriptor_sha256,
    verify_regional_evaluator_provenance,
    verify_regional_evaluator_provenance_at_root,
)
from vision.regional_evaluator import RegionalEvaluatorError


ROOT = Path(__file__).parents[1]


def test_regional_evaluator_error_exposes_its_stable_worker_code():
    error = RegionalEvaluatorError("official_catvton_invalid_garment")

    assert error.code == "official_catvton_invalid_garment"
    assert str(error) == error.code


def test_regional_evaluator_descriptor_is_canonical_exact_and_verifiable():
    raw = REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8")
    descriptor = build_regional_evaluator_descriptor()

    assert raw == canonical_regional_evaluator_descriptor_json(descriptor) + "\n"
    assert descriptor["schemaVersion"] == "vem-ai-regional-evaluator-descriptor/v1"
    assert descriptor["semantics"] == {
        "algorithm": "rgb-absolute-delta-rle/v1",
        "atr": "schp-atr",
        "lip": "schp-lip",
        "pose": "mediapipe-pose-or-frame-proportional",
    }
    assert [source["path"] for source in descriptor["sources"]] == list(
        sorted(REGIONAL_EVALUATOR_SOURCE_PATHS)
    )
    assert "vision/ai_attempt_worker.py" not in REGIONAL_EVALUATOR_SOURCE_PATHS
    assert {
        "vision/config.py",
        "vision/regional_evaluator_provenance.py",
        "vision/vendor/catvton/model/attn_processor.py",
        "vision/vendor/catvton/model/pipeline.py",
        "vision/vendor/catvton/model/utils.py",
        "vision/vendor/catvton/utils.py",
    } <= set(REGIONAL_EVALUATOR_SOURCE_PATHS)
    assert all(source["byteSize"] > 0 for source in descriptor["sources"])
    assert all(len(source["sha256"]) == 64 for source in descriptor["sources"])
    assert regional_evaluator_descriptor_sha256() == __import__("hashlib").sha256(
        raw.encode("utf-8")
    ).hexdigest()
    assert verify_regional_evaluator_provenance() is True


def test_regional_evaluator_descriptor_rejects_source_mutation_and_script_check(tmp_path):
    descriptor = build_regional_evaluator_descriptor()
    for source in descriptor["sources"]:
        source_path = Path(source["path"])
        destination = tmp_path / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source_path, destination)
    (tmp_path / "regional-evaluator-descriptor.json").write_text(
        REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8"), "utf-8"
    )

    assert verify_regional_evaluator_provenance_at_root(tmp_path) is True
    mutated = tmp_path / "vision" / "config.py"
    mutated.write_text(mutated.read_text("utf-8") + "\n# mutation\n", "utf-8")
    assert verify_regional_evaluator_provenance_at_root(tmp_path) is False

    checked = subprocess.run(
        [sys.executable, "scripts/regional_evaluator_descriptor.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr


def test_release_version_materialization_preserves_regional_evaluator_provenance(
    tmp_path,
):
    descriptor = build_regional_evaluator_descriptor()
    for source in descriptor["sources"]:
        source_path = Path(source["path"])
        destination = tmp_path / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source_path, destination)
    build_version = tmp_path / "vision" / "_build_version.py"
    shutil.copyfile(ROOT / "vision" / "_build_version.py", build_version)
    (tmp_path / "regional-evaluator-descriptor.json").write_text(
        REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8"), "utf-8"
    )

    materialized = subprocess.run(
        [
            sys.executable,
            ROOT / "scripts" / "set_release_version.py",
            "0.2.1-rc.12",
            "--root",
            tmp_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert materialized.returncode == 0, materialized.stderr
    assert build_version.read_text("utf-8") == 'APP_VERSION = "0.2.1-rc.12"\n'
    assert verify_regional_evaluator_provenance_at_root(tmp_path) is True


def test_regional_evaluator_descriptor_rejects_an_unlisted_local_import(tmp_path):
    descriptor = build_regional_evaluator_descriptor()
    for source in descriptor["sources"]:
        source_path = Path(source["path"])
        destination = tmp_path / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source_path, destination)
    (tmp_path / "regional-evaluator-descriptor.json").write_text(
        REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8"), "utf-8"
    )
    evaluator = tmp_path / "vision" / "regional_evaluator.py"
    evaluator.write_text(
        "from vision.ai_model_pack import verify_ai_model_pack\n"
        + evaluator.read_text("utf-8"),
        "utf-8",
    )
    (tmp_path / "regional-evaluator-descriptor.json").write_text(
        canonical_regional_evaluator_descriptor_json(
            build_regional_evaluator_descriptor(tmp_path)
        )
        + "\n",
        "utf-8",
    )

    assert verify_regional_evaluator_provenance_at_root(tmp_path) is False


@pytest.mark.parametrize(
    "replacement",
    (
        "vision/catvton_pose_masks.py",
        "vision/Config.py",
        "vision/vendor/../config.py",
    ),
)
def test_regional_evaluator_descriptor_rejects_duplicate_case_or_path_variant(tmp_path, replacement):
    descriptor = build_regional_evaluator_descriptor()
    for source in descriptor["sources"]:
        source_path = Path(source["path"])
        destination = tmp_path / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source_path, destination)
    descriptor["sources"][-1]["path"] = replacement
    (tmp_path / "regional-evaluator-descriptor.json").write_text(
        canonical_regional_evaluator_descriptor_json(descriptor) + "\n", "utf-8"
    )

    assert verify_regional_evaluator_provenance_at_root(tmp_path) is False
