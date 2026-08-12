import shutil
import subprocess
import sys
from pathlib import Path

from vision.regional_evaluator_provenance import (
    REGIONAL_EVALUATOR_DESCRIPTOR_PATH,
    REGIONAL_EVALUATOR_SOURCE_PATHS,
    build_regional_evaluator_descriptor,
    canonical_regional_evaluator_descriptor_json,
    regional_evaluator_descriptor_sha256,
    verify_regional_evaluator_provenance,
    verify_regional_evaluator_provenance_at_root,
)


ROOT = Path(__file__).parents[1]


def test_regional_evaluator_descriptor_is_canonical_exact_and_verifiable():
    raw = REGIONAL_EVALUATOR_DESCRIPTOR_PATH.read_text("utf-8")
    descriptor = build_regional_evaluator_descriptor()

    assert raw == canonical_regional_evaluator_descriptor_json(descriptor) + "\n"
    assert descriptor["schemaVersion"] == "vem-ai-regional-evaluator-descriptor/v1"
    assert descriptor["semantics"] == {
        "algorithm": "rgb-absolute-delta-rle/v1",
        "atr": "schp-atr",
        "lip": "schp-lip",
        "pose": "mediapipe-pose",
    }
    assert [source["path"] for source in descriptor["sources"]] == list(
        sorted(REGIONAL_EVALUATOR_SOURCE_PATHS)
    )
    assert "vision/ai_attempt_worker.py" not in REGIONAL_EVALUATOR_SOURCE_PATHS
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
    mutated = tmp_path / "vision" / "catvton_preprocess.py"
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
