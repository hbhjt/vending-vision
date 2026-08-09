import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.vem_vision_v2.python.vision_v2_models import parse_message
from scripts.check_vem_vision_v2_bundle import check_bundle


BUNDLE_ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2"


def test_generated_v2_boundary_accepts_shared_fast_fixtures():
    fixtures = json.loads((BUNDLE_ROOT / "fixtures" / "valid.json").read_text("utf-8"))
    assert [parse_message(fixture).type for fixture in fixtures] == [
        "vision.hello",
        "vision.ready",
        "vision.try_on.attempt.start",
        "vision.try_on.attempt.accepted",
        "vision.try_on.attempt.progress",
        "vision.try_on.attempt.completed",
    ]


def test_generated_v2_boundary_rejects_shared_negative_fixtures():
    fixtures = json.loads((BUNDLE_ROOT / "fixtures" / "invalid.json").read_text("utf-8"))
    for fixture in fixtures:
        with pytest.raises((ValidationError, ValueError)):
            parse_message(fixture["message"])


def test_vendored_bundle_has_an_independently_checkable_manifest():
    result = subprocess.run(
        [sys.executable, "scripts/check_vem_vision_v2_bundle.py"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_bundle_check_rejects_a_drifted_generated_file(tmp_path: Path):
    drifted_bundle = tmp_path / "vem_vision_v2"
    shutil.copytree(BUNDLE_ROOT, drifted_bundle)
    (drifted_bundle / "fixtures" / "valid.json").write_text("[]\n", encoding="utf-8")

    assert "digest mismatch: fixtures/valid.json" in check_bundle(drifted_bundle)
