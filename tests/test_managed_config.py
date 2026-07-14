import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def import_config(config_path):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "VISION_CONFIG_FILE": str(config_path),
            "VISION_CONFIG_MODE": "managed",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", "from vision.config import settings; print(settings.HOST)"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def valid_config():
    return {
        "schemaVersion": "vending-vision-site-config/v1",
        "host": "127.0.0.1",
        "port": 7892,
        "cameras": {
            "top": {"role": "presence", "rotate": 0},
            "front": {"role": "profile_tryon", "rotate": 270},
        },
        "allowed_origins": ["http://tauri.localhost"],
    }


def test_managed_config_is_required_and_validated(tmp_path):
    missing = import_config(tmp_path / "missing.json")
    assert missing.returncode != 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert import_config(malformed).returncode != 0

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({**valid_config(), "unexpected": True}), encoding="utf-8")
    assert import_config(unknown).returncode != 0

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(valid_config()), encoding="utf-8")
    result = import_config(valid)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "127.0.0.1"


def test_managed_config_cannot_enable_mock(tmp_path):
    config = tmp_path / "mock.json"
    config.write_text(json.dumps({**valid_config(), "mock_scenario": "success"}), encoding="utf-8")
    assert import_config(config).returncode != 0


def test_managed_site_config_cannot_persist_camera_indexes(tmp_path):
    config = tmp_path / "legacy-index.json"
    value = valid_config()
    value["cameras"]["top"]["index"] = 4
    config.write_text(json.dumps(value), encoding="utf-8")

    assert import_config(config).returncode != 0
