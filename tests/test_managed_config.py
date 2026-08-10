import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import jsonschema


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


def import_config_values(config_path):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "VISION_CONFIG_FILE": str(config_path),
            "VISION_CONFIG_MODE": "managed",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from vision.config import settings; "
            "print(f'{settings.HOST}:{settings.PORT}:{settings.TOP_CAMERA_CONFIG[\"role\"]}:{settings.FRONT_CAMERA_CONFIG[\"role\"]}')",
        ],
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
        "front": {"role": "profile_fast_try_on", "rotate": 270},
        },
        "allowed_origins": ["http://tauri.localhost"],
    }


def test_frozen_site_schema_and_example_are_real_json_sources():
    schema = json.loads(
        (ROOT / "config" / "vending-vision-site-config-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (ROOT / "config" / "site.example.json").read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema).validate(example)


def test_managed_config_values_come_from_the_frozen_site_source(tmp_path):
    config = tmp_path / "site.json"
    value = valid_config()
    value["port"] = 17892
    config.write_text(json.dumps(value), encoding="utf-8")

    result = import_config_values(config)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "127.0.0.1:17892:presence:profile_fast_try_on"


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


def test_managed_config_accepts_recorded_video_camera_source(tmp_path):
    config = tmp_path / "recorded-video.json"
    value = valid_config()
    value["cameras"]["top"].update({
        "source": "recorded_video",
        "video_path": r"C:\\fixtures\\top.mp4",
        "loop": False,
    })
    value["cameras"]["front"].update({
        "source": "recorded_video",
        "video_path": r"C:\\fixtures\\front.mp4",
        "loop": True,
    })
    config.write_text(json.dumps(value), encoding="utf-8")

    result = import_config(config)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["cameras"]["top"].update({"source": "recorded_video"}),
        lambda value: value["cameras"]["top"].update({"source": "recorded_video", "video_path": "top.mp4"}) or value["cameras"]["front"].update({"source": "dshow"}),
    ],
)
def test_managed_config_rejects_invalid_recorded_video_source_combinations(tmp_path, mutate):
    config = tmp_path / "invalid-recorded-video.json"
    value = valid_config()
    mutate(value)
    config.write_text(json.dumps(value), encoding="utf-8")

    result = import_config(config)

    assert result.returncode != 0


def test_managed_site_config_cannot_persist_camera_indexes(tmp_path):
    config = tmp_path / "legacy-index.json"
    value = valid_config()
    value["cameras"]["top"]["index"] = 4
    config.write_text(json.dumps(value), encoding="utf-8")

    assert import_config(config).returncode != 0


def test_managed_config_rejects_obsolete_maintenance_authorization_material(tmp_path):
    config = tmp_path / "maintenance-material.json"
    value = valid_config()
    value.update({
        "maintenance_capability_keyring_path": r"C:\\ProgramData\\VEM\\vision\\daemon-maintenance-keys.json",
        "maintenance_session_path": r"C:\\ProgramData\\VEM\\vision\\daemon-maintenance-session.json",
        "maintenance_replay_path": r"C:\\ProgramData\\VEM\\vision\\camera-maintenance-replay.json",
    })
    config.write_text(json.dumps(value), encoding="utf-8")

    result = import_config(config)
    assert result.returncode != 0
