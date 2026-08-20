import json
from pathlib import Path

import pytest

from contracts.vem_vision_v2.python.vision_v2_models import (
    parse_client_message,
    parse_server_message,
)
from vision.v2_contract_bundle import (
    parse_v2_client_message,
    parse_v2_server_message,
)


ROOT = Path(__file__).parents[1] / "contracts" / "vem_vision_v2" / "fixtures"
BUNDLE_ROOT = ROOT.parent


@pytest.mark.parametrize(
    ("direction", "parser"),
    [("client", parse_client_message), ("server", parse_server_message)],
)
def test_generated_direction_parser_accepts_only_its_explicit_valid_corpus(direction, parser):
    fixtures = json.loads((ROOT / f"{direction}-valid.json").read_text("utf-8"))
    assert [parser(fixture).type for fixture in fixtures]
    opposite = parse_server_message if direction == "client" else parse_client_message
    for fixture in fixtures:
        with pytest.raises(ValueError):
            opposite(fixture)


@pytest.mark.parametrize(
    ("direction", "parser"),
    [("client", parse_client_message), ("server", parse_server_message)],
)
def test_every_negative_mutates_a_valid_message_and_is_rejected_in_its_direction(direction, parser):
    fixtures = json.loads((ROOT / f"{direction}-invalid.json").read_text("utf-8"))
    for fixture in fixtures:
        assert parser(fixture["base"])
        with pytest.raises(ValueError, match="invalid vem.vision.v2 message"):
            parser(fixture["message"])


def test_runtime_boundary_has_no_generic_message_parser():
    hello = json.loads((ROOT / "client-valid.json").read_text("utf-8"))[0]
    ready = json.loads((ROOT / "server-valid.json").read_text("utf-8"))[0]
    assert parse_v2_client_message(hello).type == "vision.hello"
    assert parse_v2_server_message(ready).type == "vision.ready"


def test_try_on_start_contract_has_no_mode_dimension():
    schema = json.loads(
        (BUNDLE_ROOT / "vision-v2.client.schema.json").read_text("utf-8")
    )
    start = next(
        option
        for option in schema["oneOf"]
        if option["properties"]["type"].get("const")
        == "vision.try_on.attempt.start"
    )
    payload_schema = start["properties"]["payload"]
    assert "mode" not in payload_schema["properties"]
    assert "mode" not in payload_schema["required"]

    valid = json.loads((ROOT / "client-valid.json").read_text("utf-8"))
    starts = [item for item in valid if item["type"] == "vision.try_on.attempt.start"]
    assert starts
    assert all("mode" not in item["payload"] for item in starts)
