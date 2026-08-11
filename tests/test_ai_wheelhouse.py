import json

import pytest

from scripts.verify_ai_wheelhouse import (
    WheelhouseError,
    build_ai_wheelhouse_descriptor,
    canonical_json,
    verify_ai_wheelhouse,
)


def test_wheelhouse_descriptor_builder_records_exact_wheels_and_rejects_extra(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    first = wheelhouse / "torch-2.8.0-cp311-cp311-win_amd64.whl"
    second = wheelhouse / "diffusers-0.29.2-py3-none-any.whl"
    first.write_bytes(b"torch-wheel")
    second.write_bytes(b"diffusers-wheel")
    descriptor = build_ai_wheelhouse_descriptor(
        wheelhouse,
        requirements=["torch==2.8.0", "diffusers==0.29.2"],
    )
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(canonical_json(descriptor), "utf-8")

    verify_ai_wheelhouse(descriptor_path, wheelhouse)

    (wheelhouse / "extra-1.0.0-py3-none-any.whl").write_bytes(b"extra")
    with pytest.raises(WheelhouseError, match="ai_wheelhouse_extra_or_missing"):
        verify_ai_wheelhouse(descriptor_path, wheelhouse)


def test_empty_tracked_ai_wheelhouse_descriptor_fails_closed():
    descriptor = json.loads(open("requirements-ai.lock.json", encoding="utf-8").read())

    assert descriptor["wheels"] == []
