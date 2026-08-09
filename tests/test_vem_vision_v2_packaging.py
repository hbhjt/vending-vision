from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_frozen_spec_keeps_the_generated_v2_bundle_and_static_boundary_import():
    spec = (ROOT / "vending_vision.spec").read_text("utf-8")

    assert '(str(ROOT / "contracts"), "contracts")' in spec
    assert '"vision.v2_contract_bundle"' in spec
    assert '"contracts.vem_vision_v2.python.vision_v2_models"' in spec


def test_packaged_verifier_executes_the_frozen_bundle_positive_negative_probe():
    verifier = (ROOT / "scripts" / "verify_packaged_exe.py").read_text("utf-8")

    assert '"--verify-v2-contract-bundle"' in verifier
    assert '"V2 contract bundle probe passed"' in verifier
    assert '"contracts" / "vem_vision_v2" / "manifest.json"' in verifier
