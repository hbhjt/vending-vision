from vision.model_manifest import REQUIRED_ROLES, verify_model_manifest


def test_all_declared_production_models_are_present_and_hash_verified():
    result = verify_model_manifest()
    assert result["ok"], result
    assert {item["role"] for item in result["models"]} == REQUIRED_ROLES
    assert all(item["ok"] for item in result["models"])
