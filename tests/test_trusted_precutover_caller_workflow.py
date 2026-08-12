from __future__ import annotations

from pathlib import Path

from scripts.workflow_yaml import load_workflow_yaml
from scripts.trusted_precutover_proof import HANDOFF_FILES


ROOT = Path(__file__).parents[1]
CALLER = ROOT / ".github/workflows/trusted-precutover-caller.yml"
PROOF_SHA = "7d1e0bfb90fb2a58d44540202ac1c4e807eb6a2d"
CANDIDATE_INPUTS = {
    f"{name}_{field}"
    for name in (
        "candidate_archive",
        "candidate_manifest",
        "candidate_attestation",
        "candidate_evidence",
    )
    for field in ("url", "sha256", "bytes")
}
MODEL_FINAL_INPUTS = {"model_pack_sha256", "model_pack_bytes"}
MODEL_PART_INPUTS = {
    f"model_pack_part_{index:02d}_{field}"
    for index in range(1, 4)
    for field in ("url", "sha256", "bytes")
}
INPUTS = CANDIDATE_INPUTS | MODEL_FINAL_INPUTS | MODEL_PART_INPUTS | {"model_pack_url"}


def test_manual_trusted_precutover_caller_has_only_closed_data_inputs():
    workflow = load_workflow_yaml(CALLER.read_text("utf-8"))
    dispatch = workflow["on"]["workflow_dispatch"]
    assert set(dispatch["inputs"]) == INPUTS
    assert len(INPUTS) == 24
    for name, descriptor in dispatch["inputs"].items():
        assert set(descriptor) == {"description", "required", "type"}
        assert descriptor["type"] == ("number" if name.endswith("_bytes") else "string")
        assert descriptor["required"] == (
            "false" if name == "model_pack_url" or name in MODEL_PART_INPUTS else "true"
        )


def test_caller_only_sha_pins_the_reusable_proof_and_forwards_all_inputs():
    source = CALLER.read_text("utf-8")
    workflow = load_workflow_yaml(source)
    assert set(workflow["jobs"]) == {"trusted_proof"}
    job = workflow["jobs"]["trusted_proof"]
    assert job["uses"] == (
        "hbhjt/vending-vision/.github/workflows/"
        f"trusted-precutover-companion-proof.yml@{PROOF_SHA}"
    )
    assert set(job["with"]) == INPUTS
    for name in INPUTS:
        assert job["with"][name] == f"${{{{ inputs.{name} }}}}"
    assert "runs-on:" not in source
    assert "steps:" not in source
    assert "secrets:" not in source
    assert "environment:" not in source
    assert "VISION_SUPPLIER_PRIVATE_KEY_PEM" not in source


def test_caller_grants_only_reusable_proof_permissions_and_produces_its_exact_three_member_artifact():
    workflow = load_workflow_yaml(CALLER.read_text("utf-8"))
    assert workflow["permissions"] == {
        "attestations": "write",
        "contents": "read",
        "id-token": "write",
    }
    proof = (
        ROOT / ".github/workflows/trusted-precutover-companion-proof.yml"
    ).read_text("utf-8")
    assert "path: proof-handoff/*" in proof
    assert "verify-evidence --directory proof-handoff" in proof
    assert HANDOFF_FILES == {
        "precutover-ai-proof.json",
        "precutover-ai-proof.sigstore.json",
        "trusted-precutover-proof-evidence.json",
    }
