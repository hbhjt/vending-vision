"""Repository-level Windows checkout invariants for byte-pinned assets."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]


def test_canonical_text_is_forced_to_lf_in_windows_checkouts():
    """Byte-pinned contracts and descriptors must survive ``core.autocrlf=true``."""
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            "contracts/vem_vision_v2/manifest.json",
            "ai-runtime-descriptor.json",
            "regional-evaluator-descriptor.json",
            "trusted-signer-scripts.json",
            ".github/workflows/trusted-ai-candidate-signer.yml",
            "requirements.txt",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert all(line.endswith(": text: auto") or line.endswith(": eol: lf") for line in result.stdout.splitlines())
