import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
THIS_TEST = Path(__file__).resolve()

FORBIDDEN_PATTERNS = (
    ("protocol-v1", re.compile(r"\bvem[.]vision[.]v1\b")),
    (
        "legacy-try-on-wire-message",
        re.compile(r"\bvision[.]try_on[.](?:start|stop|started|stopped)\b"),
    ),
    ("legacy-try-on-client", re.compile(r"\btryon_frontend\b")),
    (
        "legacy-try-on-session-module",
        re.compile(r"\b(?:VisionTryOnSession|try_on_session|tryOnSession)\b"),
    ),
    (
        "legacy-preview-route",
        re.compile(r"/try-on/\{[^}]+}[.]mjpeg\b|try-on-preview", re.I),
    ),
    (
        "legacy-nested-customer-route",
        re.compile(
            r"#/products/[^\s\"'`]+/try-on\b"
            r"|path\s*:\s*[\"']/products/:[^\"']+/try-on\b"
        ),
    ),
    (
        "legacy-try-on-selector",
        re.compile(r"(?:data-test\s*=\s*[\"']|\[data-test=[\"'])try-on-exit\b"),
    ),
    (
        "fabricated-try-on-phase-evidence",
        re.compile(r"\b(?:accepted|progress|completed)Observed\b"),
    ),
    (
        "legacy-fast-try-on-owner",
        re.compile(r"(?<!profile_)fast_try_on\b"),
    ),
    (
        "standalone-repository-url",
        re.compile(r"https?://github[.]com/hbhjt/virtual-tryon(?:[.]git)?\b", re.I),
    ),
    (
        "standalone-repository-path",
        re.compile(
            r"(?:[.][.][\\/]|[A-Za-z]:[\\/]|/workspaces/)?virtual-tryon[\\/]"
            r"(?:run[.]ps1|app[\\/]|static[\\/]|scripts[\\/]|requirements[.]txt|vendor[\\/])",
            re.I,
        ),
    ),
    (
        "standalone-server-entrypoint",
        re.compile(
            r"\bapp[.]main:app\b|\bfrom\s+app[.]main\s+import\s+app\b|\bimport\s+app[.]main\b"
        ),
    ),
    (
        "standalone-browser-camera-owner",
        re.compile(r"\bnavigator[.]mediaDevices\b|\bgetUserMedia\s*[(]"),
    ),
)

STANDALONE_PROVENANCE_ALLOWANCES = {
    (ROOT / "vision/vendor/catvton/PROVENANCE.md").resolve(): {
        "sha256": "9afdf1fe17cdc7b8287ee008488bd902a1de27b7f0121c6d03913cd04f73bd73",
        "occurrences": {"standalone-repository-url": 1},
    },
    (ROOT / "fixtures/recorded-video/README.md").resolve(): {
        "sha256": "414c953a0f797284201636a04f0f726ab0ad5cabebcedde0874c593d28990a6d",
        "occurrences": {"standalone-repository-url": 1},
    },
}


def find_violations(paths):
    source_files = [
        path
        for root in paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file()
        and path.suffix not in {".pyc", ".mp4", ".png", ".jpg", ".jpeg", ".log"}
        and path != THIS_TEST
    ]
    violations = []
    for path in source_files:
        source = path.read_text(encoding="utf-8", errors="ignore")
        allowance = STANDALONE_PROVENANCE_ALLOWANCES.get(path.resolve())
        allowance_valid = allowance is not None and hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest() == allowance["sha256"]
        if allowance is not None and not allowance_valid:
            violations.append(f"{path}: standalone-provenance-integrity")
        for category, pattern in FORBIDDEN_PATTERNS:
            matches = pattern.findall(source)
            if not matches:
                continue
            permitted = allowance["occurrences"].get(category) if allowance_valid else None
            if permitted == len(matches):
                continue
            violations.append(f"{path}: {category}")
    return violations


def test_retired_vision_session_and_v1_surface_is_absent():
    paths = [
        ROOT / ".github" / "workflows",
        ROOT / "app.py",
        ROOT / "vision",
        ROOT / "config",
        ROOT / "contracts",
        ROOT / "fixtures",
        ROOT / "tests",
        ROOT / "scripts",
        ROOT / "docs",
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-ai.txt",
        ROOT / "requirements-ai.lock.json",
        ROOT / "ai-runtime-descriptor.json",
        ROOT / "official-ai-model-pack-descriptor.json",
        ROOT / "official-ai-source-descriptor.json",
        ROOT / "vending_vision.spec",
        ROOT / "vending_vision_ai_worker.spec",
    ]
    assert not (ROOT / "vision" / "try_on_session.py").exists()
    assert find_violations(paths) == []


def test_hard_cutover_guard_detects_every_forbidden_category(tmp_path):
    def dot(*parts: str) -> str:
        return ".".join(parts)

    def compact(*parts: str) -> str:
        return "".join(parts)

    fixtures = {
        "protocol.txt": dot("vem", "vision", "v1"),
        "wire.txt": dot("vision", "try_on", "start"),
        "client.txt": compact("tryon", "_", "frontend"),
        "route.txt": f"/{'/'.join(('try-on', '{session}'))}.{compact('m', 'jpeg')}",
        "nested-route.txt": "/".join(("#", "products", "product", "try-on")),
        "selector.txt": f'[data-test="{compact("try", "-on", "-exit")}"]',
        "phase.txt": compact("completed", "Observed"),
        "session.txt": compact("try", "_on", "_session"),
        "owner.txt": compact("fast", "_try", "_on"),
        "standalone-url.txt": "https://" + "/".join(("github.com", "hbhjt", "virtual-tryon.git")),
        "standalone-path.txt": "..\\" + "\\".join(("virtual-tryon", "run.ps1")),
        "standalone-server.txt": dot("app", "main") + ":app",
        "standalone-camera.txt": dot("navigator", "mediaDevices", "getUserMedia"),
    }
    for name, body in fixtures.items():
        (tmp_path / name).write_text(f"{body}\n", encoding="utf-8")

    categories = sorted({entry.rsplit(": ", 1)[1] for entry in find_violations([tmp_path])})

    assert categories == sorted(category for category, _pattern in FORBIDDEN_PATTERNS)
