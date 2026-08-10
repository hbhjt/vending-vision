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
)


def find_violations(paths):
    source_files = [
        path
        for root in paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file()
        and path.suffix not in {".pyc", ".mp4", ".png", ".jpg", ".jpeg", ".log"}
        and path != THIS_TEST
    ]
    return [
        f"{path}: {category}"
        for path in source_files
        for category, pattern in FORBIDDEN_PATTERNS
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]


def test_retired_vision_session_and_v1_surface_is_absent():
    paths = [
        ROOT / "app.py",
        ROOT / "vision",
        ROOT / "config",
        ROOT / "contracts",
        ROOT / "tests",
        ROOT / "scripts",
        ROOT / "docs",
        ROOT / "README.md",
        ROOT / "vending_vision.spec",
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
    }
    for name, body in fixtures.items():
        (tmp_path / name).write_text(f"{body}\n", encoding="utf-8")

    categories = sorted({entry.rsplit(": ", 1)[1] for entry in find_violations([tmp_path])})

    assert categories == sorted(category for category, _pattern in FORBIDDEN_PATTERNS)
