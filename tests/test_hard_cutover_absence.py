from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THIS_TEST = Path(__file__).resolve()
NEGATIVE_FIXTURE = ROOT / "tests" / "fixtures" / "hard-cutover-forbidden.txt"


def find_violations(paths, forbidden, *, include_negative_fixture=False):
    source_files = [
        path
        for root in paths
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file()
        and path.suffix not in {".pyc", ".mp4", ".png", ".jpg", ".jpeg", ".log"}
        and path != THIS_TEST
        and (include_negative_fixture or path != NEGATIVE_FIXTURE)
    ]
    return [
        f"{path}: {token}"
        for path in source_files
        for token in forbidden
        if token in path.read_text(encoding="utf-8", errors="ignore")
    ]


def test_retired_vision_session_and_v1_surface_is_absent():
    forbidden = (
        "vem.vision.v1",
        "vision.try_on.start",
        "vision.try_on.stop",
        "vision.try_on.started",
        "vision.try_on.stopped",
        "tryon_frontend",
        "/try-on/{session}.mjpeg",
    )
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
    assert find_violations(paths, forbidden) == []


def test_hard_cutover_guard_detects_every_forbidden_category():
    forbidden = tuple(
        line for line in NEGATIVE_FIXTURE.read_text(encoding="utf-8").splitlines() if line
    )

    violations = find_violations(
        [NEGATIVE_FIXTURE], forbidden, include_negative_fixture=True
    )

    assert violations == [f"{NEGATIVE_FIXTURE}: {token}" for token in forbidden]
