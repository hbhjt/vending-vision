from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    paths = [ROOT / "app.py", ROOT / "vision", ROOT / "config", ROOT / "contracts"]
    source_files = [
        path for root in paths for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix not in {".pyc"}
    ]
    assert not (ROOT / "vision" / "try_on_session.py").exists()
    violations = [
        f"{path}: {token}"
        for path in source_files
        for token in forbidden
        if token in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert violations == []
