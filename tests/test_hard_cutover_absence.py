from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[1]
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
        re.compile(
            r"(?:https?://github[.]com/hbhjt/"
            r"|(?:git[+]ssh|ssh)://(?:git@)?github[.]com/hbhjt/"
            r"|git@github[.]com:hbhjt/)virtual-tryon(?:[.]git)?(?![-\w])",
            re.I,
        ),
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
            r"\bapp[.]main:app\b"
            r"|\bfrom\s+app[.]main\s+import\s+[A-Za-z_]\w*"
            r"|\bimport\s+app[.]main\b"
            r"|\bimportlib(?:[.]import_module)?\s*[(]\s*[\"']app[.]main[\"']"
            r"|\buvicorn(?:[.]run)?\s*[(]?\s*[\"']app[.]main:app[\"']"
        ),
    ),
    (
        "standalone-browser-camera-owner",
        re.compile(
            r"\bnavigator\s*"
            r"(?:[?]?[.]\s*mediaDevices|(?:[?][.])?\s*\[\s*[\"']mediaDevices[\"']\s*\])"
            r"\s*(?:[?]?[.]\s*getUserMedia|(?:[?][.])?\s*\[\s*[\"']getUserMedia[\"']\s*\])"
            r"\s*[(]"
        ),
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

# The guard scans this test too. Only these exact line bytes and occurrence
# counts may contain the retired tokens needed to define and assert the guard.
AUDITED_LINE_ALLOWANCES = {
    Path(__file__).resolve(): {
        "legacy-try-on-session-module": {
            "737299ff1dc794babc5f7a76dedc8aa3507bccda8895071fde63c4fc5e5c8251": 3,
            "baa98c45a48033cc3645b5e77950aad08f113317b3c580986439c28606927350": 1,
        },
        "legacy-preview-route": {
            "3eb68b4fabd25f562d16ef0a49a7e21937e342caaa2b1e37e8a976d7bd3dc704": 1,
        },
        "legacy-fast-try-on-owner": {
            "a123179c3edb7f72f45389c0c8a0cb5596ddfbe11bdbf940a606bd03b4891245": 1,
        },
    }
}


def _matches_are_exact_audited_lines(
    path: Path,
    category: str,
    source: str,
    matches: list[re.Match[str]],
) -> bool:
    expected = AUDITED_LINE_ALLOWANCES.get(path.resolve(), {}).get(category)
    if expected is None:
        return False
    lines = source.splitlines()
    observed = Counter(
        hashlib.sha256(lines[source.count("\n", 0, match.start())].encode()).hexdigest()
        for match in matches
    )
    return observed == expected


def _tracked_regular_files(root: Path, diagnostics: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    tracked = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        relative_path = os.fsdecode(raw_path)
        path = root / relative_path
        if mode in {b"100644", b"100755"}:
            tracked.append(path)
        elif mode == b"120000":
            diagnostics.append(f"{path}: tracked-symlink-skipped")
        elif mode == b"160000":
            diagnostics.append(f"{path}: tracked-submodule-skipped")
        else:
            diagnostics.append(f"{path}: tracked-type-{os.fsdecode(mode)}-skipped")
    return tracked


def find_violations(
    root: Path,
    diagnostics: list[str] | None = None,
) -> list[str]:
    root = root.resolve()
    scan_diagnostics = diagnostics if diagnostics is not None else []
    violations = []
    for path in _tracked_regular_files(root, scan_diagnostics):
        try:
            worktree_stat = path.lstat()
        except OSError:
            violations.append(f"{path}: tracked-file-unreadable")
            continue
        if not stat.S_ISREG(worktree_stat.st_mode):
            violations.append(f"{path}: tracked-worktree-type-mismatch")
            continue
        raw_source = path.read_bytes()
        if b"\0" in raw_source:
            scan_diagnostics.append(f"{path}: binary-nul-skipped")
            continue
        try:
            source = raw_source.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            scan_diagnostics.append(f"{path}: binary-non-utf8-skipped")
            continue
        allowance = STANDALONE_PROVENANCE_ALLOWANCES.get(path.resolve())
        allowance_valid = allowance is not None and hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest() == allowance["sha256"]
        if allowance is not None and not allowance_valid:
            violations.append(f"{path}: standalone-provenance-integrity")
        for category, pattern in FORBIDDEN_PATTERNS:
            matches = list(pattern.finditer(source))
            if not matches:
                continue
            if _matches_are_exact_audited_lines(path, category, source, matches):
                continue
            permitted = allowance["occurrences"].get(category) if allowance_valid else None
            if permitted == len(matches):
                continue
            violations.append(f"{path}: {category}")
    return violations


def test_retired_vision_session_and_v1_surface_is_absent():
    assert not (ROOT / "vision" / "try_on_session.py").exists()
    assert find_violations(ROOT) == []


def test_hard_cutover_guard_detects_every_forbidden_category(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

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
        "standalone-url.txt": "https://"
        + "/".join(("github.com", "hbhjt", "virtual-tryon.git")),
        "standalone-path.txt": "..\\" + "\\".join(("virtual-tryon", "run.ps1")),
        "standalone-server.txt": dot("app", "main") + ":app",
        "standalone-camera.txt": dot("navigator", "mediaDevices", "getUserMedia")
        + "()",
    }
    for name, body in fixtures.items():
        (tmp_path / name).write_text(f"{body}\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", *fixtures], cwd=tmp_path, check=True)

    categories = sorted(
        {entry.rsplit(": ", 1)[1] for entry in find_violations(tmp_path)}
    )

    assert categories == sorted(category for category, _pattern in FORBIDDEN_PATTERNS)


def test_hard_cutover_guard_scans_every_tracked_regular_file(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    forbidden = "https://" + "/".join(("github.com", "hbhjt", "virtual-tryon"))
    tracked = (
        "run.ps1",
        "app/main.py",
        "deployment/deploy.ps1",
        "arbitrary/reference.sh",
        "arbitrary/reference.bat",
        "arbitrary/reference.toml",
        "arbitrary/reference.psm1",
    )
    for relative_path in tracked:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{forbidden}\n", encoding="utf-8")
    ignored = tmp_path / "untracked.py"
    ignored.write_text(f"{forbidden}\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", *tracked], cwd=tmp_path, check=True)

    violations = find_violations(tmp_path)

    assert sorted(entry.split(": ", 1)[0] for entry in violations) == sorted(
        str(tmp_path / relative_path) for relative_path in tracked
    )
    assert all("untracked.py" not in entry for entry in violations)


def test_hard_cutover_guard_rejects_standalone_dependency_variants_and_guard_self_hiding(
    tmp_path,
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    dot = "."
    repository = "/".join(("github.com", "hbhjt", "virtual-tryon.git"))
    module = dot.join(("app", "main"))
    media = "media" + "Devices"
    capture = "get" + "User" + "Media"
    fixtures = {
        "url-https.py": "https://" + repository,
        "url-git-ssh.sh": "git+ssh://git@" + repository,
        "url-scp.toml": "git@github.com:"
        + "/".join(("hbhjt", "virtual-tryon.git")),
        "path-relative.bat": "..\\" + "\\".join(("virtual-tryon", "run.ps1")),
        "path-posix.psm1": "/opt/" + "/".join(("virtual-tryon", "run.ps1")),
        "path-windows.py": "C:\\src\\" + "\\".join(("virtual-tryon", "run.ps1")),
        "server-from.py": f"from {module} import app",
        "server-import.py": f"import {module}",
        "server-importlib.py": f'importlib.import_module("{module}")',
        "server-uvicorn.py": f'uvicorn.run("{module}:app")',
        "camera-dot.js": f"navigator.{media}.{capture}()",
        "camera-optional.js": f"navigator?.{media}?.{capture}()",
        "camera-bracket.js": f'navigator["{media}"]["{capture}"]()',
        "camera-mixed.js": f'navigator?.["{media}"]?.{capture}()',
        "tests/test_hard_cutover_absence.py": (
            "subprocess.run([\"powershell\", \"../"
            + "/".join(("virtual-tryon", "run.ps1"))
            + "\"], check=True)"
        ),
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{source}\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", *fixtures], cwd=tmp_path, check=True)

    violations = find_violations(tmp_path)

    assert {
        Path(entry.split(": ", 1)[0]).relative_to(tmp_path).as_posix()
        for entry in violations
    } == set(fixtures)


def test_hard_cutover_guard_allows_similar_non_dependencies(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    capture = "get" + "User" + "Media"
    similar = "\n".join(
        (
            "https://"
            + "/".join(("github.com", "hbhjt", "virtual-tryon-docs")),
            "from " + ".".join(("myapp", "main")) + " import app",
            "camera." + capture + "()",
            ".".join(("navigator", "mediaDevices", "enumerateDevices")) + "()",
        )
    )
    (tmp_path / "similar.txt").write_text(similar, encoding="utf-8")
    subprocess.run(["git", "add", "similar.txt"], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_records_binary_symlink_and_submodule_types(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "nul.bin").write_bytes(b"text\0payload")
    (tmp_path / "non-utf8.bin").write_bytes(b"\xff\xfe")
    (tmp_path / "target.txt").write_text("not tracked\n", encoding="utf-8")
    os.symlink("target.txt", tmp_path / "reference-link")
    (tmp_path / "missing.txt").write_text("tracked\n", encoding="utf-8")
    (tmp_path / "replaced.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "nul.bin",
            "non-utf8.bin",
            "reference-link",
            "missing.txt",
            "replaced.txt",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            "160000,1111111111111111111111111111111111111111,vendor/reference",
        ],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "missing.txt").unlink()
    (tmp_path / "replaced.txt").unlink()
    os.symlink("target.txt", tmp_path / "replaced.txt")
    diagnostics = []

    violations = find_violations(tmp_path, diagnostics)

    assert violations == [
        f"{tmp_path / 'missing.txt'}: tracked-file-unreadable",
        f"{tmp_path / 'replaced.txt'}: tracked-worktree-type-mismatch",
    ]
    assert sorted(entry.rsplit(": ", 1)[-1] for entry in diagnostics) == [
        "binary-non-utf8-skipped",
        "binary-nul-skipped",
        "tracked-submodule-skipped",
        "tracked-symlink-skipped",
    ]
