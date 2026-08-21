from collections import Counter
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess

import pytest

from scripts.hard_cutover_policy import retired_packaged_entries, semantic_policy_categories


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

_GENERATION_PREFIX = "".join(("a", "i"))
_QUICK_PREFIX = "".join(("fa", "st"))
_PROFILE_RETIRED_ROLE = "".join(("profile", "_fast", "_try_on"))
_VENDOR_RUNTIME = "".join(("cat", "vton"))
_REGIONAL_RUNTIME = "".join(("regional", "_evaluator"))
_REGIONAL_SYMBOL = "".join(("Regional", "Evaluator"))
_RETIRED_MODE_VALUES = "|".join((_GENERATION_PREFIX, _QUICK_PREFIX))

RETIRED_TRY_ON_EXACT_PATHS = {
    "vision/process_supervisor.py",
    "vision/source_provenance.py",
}
HISTORICAL_FIXTURE_DOCUMENTS = {"fixtures/recorded-video/README.md"}

SINGLE_PATH_CONTENT_RULES = (
    (
        "retired-ai-try-on-mode",
        re.compile(
            rf"(?<![\w])(?:[\"']mode[\"']|mode)\s*[:=]\s*"
            rf"(?:[\"'](?:{_RETIRED_MODE_VALUES})[\"']|(?:{_RETIRED_MODE_VALUES})(?!\w))",
            re.I,
        ),
    ),
    (
        "retired-ai-try-on-bracket-mode",
        re.compile(
            r"payload\[\s*[\"']mode[\"']\s*\]\s*=\s*"
            r"(?:[\"'][^\"'\r\n]+[\"']|[A-Za-z_]\w*)",
            re.I,
        ),
    ),
    (
        "retired-try-on-start-mode",
        re.compile(
            r"\b(?:TryOnStart|Start)\s*\([^)]*\bmode\s*=\s*"
            r"(?:[\"'][^\"'\r\n]+[\"']|[A-Za-z_]\w*)",
            re.I,
        ),
    ),
    (
        "retired-try-on-payload-mode",
        re.compile(
            r"\btry_on(?:_start)?_payload\s*=\s*\{[^}\r\n]*"
            r"(?:[\"']mode[\"']|mode)\s*:",
            re.I,
        ),
    ),
    (
        "retired-try-on-mode-symbol",
        re.compile(
            rf"\b(?:try_on_(?:{_GENERATION_PREFIX}|{_QUICK_PREFIX})|{_PROFILE_RETIRED_ROLE})\b",
            re.I,
        ),
    ),
    ("retired-ai-try-on-readiness", re.compile(r"\b(?:ai|fast)(?:Ready|ReadinessDiagnostic)\b", re.I)),
    (
        "retired-ai-try-on-terminal",
        re.compile(rf"\b(?:{_GENERATION_PREFIX}|{_QUICK_PREFIX})_failed\b", re.I),
    ),
    (
        "retired-try-on-compatibility-symbol",
        re.compile(
            rf"\b(?:{_QUICK_PREFIX}(?:RenderBroker|AttemptRegistry|ResultStore|AdjustmentStore)"
            rf"|{_GENERATION_PREFIX}(?:AcceptanceEvidence|AttemptProcess|AttemptWorker|ModelPack|ProcessTreeWorker|RuntimeDescriptor)"
            rf"|{_REGIONAL_SYMBOL})\b"
            rf"|(?<![\w])_(?:{_QUICK_PREFIX}|{_GENERATION_PREFIX})_"
            r"(?:runtime|render_broker|attempt_registry|result_store|adjustment_store)\b",
            re.I,
        ),
    ),
    ("retired-fast-try-on-role", re.compile(r"\bprofile_fast_try_on\b")),
    (
        "retired-ai-try-on-artifact",
        re.compile(
            rf"\b(?:VEM_{_GENERATION_PREFIX}_[A-Z0-9_]+"
            rf"|vending[-_]vision[-_]{_GENERATION_PREFIX}[-_]worker"
            rf"|requirements-{_GENERATION_PREFIX}|official-{_GENERATION_PREFIX}"
            rf"|{_VENDOR_RUNTIME}|regional[-_]evaluator"
            rf"|{_GENERATION_PREFIX}[-_](?:worker|runtime|model(?:[-_]pack)?|wheelhouse|env(?:ironment)?))\b",
            re.I,
        ),
    ),
)

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


def _tracked_entries(root: Path) -> list[tuple[Path, str]]:
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
        mode = os.fsdecode(metadata.split(b" ", 1)[0])
        relative_path = os.fsdecode(raw_path)
        path = root / relative_path
        tracked.append((path, mode))
    return tracked


def find_violations(
    root: Path,
) -> list[str]:
    root = root.resolve()
    violations = []
    tracked_entries = _tracked_entries(root)
    for path, git_mode in tracked_entries:
        if git_mode == "120000":
            violations.append(f"{path}: tracked-symlink-forbidden")
            continue
        if git_mode == "160000":
            violations.append(f"{path}: tracked-submodule-forbidden")
            continue
        if git_mode not in {"100644", "100755"}:
            violations.append(f"{path}: tracked-mode-{git_mode}-forbidden")
            continue
        try:
            worktree_stat = path.lstat()
        except OSError:
            violations.append(f"{path}: tracked-file-unreadable")
            continue
        if not stat.S_ISREG(worktree_stat.st_mode):
            violations.append(f"{path}: tracked-worktree-type-mismatch")
            continue
        raw_source = path.read_bytes()
        relative_path = path.relative_to(root).as_posix()
        if (
            relative_path.lower() in RETIRED_TRY_ON_EXACT_PATHS
            or retired_packaged_entries([relative_path])
        ):
            violations.append(f"{path}: retired-try-on-path")
            continue
        if b"\0" in raw_source:
            continue
        try:
            source = raw_source.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        for category, pattern in FORBIDDEN_PATTERNS:
            matches = list(pattern.finditer(source))
            if not matches:
                continue
            if (
                category == "standalone-repository-url"
                and relative_path in HISTORICAL_FIXTURE_DOCUMENTS
            ):
                continue
            if _matches_are_exact_audited_lines(path, category, source, matches):
                continue
            violations.append(f"{path}: {category}")
        for category, pattern in SINGLE_PATH_CONTENT_RULES:
            matches = list(pattern.finditer(source))
            if not matches:
                continue
            if _matches_are_exact_audited_lines(path, category, source, matches):
                continue
            violations.append(f"{path}: {category}")
        for category in sorted(semantic_policy_categories(relative_path, source)):
            violations.append(f"{path}: {category}")
    return violations


def _init_guard_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)


def test_retired_vision_session_and_v1_surface_is_absent():
    assert not (ROOT / "vision" / "try_on_session.py").exists()
    assert find_violations(ROOT) == []


def test_hard_cutover_guard_rejects_new_retired_try_on_path(tmp_path):
    _init_guard_repo(tmp_path)
    retired = tmp_path / "vision" / "ai_attempt_runtime.py"
    retired.parent.mkdir()
    retired.write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "vision/ai_attempt_runtime.py"], cwd=tmp_path, check=True)
    assert any("retired-try-on-path" in item for item in find_violations(tmp_path))


def test_hard_cutover_guard_rejects_ai_mode_readiness_role_and_artifact_variants(tmp_path):
    _init_guard_repo(tmp_path)
    joined = "".join
    payload_mode_assignment = joined(('payload["', "mode", '"] = '))
    fixtures = {
        "object.py": '{"mode":"' + joined(("a", "i")) + '"}\n',
        "object.ts": 'const request = { mode: "' + _QUICK_PREFIX + '" };\n',
        "constructor.py": 'request = Start(mode="' + joined(("a", "i")) + '")\n',
        "bracket.py": payload_mode_assignment + '"' + _QUICK_PREFIX + '"\n',
        "bare-object.py": 'request = { mode: ' + joined(("a", "i")) + ' }\n',
        "bare-assignment.py": 'mode = ' + _QUICK_PREFIX + '\n',
        "bare-bracket.py": payload_mode_assignment + joined(("a", "i")) + '\n',
        "automatic.py": payload_mode_assignment + joined(("auto", "matic")) + '\n',
        "start-automatic.py": (
            'request = ' + joined(("Try", "On", "Start"))
            + '(mode="' + joined(("auto", "matic")) + '")\n'
        ),
        "named-payload.py": (
            joined(("try", "_on", "_payload"))
            + ' = {"' + joined(("mo", "de")) + '": "legacy"}\n'
        ),
        "ready.py": 'state = "' + joined(("a", "i", "Ready")) + '"\n',
        "quick-ready.py": 'state = "' + _QUICK_PREFIX + 'ReadinessDiagnostic"\n',
        "role.json": '{"role":"' + joined(("profile", "_fast", "_try_on")) + '"}\n',
        "symbol.py": (
            'handler = ' + joined(("try_on_", "a", "i"))
            + ' or ' + joined(("try_on_", "fa", "st")) + '\n'
        ),
        "artifact.py": 'env = "VEM_' + joined(("A", "I", "_MODEL_PACK")) + '"\n',
        "worker.py": 'entry = "' + joined(("a", "i", "-worker")) + '"\n',
        "environment.py": 'root = "' + joined(("a", "i", "_wheelhouse")) + '"\n',
        "model.py": 'package = "' + joined(("a", "i", "-model-pack")) + '"\n',
        "vendor.py": 'runtime = "' + joined(("Cat", "VTON")) + '"\n',
        "regional.py": 'descriptor = "' + joined(("regional", "_evaluator")) + '"\n',
        "runtime.py": 'runtime = "' + joined(("a", "i", "_runtime")) + '"\n',
        "terminal.py": 'reason = "' + joined(("a", "i", "_failed")) + '"\n',
        "compatibility.py": (
            'broker = ' + joined(("Fa", "st", "RenderBroker"))
            + '; registry = ' + joined(("Fa", "st", "AttemptRegistry"))
            + '; runtime = _' + joined(("fa", "st", "_runtime")) + '\n'
        ),
        "generation-symbol.py": (
            'process = ' + joined(("A", "i", "AttemptProcess"))
            + '; evidence = ' + joined(("A", "i", "AcceptanceEvidence"))
            + '; tree = ' + joined(("A", "i", "ProcessTreeWorker"))
            + '; regional = ' + joined(("Regional", "Evaluator")) + '\n'
        ),
    }
    for name, source in fixtures.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)
    violations = find_violations(tmp_path)
    violated_paths = {Path(item.rsplit(": ", 1)[0]).name for item in violations}
    assert violated_paths == set(fixtures)
    assert {item.rsplit(": ", 1)[-1] for item in violations} == {
        "retired-ai-try-on-artifact",
        "retired-ai-try-on-bracket-mode",
        "retired-ai-try-on-mode",
        "retired-ai-try-on-readiness",
        "retired-ai-try-on-terminal",
        "retired-try-on-compatibility-symbol",
        "retired-try-on-start-mode",
        "retired-try-on-payload-mode",
        "retired-fast-try-on-role",
        "retired-try-on-mode-symbol",
        "retired-try-on-mode-access",
    }


@pytest.mark.parametrize(
    "relative_path",
    (
        "vision/" + "".join(("try_on_", "a", "i", ".py")),
        "vision/" + "".join(("try_on_", "fa", "st", ".py")),
        "config/" + "".join(("profile", "_fast", "_try_on.json")),
        "runtime/" + "".join(("a", "i", "-worker/entry.py")),
        "models/" + "".join(("a", "i", "_model_pack/manifest.json")),
        "vision/" + "".join(("regional", "_evaluator/runtime.py")),
        "vision/" + "".join(("a", "i", "_acceptance_evidence.py")),
        "tests/" + "".join(("a", "i", "_process_tree_worker.py")),
        "scripts/" + "".join(("materialize_", "a", "i", "_wheelhouse.py")),
        "scripts/" + "".join(("verify_", "a", "i", "_wheelhouse.py")),
        "scripts/" + "".join(("render_", "a", "i", "_build_requirements.py")),
        "vision/" + "".join(("process", "_supervisor.py")),
        "vision/" + "".join(("source", "_provenance.py")),
    ),
)
def test_hard_cutover_guard_rejects_every_retired_path_category(
    tmp_path, relative_path
):
    _init_guard_repo(tmp_path)
    retired = tmp_path / relative_path
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "add", relative_path], cwd=tmp_path, check=True)

    assert any("retired-try-on-path" in item for item in find_violations(tmp_path))


def test_hard_cutover_guard_allows_framework_performance_and_production_models(tmp_path):
    _init_guard_repo(tmp_path)
    fixtures = {
        "framework.py": "from fastapi import FastAPI\napp = FastAPI()\n",
        "performance.md": "Fast startup and fast frame processing are performance goals.\n",
        "mode.py": 'mode = "performance"\n',
        "automatic-mode.py": 'mode = "automatic"\n',
        "framework-mode.py": 'mode = "fastapi"\n',
        "performance-mode.py": 'mode = fastest\n',
        "pipeline-mode.py": 'mode = airflow\n',
        "official-air-guide.md": "Official airflow guidance.\n",
        "requirements-airflow.md": "Requirements for airflow monitoring.\n",
        "models.json": (
            '{"weights":["models/age_gender/age_net.caffemodel",'
            '"models/age_gender/gender_net.caffemodel",'
            '"models/person_detection/person_yolov8n.onnx"]}\n'
        ),
    }
    for name, source in fixtures.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_allows_future_scoped_generic_runtime_modules(tmp_path):
    _init_guard_repo(tmp_path)
    fixtures = {
        "vision/directshow/source_provenance.py": "class DirectShowProvenance:\n    pass\n",
        "tools/process_supervisor.py": "class MaintenanceSupervisor:\n    pass\n",
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_semantically_rejects_try_on_payload_mode_accesses(tmp_path):
    _init_guard_repo(tmp_path)
    mode_key = "".join(("mo", "de"))
    source = "\n".join(
        (
            "def run_v2_try_on_attempt(payload, try_on_payload):",
            f'    direct = payload["{mode_key}"]',
            f'    optional = payload.get("{mode_key}")',
            f"    attribute = try_on_payload.{mode_key}",
            f"    request = TryOnStart({mode_key}=automatic)",
            "    return direct, optional, attribute, request",
            "",
        )
    )
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)

    assert any(
        item.endswith(": retired-try-on-mode-access")
        for item in find_violations(tmp_path)
    )


@pytest.mark.parametrize(
    "body",
    (
        "request = payload.copy()\n    return request['mode']",
        "request = payload\n    return request.get('mode')",
        "return payload.pop('mode')",
        "request = payload.copy()\n    return request.setdefault('mode', 'automatic')",
        "request = payload.copy()\n    request.update({'mode': 'automatic'})",
        "request = payload\n    alias = request.copy()\n    alias.update(mode='automatic')",
        (
            "request = payload\n    if reset:\n        request = {}\n"
            "    return request.get('mode')"
        ),
    ),
    ids=(
        "copy-subscript",
        "direct-get",
        "direct-pop",
        "setdefault",
        "update-map",
        "transitive-update-keyword",
        "conditional-reassignment",
    ),
)
def test_hard_cutover_guard_rejects_try_on_payload_alias_mode_operations(
    tmp_path, body
):
    _init_guard_repo(tmp_path)
    source = f"def handle_try_on(payload):\n    {body}\n"
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)

    assert any(
        item.endswith(": retired-try-on-mode-access")
        for item in find_violations(tmp_path)
    )


def test_hard_cutover_guard_allows_unconditionally_replaced_try_on_alias(tmp_path):
    _init_guard_repo(tmp_path)
    source = "\n".join(
        (
            "def handle_try_on(payload):",
            "    request = payload.copy()",
            "    request = {}",
            "    return request.get('mode')",
            "",
        )
    )
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_allows_unrelated_semantic_mode_uses(tmp_path):
    _init_guard_repo(tmp_path)
    source = "\n".join(
        (
            "from fastapi import FastAPI",
            "def configure_camera(camera_payload, image, gender_mode):",
            '    camera_mode = camera_payload.get("mode")',
            "    image_mode = image.mode",
            '    camera = Camera(mode="dshow")',
            '    pipeline = Pipeline(mode="airflow")',
            "    return camera_mode, image_mode, gender_mode, camera, pipeline",
            "",
        )
    )
    path = tmp_path / "vision" / "camera.py"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "vision/camera.py"], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_rejects_generative_runtime_dependencies_everywhere(tmp_path):
    _init_guard_repo(tmp_path)
    joined = "".join
    fixtures = {
        "requirements.txt": joined(("tor", "ch")) + "==2.0.0\n",
        "requirements-build.txt": joined(("torch", "vision")) + "==0.15.0\n",
        "vision/runtime.py": "import " + joined(("diff", "users")) + "\n",
        "vision/loader.py": "from " + joined(("transform", "ers")) + " import AutoModel\n",
        "scripts/runtime.py": "import " + joined(("acceler", "ate")) + "\n",
        "scripts/weights.py": "from " + joined(("safe", "tensors")) + " import safe_open\n",
        "scripts/hub.py": "import " + joined(("huggingface", "_hub")) + "\n",
        "vending_vision.spec": (
            "hiddenimports = [\"" + joined(("cat", "vton")) + "\"]\n"
        ),
        ".github/workflows/ci.yml": (
            "steps:\n  - run: pip install " + joined(("diff", "users")) + "\n"
        ),
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    violations = find_violations(tmp_path)
    dependency_paths = {
        Path(item.rsplit(": ", 1)[0]).relative_to(tmp_path).as_posix()
        for item in violations
        if item.endswith(": retired-generative-runtime-dependency")
    }
    assert dependency_paths == set(fixtures)


@pytest.mark.parametrize(
    "source",
    (
        "import importlib\nruntime = importlib.import_module('torch')\n",
        "runtime = __import__('diffusers')\n",
    ),
    ids=("importlib", "builtin-import"),
)
def test_hard_cutover_guard_rejects_constant_dynamic_runtime_imports(
    tmp_path, source
):
    _init_guard_repo(tmp_path)
    path = tmp_path / "vision" / "dynamic_loader.py"
    path.parent.mkdir()
    path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "vision/dynamic_loader.py"], cwd=tmp_path, check=True)

    assert any(
        item.endswith(": retired-generative-runtime-dependency")
        for item in find_violations(tmp_path)
    )


def test_hard_cutover_guard_scans_new_tracked_production_python_without_root_allowlist(
    tmp_path,
):
    _init_guard_repo(tmp_path)
    production = tmp_path / "runtime_plugin.py"
    production.write_text("import transformers\n", encoding="utf-8")
    for relative_path in ("tests/attack_fixture.py", "archive/legacy_runtime.py"):
        historical = tmp_path / relative_path
        historical.parent.mkdir(parents=True, exist_ok=True)
        historical.write_text("import diffusers\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "runtime_plugin.py",
            "tests/attack_fixture.py",
            "archive/legacy_runtime.py",
        ],
        cwd=tmp_path,
        check=True,
    )

    dependency_paths = {
        Path(item.rsplit(": ", 1)[0]).relative_to(tmp_path).as_posix()
        for item in find_violations(tmp_path)
        if item.endswith(": retired-generative-runtime-dependency")
    }
    assert dependency_paths == {"runtime_plugin.py"}


@pytest.mark.parametrize(
    ("relative_path", "source"),
    (
        (
            "pyproject.toml",
            "[project]\nname='attack'\ndependencies=['diffusers==1.0']\n",
        ),
        (
            "config/runtime.json",
            '{"runtimeDependency":"transformers"}\n',
        ),
    ),
    ids=("pyproject-dependency", "runtime-json-dependency"),
)
def test_hard_cutover_guard_rejects_consumed_dependency_and_config_entries(
    tmp_path, relative_path, source
):
    _init_guard_repo(tmp_path)
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", relative_path], cwd=tmp_path, check=True)

    assert any(
        item.endswith(": retired-generative-runtime-dependency")
        for item in find_violations(tmp_path)
    )


def test_hard_cutover_guard_allows_production_non_generative_configs(tmp_path):
    _init_guard_repo(tmp_path)
    fixtures = {
        "pyproject.toml": (
            "[project]\nname='runtime'\ndependencies=['fastapi==1.0', 'airflow-monitor==1.0']\n"
        ),
        "config/runtime.json": (
            '{"cameraMode":"dshow","imageMode":"RGBA",'
            '"genderMode":"automatic","runtimeDependency":"fastapi"}\n'
        ),
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_excludes_nested_test_fixtures_and_historical_archives(
    tmp_path,
):
    _init_guard_repo(tmp_path)
    fixtures = {
        "contracts/v2/fixtures/attack.json": (
            '{"runtimeDependency":"transformers"}\n'
        ),
        "docs/archive/legacy.toml": "dependencies=['diffusers==1.0']\n",
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_allows_similar_runtime_dependencies(tmp_path):
    _init_guard_repo(tmp_path)
    fixtures = {
        "requirements.txt": "fastapi==1.0.0\nairflow-monitor==1.0.0\n",
        "vision/runtime.py": (
            "from fastapi import FastAPI\n"
            "import airflow_monitor\n"
            "from vision.age_gender_estimator import AgeGenderEstimator\n"
        ),
        "vending_vision.spec": 'hiddenimports = ["fastapi", "vision.model_manifest"]\n',
    }
    for relative_path, source in fixtures.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", *fixtures], cwd=tmp_path, check=True)

    assert find_violations(tmp_path) == []


def test_hard_cutover_guard_does_not_trust_a_same_named_test_file(tmp_path):
    _init_guard_repo(tmp_path)
    disguised = tmp_path / "tests" / "test_hard_cutover_absence.py"
    disguised.parent.mkdir()
    disguised.write_text(
        'payload["mode"] = "' + _QUICK_PREFIX + '"\n', encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "tests/test_hard_cutover_absence.py"],
        cwd=tmp_path,
        check=True,
    )

    assert any(
        "retired-ai-try-on-bracket-mode" in item
        for item in find_violations(tmp_path)
    )


def test_hard_cutover_guard_detects_every_forbidden_category(tmp_path):
    _init_guard_repo(tmp_path)

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
    _init_guard_repo(tmp_path)
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
    _init_guard_repo(tmp_path)
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
    _init_guard_repo(tmp_path)
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


def test_hard_cutover_guard_rejects_symlink_submodule_and_type_drift(tmp_path):
    _init_guard_repo(tmp_path)
    (tmp_path / "target.txt").write_text("not tracked\n", encoding="utf-8")
    os.symlink("target.txt", tmp_path / "reference-link")
    (tmp_path / "missing.txt").write_text("tracked\n", encoding="utf-8")
    (tmp_path / "replaced.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
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
    (tmp_path / "reference-link").unlink()
    (tmp_path / "reference-link").write_text("regular drift\n", encoding="utf-8")
    (tmp_path / "replaced.txt").unlink()
    os.symlink("target.txt", tmp_path / "replaced.txt")
    violations = find_violations(tmp_path)

    assert violations == [
        f"{tmp_path / 'missing.txt'}: tracked-file-unreadable",
        f"{tmp_path / 'reference-link'}: tracked-symlink-forbidden",
        f"{tmp_path / 'replaced.txt'}: tracked-worktree-type-mismatch",
        f"{tmp_path / 'vendor/reference'}: tracked-submodule-forbidden",
    ]
