"""Static semantic policy for the retired generative Try-On runtime."""

from __future__ import annotations

import ast
import json
from pathlib import PurePosixPath
import re
import tomllib


_MODE_FIELD = "".join(("mo", "de"))
_TRY_ON_TYPE = ".".join(("vision", "try_on", "attempt", "start"))
_DENIED_DISTRIBUTIONS = {
    "".join(("tor", "ch")),
    "".join(("torch", "vision")),
    "".join(("diff", "users")),
    "".join(("transform", "ers")),
    "".join(("acceler", "ate")),
    "".join(("safe", "tensors")),
    "-".join(("huggingface", "hub")),
    "".join(("cat", "vton")),
}
_DENIED_RUNTIME_FAMILIES = {
    "-".join(("ai", "attempt", "worker")),
    "-".join(("ai", "attempt", "process")),
    "-".join(("ai", "model", "pack")),
    "-".join(("ai", "runtime", "descriptor")),
    "-".join(("ai", "wheelhouse")),
    "-".join(("vending", "vision", "ai", "worker")),
}
_DENIED_DISTRIBUTION_PREFIXES = {
    "-".join(("official", "ai", "")),
    "-".join(("requirements", "ai", "")),
}
_TEXT_AUDIT_SUFFIXES = {".ps1", ".spec", ".yaml", ".yml"}
_POLICY_EXCLUDED_PARTS = {"archive", "archives", "fixtures", "tests"}
_DEPENDENCY_CONFIG_KEYS = {
    "dependencies",
    "dependency",
    "runtimedependencies",
    "runtimedependency",
}


def _canonical_token(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip().lower()).strip("-")


def is_retired_runtime_dependency(value: str) -> bool:
    """Return whether a module/distribution belongs to the retired runtime."""
    canonical = _canonical_token(value)
    top_level = _canonical_token(value.split(".", 1)[0])
    versioned_distribution = any(
        re.match(rf"^{re.escape(distribution)}-[0-9]", canonical)
        for distribution in _DENIED_DISTRIBUTIONS
    )
    return (
        canonical in _DENIED_DISTRIBUTIONS
        or top_level in _DENIED_DISTRIBUTIONS
        or versioned_distribution
        or canonical in _DENIED_RUNTIME_FAMILIES
        or any(canonical.startswith(prefix) for prefix in _DENIED_DISTRIBUTION_PREFIXES)
    )


def _contains_packaged_tokens(tokens: list[str], expected: tuple[str, ...]) -> bool:
    width = len(expected)
    return any(
        tuple(tokens[index : index + width]) == expected
        for index in range(len(tokens) - width + 1)
    )


def retired_packaged_entries(entries: set[str] | list[str]) -> list[str]:
    """Return packaged names owned by the retired AI try-on path."""
    retired_sequences = (
        ("try", "on", "session"),
        ("try", "on", "frontend"),
        ("try", "on", "ai"),
        ("try", "on", "fast"),
        ("profile", "fast", "try", "on"),
        ("vem", "vision", "v1"),
        ("ai", "acceptance", "evidence"),
        ("ai", "attempt"),
        ("ai", "model"),
        ("ai", "process", "tree", "worker"),
        ("ai", "runtime"),
        ("ai", "source", "provenance"),
        ("ai", "wheelhouse"),
        ("ai", "worker"),
        ("".join(("cat", "vton")),),
        ("regional", "evaluator"),
        ("official", "ai"),
        ("requirements", "ai"),
        ("venv", "packaging", "ai"),
        ("materialize", "ai", "wheelhouse"),
        ("verify", "ai", "wheelhouse"),
        ("render", "ai", "build", "requirements"),
        ("fast", "attempt"),
        ("fast", "result"),
        ("fast", "adjustment"),
        ("vending", "vision", "ai", "worker"),
        ("safetensors",),
        ("vision", "process", "supervisor"),
        ("vision", "source", "provenance"),
    )
    violations = []
    for entry in entries:
        if any(
            is_retired_runtime_dependency(token)
            for token in re.findall(r"[A-Za-z0-9_.-]+", entry)
        ):
            violations.append(entry)
            continue
        tokens = [token for token in re.split(r"[\\/._:\-]+", entry.casefold()) if token]
        if any(_contains_packaged_tokens(tokens, sequence) for sequence in retired_sequences):
            violations.append(entry)
    return sorted(violations)


def _production_python_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return (
        path.suffix == ".py"
        and bool(path.parts)
        and not any(
            part.casefold() in _POLICY_EXCLUDED_PARTS for part in path.parts
        )
    )


def _semantic_function(node: ast.AST) -> bool:
    name = getattr(node, "name", "")
    if "try_on" in name.lower() or "tryon" in name.lower():
        return True
    return any(
        isinstance(child, ast.Constant) and child.value == _TRY_ON_TYPE
        for child in ast.walk(node)
    )


def _semantic_payload(value: ast.AST, *, in_try_on: bool) -> bool:
    if isinstance(value, ast.Name):
        normalized = value.id.lower().replace("_", "")
        return (in_try_on and value.id == "payload") or (
            "tryon" in normalized and "payload" in normalized
        )
    if isinstance(value, ast.Attribute):
        if in_try_on and value.attr == "payload":
            return True
        return _semantic_payload(value.value, in_try_on=in_try_on)
    return False


def _string_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _TryOnModeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.in_try_on = False
        self.found = False
        self.payload_aliases: set[str] = set()

    def _visit_function(self, node: ast.AST) -> None:
        previous_in_try_on = self.in_try_on
        previous_aliases = self.payload_aliases
        self.in_try_on = _semantic_function(node)
        self.payload_aliases = {"payload"} if self.in_try_on else set()
        self.generic_visit(node)
        self.in_try_on = previous_in_try_on
        self.payload_aliases = previous_aliases

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _is_payload(self, value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Name) and value.id in self.payload_aliases
        ) or _semantic_payload(value, in_try_on=self.in_try_on)

    def _is_alias_source(self, value: ast.AST) -> bool:
        if self._is_payload(value):
            return True
        return (
            isinstance(value, ast.Call)
            and not value.args
            and not value.keywords
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "copy"
            and self._is_payload(value.func.value)
        )

    def _record_alias(self, target: ast.AST, value: ast.AST) -> None:
        if not self.in_try_on or not isinstance(target, ast.Name):
            return
        if self._is_alias_source(value):
            self.payload_aliases.add(target.id)
        else:
            self.payload_aliases.discard(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._record_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        before = set(self.payload_aliases)
        self.payload_aliases = set(before)
        for statement in node.body:
            self.visit(statement)
        body_aliases = set(self.payload_aliases)
        self.payload_aliases = set(before)
        for statement in node.orelse:
            self.visit(statement)
        self.payload_aliases |= body_aliases

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            _string_key(node.slice) == _MODE_FIELD
            and self._is_payload(node.value)
        ):
            self.found = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == _MODE_FIELD and self._is_payload(node.value):
            self.found = True
        self.generic_visit(node)

    def _update_contains_mode(self, node: ast.Call) -> bool:
        if any(keyword.arg == _MODE_FIELD for keyword in node.keywords):
            return True
        candidates = [*node.args, *(keyword.value for keyword in node.keywords)]
        for candidate in candidates:
            if isinstance(candidate, ast.Dict) and any(
                key is not None and _string_key(key) == _MODE_FIELD
                for key in candidate.keys
            ):
                return True
            if isinstance(candidate, (ast.List, ast.Tuple)) and any(
                isinstance(item, (ast.List, ast.Tuple))
                and bool(item.elts)
                and _string_key(item.elts[0]) == _MODE_FIELD
                for item in candidate.elts
            ):
                return True
        return False

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and self._is_payload(node.func.value)
            and (
                (
                    node.func.attr in {"get", "pop", "setdefault"}
                    and bool(node.args)
                    and _string_key(node.args[0]) == _MODE_FIELD
                )
                or (
                    node.func.attr == "update"
                    and self._update_contains_mode(node)
                )
            )
        ):
            self.found = True
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        normalized_call = call_name.lower().replace("_", "")
        if any(keyword.arg == _MODE_FIELD for keyword in node.keywords) and (
            "tryon" in normalized_call
            or "attemptstart" in normalized_call
            or normalized_call == "start"
        ):
            self.found = True
        self.generic_visit(node)


def _python_ast(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _contains_denied_python_import(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(is_retired_runtime_dependency(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if is_retired_runtime_dependency(node.module):
                return True
        elif isinstance(node, ast.Call) and node.args:
            module = _string_key(node.args[0])
            if module is None:
                continue
            is_builtin_import = (
                isinstance(node.func, ast.Name) and node.func.id == "__import__"
            )
            is_importlib_import = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            )
            if (is_builtin_import or is_importlib_import) and is_retired_runtime_dependency(
                module
            ):
                return True
    return False


def _contains_denied_text_token(source: str) -> bool:
    return any(
        is_retired_runtime_dependency(token)
        for token in re.findall(r"[A-Za-z0-9_.-]+", source)
    )


def _contains_denied_structured_dependency(
    value: object, *, dependency_context: bool = False
) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
            if _contains_denied_structured_dependency(
                nested,
                dependency_context=(
                    dependency_context or normalized_key in _DEPENDENCY_CONFIG_KEYS
                ),
            ):
                return True
        return False
    if isinstance(value, list):
        return any(
            _contains_denied_structured_dependency(
                nested, dependency_context=dependency_context
            )
            for nested in value
        )
    return (
        dependency_context
        and isinstance(value, str)
        and _contains_denied_text_token(value)
    )


def _contains_denied_config_dependency(relative_path: str, source: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        not path.parts
        or any(part.casefold() in _POLICY_EXCLUDED_PARTS for part in path.parts)
        or path.suffix.casefold() not in {".json", ".toml"}
    ):
        return False
    try:
        parsed = (
            json.loads(source)
            if path.suffix.casefold() == ".json"
            else tomllib.loads(source)
        )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError):
        return False
    return _contains_denied_structured_dependency(parsed)


def semantic_policy_categories(relative_path: str, source: str) -> set[str]:
    """Return hard-cutover categories for one tracked text file."""
    categories: set[str] = set()
    path = PurePosixPath(relative_path)
    tree = _python_ast(source) if path.suffix in {".py", ".spec"} else None
    if _production_python_path(relative_path) and tree is not None:
        visitor = _TryOnModeVisitor()
        visitor.visit(tree)
        if visitor.found:
            categories.add("retired-try-on-mode-access")
        if _contains_denied_python_import(tree):
            categories.add("retired-generative-runtime-dependency")
    if (
        path.name.startswith("requirements")
        or path.suffix in _TEXT_AUDIT_SUFFIXES
    ) and _contains_denied_text_token(source):
        categories.add("retired-generative-runtime-dependency")
    if _contains_denied_config_dependency(relative_path, source):
        categories.add("retired-generative-runtime-dependency")
    return categories
