"""Static semantic policy for the retired generative Try-On runtime."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath
import re


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


def _production_python_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    return path.suffix == ".py" and (
        path.as_posix() in {"app.py", "run_vision_server.py"}
        or (bool(path.parts) and path.parts[0] in {"vision", "scripts"})
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

    def _visit_function(self, node: ast.AST) -> None:
        previous = self.in_try_on
        self.in_try_on = previous or _semantic_function(node)
        self.generic_visit(node)
        self.in_try_on = previous

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            _string_key(node.slice) == _MODE_FIELD
            and _semantic_payload(node.value, in_try_on=self.in_try_on)
        ):
            self.found = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == _MODE_FIELD and _semantic_payload(
            node.value, in_try_on=self.in_try_on
        ):
            self.found = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and _string_key(node.args[0]) == _MODE_FIELD
            and _semantic_payload(node.func.value, in_try_on=self.in_try_on)
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
    return False


def _contains_denied_text_token(source: str) -> bool:
    return any(
        is_retired_runtime_dependency(token)
        for token in re.findall(r"[A-Za-z0-9_.-]+", source)
    )


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
    return categories
