from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


class WorkflowYamlError(ValueError):
    pass


_MAPPING_ENTRY = re.compile(r"^(?P<key>[A-Za-z0-9_.-]+):(?:[ ](?P<value>.*))?$")
_BLOCK_HEADER = re.compile(r"^(?P<style>[|>])(?:[+-]?[1-9]?|[1-9][+-]?)$")


@dataclass(frozen=True)
class _Line:
    number: int
    indent: int
    content: str
    raw: str


class _Parser:
    def __init__(self, source: str) -> None:
        self._lines: list[_Line] = []
        for number, raw in enumerate(source.splitlines(), start=1):
            leading = raw[: len(raw) - len(raw.lstrip(" \t"))]
            if "\t" in leading:
                raise WorkflowYamlError(f"tab_indentation:{number}")
            self._lines.append(_Line(number, len(leading), raw[len(leading) :], raw))
        self._index = 0

    @staticmethod
    def _is_ignorable(line: _Line) -> bool:
        return not line.content or line.content.startswith("#")

    def _skip_ignorable(self) -> None:
        while self._index < len(self._lines) and self._is_ignorable(self._lines[self._index]):
            self._index += 1

    def parse(self) -> dict[str, Any]:
        self._skip_ignorable()
        if self._index == len(self._lines):
            raise WorkflowYamlError("empty_workflow")
        root = self._parse_block(self._lines[self._index].indent)
        self._skip_ignorable()
        if self._index != len(self._lines):
            line = self._lines[self._index]
            raise WorkflowYamlError(f"trailing_content:{line.number}")
        if not isinstance(root, dict):
            raise WorkflowYamlError("workflow_root_not_mapping")
        return root

    def _parse_block(self, indent: int) -> Any:
        self._skip_ignorable()
        if self._index == len(self._lines):
            return None
        line = self._lines[self._index]
        if line.indent != indent:
            raise WorkflowYamlError(f"unexpected_indentation:{line.number}")
        if line.content == "-" or line.content.startswith("- "):
            return self._parse_sequence(indent)
        return self._parse_mapping(indent)

    def _parse_mapping(self, indent: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while True:
            self._skip_ignorable()
            if self._index == len(self._lines):
                break
            line = self._lines[self._index]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise WorkflowYamlError(f"unexpected_indentation:{line.number}")
            if line.content == "-" or line.content.startswith("- "):
                break
            self._index += 1
            key, value = self._parse_mapping_entry(line.content, line.number, indent)
            if key in result:
                raise WorkflowYamlError(f"duplicate_key:{line.number}:{key}")
            result[key] = value
        return result

    def _parse_sequence(self, indent: int) -> list[Any]:
        result: list[Any] = []
        while True:
            self._skip_ignorable()
            if self._index == len(self._lines):
                break
            line = self._lines[self._index]
            if line.indent < indent:
                break
            if line.indent != indent:
                raise WorkflowYamlError(f"unexpected_indentation:{line.number}")
            if not (line.content == "-" or line.content.startswith("- ")):
                break
            self._index += 1
            payload = line.content[1:].lstrip(" ")
            if not payload:
                self._skip_ignorable()
                if self._index == len(self._lines) or self._lines[self._index].indent <= indent:
                    result.append(None)
                else:
                    result.append(self._parse_block(self._lines[self._index].indent))
                continue
            if _MAPPING_ENTRY.match(payload):
                item: dict[str, Any] = {}
                key, value = self._parse_mapping_entry(payload, line.number, indent + 2)
                item[key] = value
                self._skip_ignorable()
                if self._index < len(self._lines) and self._lines[self._index].indent > indent:
                    continuation_indent = self._lines[self._index].indent
                    if continuation_indent != indent + 2:
                        raise WorkflowYamlError(
                            f"unexpected_sequence_mapping_indentation:{self._lines[self._index].number}"
                        )
                    continuation = self._parse_mapping(continuation_indent)
                    duplicate = item.keys() & continuation.keys()
                    if duplicate:
                        raise WorkflowYamlError(f"duplicate_key:{line.number}:{sorted(duplicate)[0]}")
                    item.update(continuation)
                result.append(item)
            else:
                result.append(self._parse_inline_scalar(payload, line.number))
        return result

    def _parse_mapping_entry(self, content: str, number: int, indent: int) -> tuple[str, Any]:
        match = _MAPPING_ENTRY.match(content)
        if match is None:
            raise WorkflowYamlError(f"invalid_mapping_entry:{number}")
        key = match.group("key")
        if key == "<<":
            raise WorkflowYamlError(f"yaml_merge_key_not_allowed:{number}")
        raw_value = match.group("value")
        if raw_value is None or raw_value == "":
            self._skip_ignorable()
            if self._index == len(self._lines) or self._lines[self._index].indent <= indent:
                return key, None
            return key, self._parse_block(self._lines[self._index].indent)
        block_match = _BLOCK_HEADER.fullmatch(raw_value)
        if block_match is not None:
            return key, self._parse_block_scalar(indent, block_match.group("style"))
        if raw_value.startswith(("&", "*", "!")):
            raise WorkflowYamlError(f"yaml_indirection_not_allowed:{number}")
        return key, self._parse_inline_scalar(raw_value, number)

    def _parse_block_scalar(self, parent_indent: int, style: str) -> str:
        raw_lines: list[str] = []
        content_indents: list[int] = []
        while self._index < len(self._lines):
            line = self._lines[self._index]
            if line.content and line.indent <= parent_indent:
                break
            if line.content:
                content_indents.append(line.indent)
            raw_lines.append(line.raw)
            self._index += 1
        if not content_indents:
            return ""
        content_indent = min(content_indents)
        values = [raw[content_indent:] if raw.strip() else "" for raw in raw_lines]
        if style == "|":
            return "\n".join(values)
        folded: list[str] = []
        for value in values:
            if not value:
                folded.append("\n")
            elif folded and folded[-1] != "\n":
                folded.append(" " + value)
            else:
                folded.append(value)
        return "".join(folded)

    @staticmethod
    def _parse_inline_scalar(value: str, number: int) -> str:
        if value.startswith('"'):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise WorkflowYamlError(f"invalid_double_quoted_scalar:{number}") from exc
            if not isinstance(parsed, str):
                raise WorkflowYamlError(f"non_string_scalar:{number}")
            return parsed
        if value.startswith("'"):
            if len(value) < 2 or not value.endswith("'"):
                raise WorkflowYamlError(f"invalid_single_quoted_scalar:{number}")
            return value[1:-1].replace("''", "'")
        return value


def load_workflow_yaml(source: str) -> dict[str, Any]:
    """Parse the structural YAML subset used by GitHub Actions workflows."""
    return _Parser(source).parse()


def workflow_run_scalars(source: str) -> list[str]:
    workflow = load_workflow_yaml(source)
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        raise WorkflowYamlError("jobs_not_mapping")
    runs: list[str] = []
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            raise WorkflowYamlError(f"job_not_mapping:{job_name}")
        steps = job.get("steps")
        if steps is None:
            continue
        if not isinstance(steps, list):
            raise WorkflowYamlError(f"steps_not_sequence:{job_name}")
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise WorkflowYamlError(f"step_not_mapping:{job_name}:{index}")
            if "run" not in step:
                continue
            run = step["run"]
            if not isinstance(run, str):
                raise WorkflowYamlError(f"run_not_string:{job_name}:{index}")
            runs.append(run)
    return runs
