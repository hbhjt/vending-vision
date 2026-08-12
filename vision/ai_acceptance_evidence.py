"""Private installed-acceptance sink for completed AI regional evidence."""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
from collections.abc import Callable
from pathlib import Path

_ATTEMPT_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_MAX_SIDECAR_BYTES = 512 * 1024


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _acceptance_root() -> Path | None:
    raw = os.environ.get("VEM_AI_ACCEPTANCE_EVIDENCE_ROOT", "")
    if not raw:
        return None
    root = Path(raw)
    if not root.is_absolute():
        raise RuntimeError("ai_acceptance_evidence_root_invalid")
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("ai_acceptance_evidence_root_invalid")
    return root.resolve(strict=True)


def _fsync_directory(root: Path) -> None:
    """Persist a directory entry where the platform exposes directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError:
        # Windows does not expose POSIX directory descriptors.  The file data
        # is still flushed before the non-overwriting hard-link publish.
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _claim_empty_invocation_root(root: Path, destination: Path) -> Path:
    """Reserve one empty, test-owned root without deleting another publisher's data."""
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("ai_acceptance_evidence_exists")
    if any(root.iterdir()):
        raise RuntimeError("ai_acceptance_evidence_root_not_empty")
    claim = root / ".ai-regional-evidence-publishing"
    try:
        claim.mkdir()
    except FileExistsError as exc:
        raise RuntimeError("ai_acceptance_evidence_root_not_empty") from exc
    return claim


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_dev, info.st_ino


def _unlink_created_file(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is not None and _identity(path) == identity:
        path.unlink()


def _raise_with_cleanup(
    primary: BaseException | None, cleanup_errors: list[BaseException]
) -> None:
    if primary is None and not cleanup_errors:
        return
    if primary is not None and not cleanup_errors:
        raise primary
    errors = ([primary] if primary is not None else []) + cleanup_errors
    raise ExceptionGroup("AI acceptance evidence publication failed", errors)


def publish_completed_ai_regional_evidence(attempt_id: str, content: bytes) -> Path | None:
    """Publish one already-committed sidecar; disabled unless an owner sets a root."""
    root = _acceptance_root()
    if root is None:
        return None
    if not _ATTEMPT_ID.fullmatch(attempt_id) or not 0 < len(content) <= _MAX_SIDECAR_BYTES:
        raise RuntimeError("ai_acceptance_evidence_invalid")
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("ai_acceptance_evidence_invalid") from exc
    if _canonical(value) != content:
        raise RuntimeError("ai_acceptance_evidence_noncanonical")
    destination = root / f"{attempt_id}.regional-evidence.json"
    if destination.parent != root:
        raise RuntimeError("ai_acceptance_evidence_path_invalid")
    claim = _claim_empty_invocation_root(root, destination)
    temporary = root / f".{attempt_id}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    fd = None
    temporary_identity = None
    published_identity = None
    primary = None
    cleanup_errors: list[BaseException] = []
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_identity = _identity(temporary)
        if temporary_identity is None:
            raise RuntimeError("ai_acceptance_evidence_temporary_invalid")
        os.link(temporary, destination)
        published_identity = _identity(destination)
        if published_identity is None:
            raise RuntimeError("ai_acceptance_evidence_publish_invalid")
        _fsync_directory(root)
        temporary.unlink()
    except FileExistsError as exc:
        primary = RuntimeError("ai_acceptance_evidence_exists")
        primary.__cause__ = exc
    except BaseException as exc:
        primary = exc
    finally:
        def cleanup(action: Callable[[], None]) -> None:
            try:
                action()
            except FileNotFoundError:
                # A prior cleanup may already have removed our own entry.
                return
            except BaseException as exc:
                cleanup_errors.append(exc)

        if fd is not None:
            cleanup(lambda: os.close(fd))
        cleanup(lambda: _unlink_created_file(temporary, temporary_identity))
        cleanup(claim.rmdir)
        if primary is None and cleanup_errors:
            primary = cleanup_errors.pop(0)
        if primary is not None:
            cleanup(lambda: _unlink_created_file(destination, published_identity))
            cleanup(lambda: _unlink_created_file(temporary, temporary_identity))
            cleanup(claim.rmdir)
        try:
            _fsync_directory(root)
        except BaseException as exc:
            if primary is None:
                primary = exc
                cleanup(lambda: _unlink_created_file(destination, published_identity))
                cleanup(lambda: _unlink_created_file(temporary, temporary_identity))
                cleanup(claim.rmdir)
                try:
                    _fsync_directory(root)
                except BaseException as retry_error:
                    cleanup_errors.append(retry_error)
            else:
                cleanup_errors.append(exc)
        _raise_with_cleanup(primary, cleanup_errors)
    return destination
