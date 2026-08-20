from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import zlib

import pytest

from scripts.candidate_artifact_manifest import retired_packaged_entries
from scripts.hard_cutover_policy import semantic_policy_categories


ROOT = Path(__file__).resolve().parents[1]
BINARY_ALLOWLIST_NAME = "hard-cutover-binary-allowlist.json"
BINARY_ALLOWLIST_SCHEMA = "vem-hard-cutover-binary-allowlist/v1"
BINARY_POLICIES = {
    "recorded-video-fixture": {
        "prefix": "fixtures/recorded-video/",
        "suffixes": {".mp4", ".png"},
        "reason": "Recorded acquisition fixture with repository provenance.",
    },
    "runtime-model": {
        "prefix": "models/",
        "suffixes": {".caffemodel", ".onnx"},
        "reason": "Digest-pinned production acquisition model.",
    },
}
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
    (ROOT / "fixtures/recorded-video/README.md").resolve(): {
        "sha256": "e2076983553408705fe4e5f17d2f252f9ef14b3ebf5d31f1690a26ea1aa56960",
        "occurrences": {"standalone-repository-url": 1},
    },
}

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
            rf"payload\[\s*[\"']mode[\"']\s*\]\s*=\s*"
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


def _load_binary_allowlist(
    root: Path,
    tracked: dict[str, tuple[Path, str]],
    violations: list[str],
) -> dict[str, dict[str, str]]:
    manifest_path = root / BINARY_ALLOWLIST_NAME
    manifest_tracked = tracked.get(BINARY_ALLOWLIST_NAME)
    if manifest_tracked is None or manifest_tracked[1] != "100644":
        violations.append(f"{manifest_path}: binary-allowlist-untracked")
        return {}
    try:
        raw_manifest = manifest_path.read_bytes()
        manifest = json.loads(raw_manifest.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        violations.append(f"{manifest_path}: binary-allowlist-invalid")
        return {}
    canonical = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if (
        not isinstance(manifest, dict)
        or raw_manifest != canonical
        or set(manifest) != {"entries", "schemaVersion"}
    ):
        violations.append(f"{manifest_path}: binary-allowlist-invalid")
        return {}
    entries = manifest.get("entries")
    if manifest.get("schemaVersion") != BINARY_ALLOWLIST_SCHEMA or not isinstance(
        entries, list
    ):
        violations.append(f"{manifest_path}: binary-allowlist-invalid")
        return {}
    approved = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "category",
            "gitMode",
            "path",
            "reason",
            "sha256",
        }:
            violations.append(f"{manifest_path}: binary-allowlist-invalid")
            return {}
        if not all(isinstance(value, str) for value in entry.values()):
            violations.append(f"{manifest_path}: binary-allowlist-invalid")
            return {}
        path = entry["path"]
        policy = BINARY_POLICIES.get(entry["category"])
        if (
            path in approved
            or path.startswith("/")
            or "\\" in path
            or ".." in Path(path).parts
            or entry["gitMode"] != "100644"
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or policy is None
            or not path.startswith(policy["prefix"])
            or Path(path).suffix.lower() not in policy["suffixes"]
            or entry["reason"] != policy["reason"]
        ):
            violations.append(f"{manifest_path}: binary-allowlist-invalid")
            return {}
        approved[path] = entry
    if list(approved) != sorted(approved):
        violations.append(f"{manifest_path}: binary-allowlist-invalid")
        return {}
    return approved


EXECUTABLE_MAGICS = (
    b"MZ",
    b"\x7fELF",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
)


def _valid_png(data: bytes) -> bool:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    offset = 8
    saw_header = False
    saw_image_data = False
    image_data_ended = False
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            return False
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length : chunk_end], "big")
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            return False
        if not saw_header:
            if chunk_type != b"IHDR" or length != 13:
                return False
            width, height = struct.unpack(">II", chunk_data[:8])
            if width == 0 or height == 0:
                return False
            saw_header = True
        elif chunk_type == b"IHDR":
            return False
        if chunk_type == b"IDAT":
            if length == 0 or image_data_ended:
                return False
            saw_image_data = True
        elif saw_image_data and chunk_type != b"IEND":
            image_data_ended = True
        if chunk_type == b"IEND":
            return (
                length == 0
                and saw_header
                and saw_image_data
                and chunk_end == len(data)
            )
        offset = chunk_end
    return False


def _valid_jpeg(data: bytes) -> bool:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return False
    offset = 2
    saw_frame = False
    saw_scan = False
    saw_scan_data = False
    while offset < len(data):
        if data[offset] != 0xFF:
            return False
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            return False
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            return saw_frame and saw_scan and saw_scan_data and offset == len(data)
        if marker in {0x00, 0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            return False
        if offset + 2 > len(data):
            return False
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            return False
        segment_start = offset + 2
        segment_end = offset + segment_length
        is_start_of_frame = (
            0xC0 <= marker <= 0xC3
            or 0xC5 <= marker <= 0xC7
            or 0xC9 <= marker <= 0xCB
            or 0xCD <= marker <= 0xCF
        )
        if is_start_of_frame:
            if saw_frame or segment_length < 8:
                return False
            component_count = data[segment_start + 5]
            if (
                int.from_bytes(data[segment_start + 1 : segment_start + 3], "big")
                == 0
                or int.from_bytes(
                    data[segment_start + 3 : segment_start + 5], "big"
                )
                == 0
                or component_count == 0
                or segment_length != 8 + 3 * component_count
            ):
                return False
            saw_frame = True
        if marker == 0xDA:
            component_count = data[segment_start]
            if (
                not saw_frame
                or saw_scan
                or component_count == 0
                or segment_length != 6 + 2 * component_count
            ):
                return False
            saw_scan = True
            offset = segment_end
            while offset < len(data):
                if data[offset] != 0xFF:
                    saw_scan_data = True
                    offset += 1
                    continue
                marker_offset = offset + 1
                while marker_offset < len(data) and data[marker_offset] == 0xFF:
                    marker_offset += 1
                if marker_offset >= len(data):
                    return False
                scan_marker = data[marker_offset]
                if scan_marker == 0x00:
                    saw_scan_data = True
                    offset = marker_offset + 1
                    continue
                if 0xD0 <= scan_marker <= 0xD7:
                    offset = marker_offset + 1
                    continue
                if scan_marker == 0xD9:
                    return saw_scan_data and marker_offset + 1 == len(data)
                return False
            return False
        offset = segment_end
    return False


def _valid_ico(data: bytes) -> bool:
    if len(data) < 6:
        return False
    reserved, image_type, count = struct.unpack_from("<HHH", data)
    if (
        reserved != 0
        or image_type != 1
        or count == 0
        or len(data) < 6 + count * 16
    ):
        return False
    for index in range(count):
        offset = 6 + index * 16
        image_size, image_offset = struct.unpack_from("<II", data, offset + 8)
        if image_size == 0 or image_offset < 6 + count * 16:
            return False
        if image_offset + image_size > len(data):
            return False
        image = data[image_offset : image_offset + image_size]
        if not _valid_png(image) and not _valid_dib(image):
            return False
    return True


def _valid_dib(data: bytes) -> bool:
    if len(data) < 12:
        return False
    header_size = int.from_bytes(data[:4], "little")
    if header_size == 12:
        width, height, planes, bit_count = struct.unpack_from("<HHHH", data, 4)
        return (
            width > 0
            and height > 0
            and planes == 1
            and bit_count in {1, 4, 8, 16, 24, 32}
        )
    if header_size not in {40, 52, 56, 108, 124} or len(data) < header_size:
        return False
    width, height, planes, bit_count = struct.unpack_from("<iiHH", data, 4)
    return (
        width != 0
        and height != 0
        and planes == 1
        and bit_count in {1, 4, 8, 16, 24, 32}
    )


def _valid_wav(data: bytes) -> bool:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        return False
    offset = 12
    saw_format = False
    saw_data = False
    while offset + 8 <= len(data):
        chunk_type = data[offset : offset + 4]
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        chunk_end = offset + 8 + length
        padded_end = chunk_end + (length & 1)
        if padded_end > len(data):
            return False
        if chunk_type == b"fmt ":
            if saw_format or length < 16:
                return False
            saw_format = True
        elif chunk_type == b"data":
            if not saw_format or saw_data or length == 0:
                return False
            saw_data = True
        offset = padded_end
    return offset == len(data) and saw_format and saw_data


def _valid_mp3(data: bytes) -> bool:
    if data.startswith(b"ID3"):
        if len(data) < 10 or any(byte & 0x80 for byte in data[6:10]):
            return False
        tag_size = sum(
            byte << shift
            for byte, shift in zip(data[6:10], (21, 14, 7, 0), strict=True)
        )
        return tag_size + 10 <= len(data)
    if len(data) < 4 or data[0] != 0xFF or data[1] & 0xE0 != 0xE0:
        return False
    version = (data[1] >> 3) & 0x03
    layer = (data[1] >> 1) & 0x03
    bitrate = (data[2] >> 4) & 0x0F
    sample_rate = (data[2] >> 2) & 0x03
    return version != 1 and layer != 0 and bitrate not in {0, 15} and sample_rate != 3


def _valid_ogg(data: bytes) -> bool:
    if len(data) < 27 or data[:4] != b"OggS" or data[4] != 0:
        return False
    segment_count = data[26]
    if len(data) < 27 + segment_count:
        return False
    payload_size = sum(data[27 : 27 + segment_count])
    return 27 + segment_count + payload_size <= len(data)


def _valid_mp4(data: bytes) -> bool:
    offset = 0
    box_count = 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > len(data):
                return False
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = len(data) - offset
        if size < header_size or offset + size > len(data):
            return False
        if box_count == 0:
            if box_type != b"ftyp" or size < header_size + 8:
                return False
            compatible_size = size - header_size - 8
            if compatible_size % 4:
                return False
        box_count += 1
        offset += size
    return box_count > 0 and offset == len(data)


def _valid_webp(data: bytes) -> bool:
    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return False
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        return False
    offset = 12
    saw_image = False
    while offset + 8 <= len(data):
        chunk_type = data[offset : offset + 4]
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        chunk_end = offset + 8 + length
        if chunk_end > len(data):
            return False
        saw_image |= chunk_type in {b"VP8 ", b"VP8L", b"VP8X"}
        offset = chunk_end + (length & 1)
    return saw_image and offset == len(data)


def _read_varint(data: bytes, offset: int) -> tuple[int, int] | None:
    value = 0
    for index in range(10):
        if offset + index >= len(data):
            return None
        byte = data[offset + index]
        value |= (byte & 0x7F) << (index * 7)
        if byte & 0x80 == 0:
            return value, offset + index + 1
    return None


def _valid_onnx(data: bytes) -> bool:
    if not data or data[0] != 0x08:
        return False
    decoded = _read_varint(data, 1)
    return decoded is not None and 0 < decoded[0] < 1_000


def _valid_caffemodel(data: bytes) -> bool:
    if not data or data[0] != 0x0A:
        return False
    decoded = _read_varint(data, 1)
    if decoded is None:
        return False
    length, offset = decoded
    name = data[offset : offset + length]
    return (
        0 < length <= 256
        and len(name) == length
        and all(32 <= byte < 127 for byte in name)
    )


def _binary_format_is_valid(path: str, data: bytes) -> bool:
    if data.startswith(b"#!") or any(
        data.startswith(magic) for magic in EXECUTABLE_MAGICS
    ):
        return False
    validators = {
        ".caffemodel": _valid_caffemodel,
        ".ico": _valid_ico,
        ".jpeg": _valid_jpeg,
        ".jpg": _valid_jpeg,
        ".mp3": _valid_mp3,
        ".mp4": _valid_mp4,
        ".ogg": _valid_ogg,
        ".onnx": _valid_onnx,
        ".png": _valid_png,
        ".wav": _valid_wav,
        ".webp": _valid_webp,
    }
    validator = validators.get(Path(path).suffix.lower())
    return validator is not None and validator(data)


def find_violations(
    root: Path,
) -> list[str]:
    root = root.resolve()
    violations = []
    tracked_entries = _tracked_entries(root)
    tracked = {
        path.relative_to(root).as_posix(): (path, git_mode)
        for path, git_mode in tracked_entries
    }
    approved_binary = _load_binary_allowlist(root, tracked, violations)
    actual_binary = {}
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
            or retired_packaged_entries(
                [relative_path], include_historical_generic_modules=False
            )
        ):
            violations.append(f"{path}: retired-try-on-path")
            continue
        if relative_path in approved_binary:
            actual_binary[relative_path] = {
                "gitMode": git_mode,
                "sha256": hashlib.sha256(raw_source).hexdigest(),
            }
            if not _binary_format_is_valid(relative_path, raw_source):
                violations.append(f"{path}: binary-format-invalid")
            continue
        if b"\0" in raw_source:
            actual_binary[relative_path] = {
                "gitMode": git_mode,
                "sha256": hashlib.sha256(raw_source).hexdigest(),
            }
            continue
        try:
            source = raw_source.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            actual_binary[relative_path] = {
                "gitMode": git_mode,
                "sha256": hashlib.sha256(raw_source).hexdigest(),
            }
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
        for category, pattern in SINGLE_PATH_CONTENT_RULES:
            matches = list(pattern.finditer(source))
            if not matches:
                continue
            if _matches_are_exact_audited_lines(path, category, source, matches):
                continue
            violations.append(f"{path}: {category}")
        for category in sorted(semantic_policy_categories(relative_path, source)):
            violations.append(f"{path}: {category}")
    for path in sorted(actual_binary.keys() - approved_binary.keys()):
        violations.append(f"{root / path}: binary-unapproved")
    for path in sorted(approved_binary.keys() - actual_binary.keys()):
        violations.append(f"{root / path}: binary-allowlist-entry-missing")
    for path in sorted(actual_binary.keys() & approved_binary.keys()):
        actual = actual_binary[path]
        expected = approved_binary[path]
        if actual["gitMode"] != expected["gitMode"] or actual["sha256"] != expected["sha256"]:
            violations.append(f"{root / path}: binary-identity-mismatch")
    return violations


def _init_guard_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / BINARY_ALLOWLIST_NAME).write_text(
        '{"entries":[],"schemaVersion":"vem-hard-cutover-binary-allowlist/v1"}\n',
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["git", "add", BINARY_ALLOWLIST_NAME], cwd=root, check=True)


def _write_binary_allowlist(root: Path, entries: list[dict[str, str]]) -> None:
    manifest = {"entries": entries, "schemaVersion": BINARY_ALLOWLIST_SCHEMA}
    (root / BINARY_ALLOWLIST_NAME).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(["git", "add", BINARY_ALLOWLIST_NAME], cwd=root, check=True)


def _recorded_fixture_entry(path: str, payload: bytes) -> dict[str, str]:
    return {
        "category": "recorded-video-fixture",
        "gitMode": "100644",
        "path": path,
        "reason": "Recorded acquisition fixture with repository provenance.",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


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


def test_binary_allowlist_binds_exact_binary_set_and_identity(tmp_path):
    approved_root = tmp_path / "approved"
    approved_root.mkdir()
    _init_guard_repo(approved_root)
    relative_path = "fixtures/recorded-video/approved.png"
    payload = (
        ROOT / "fixtures/recorded-video/sources/person-man-front.png"
    ).read_bytes()
    binary = approved_root / relative_path
    binary.parent.mkdir(parents=True)
    binary.write_bytes(payload)
    _write_binary_allowlist(
        approved_root,
        [_recorded_fixture_entry(relative_path, payload)],
    )
    subprocess.run(["git", "add", relative_path], cwd=approved_root, check=True)

    assert find_violations(approved_root) == []

    binary.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0x01]))
    assert f"{binary}: binary-identity-mismatch" in find_violations(approved_root)

    deleted_root = tmp_path / "deleted"
    deleted_root.mkdir()
    _init_guard_repo(deleted_root)
    deleted = deleted_root / relative_path
    deleted.parent.mkdir(parents=True)
    deleted.write_bytes(payload)
    _write_binary_allowlist(
        deleted_root,
        [_recorded_fixture_entry(relative_path, payload)],
    )
    subprocess.run(["git", "add", relative_path], cwd=deleted_root, check=True)
    subprocess.run(
        ["git", "rm", "-f", "--", relative_path],
        cwd=deleted_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    deleted_violations = find_violations(deleted_root)
    assert f"{deleted}: binary-allowlist-entry-missing" in deleted_violations


def test_binary_allowlist_rejects_new_executable_and_manifest_mutations(tmp_path):
    executable_root = tmp_path / "executable"
    executable_root.mkdir()
    _init_guard_repo(executable_root)
    executable = executable_root / "standalone-service.exe"
    executable.write_bytes(b"MZ\0untrusted")
    subprocess.run(["git", "add", executable.name], cwd=executable_root, check=True)

    assert f"{executable}: binary-unapproved" in find_violations(executable_root)
    _write_binary_allowlist(
        executable_root,
        [
            {
                "category": "recorded-video-fixture",
                "gitMode": "100644",
                "path": executable.name,
                "reason": "Recorded acquisition fixture with repository provenance.",
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            }
        ],
    )
    assert any(
        violation.endswith(": binary-allowlist-invalid")
        for violation in find_violations(executable_root)
    )

    relative_paths = (
        "fixtures/recorded-video/a.png",
        "fixtures/recorded-video/b.png",
    )
    payloads = (b"a\0", b"b\0")
    valid_entries = [
        _recorded_fixture_entry(path, payload)
        for path, payload in zip(relative_paths, payloads, strict=True)
    ]
    mutations = (
        {"entries": list(reversed(valid_entries)), "schemaVersion": BINARY_ALLOWLIST_SCHEMA},
        {"entries": [valid_entries[0], valid_entries[0]], "schemaVersion": BINARY_ALLOWLIST_SCHEMA},
        {"entries": valid_entries, "schemaVersion": BINARY_ALLOWLIST_SCHEMA, "extra": True},
        {
            "entries": [{**valid_entries[0], "extra": "field"}],
            "schemaVersion": BINARY_ALLOWLIST_SCHEMA,
        },
    )
    for index, manifest in enumerate(mutations):
        root = tmp_path / f"manifest-{index}"
        root.mkdir()
        _init_guard_repo(root)
        for path, payload in zip(relative_paths, payloads, strict=True):
            binary = root / path
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(payload)
        (root / BINARY_ALLOWLIST_NAME).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.run(
            ["git", "add", BINARY_ALLOWLIST_NAME, *relative_paths],
            cwd=root,
            check=True,
        )

        assert any(
            violation.endswith(": binary-allowlist-invalid")
            for violation in find_violations(root)
        )

    noncanonical_root = tmp_path / "manifest-noncanonical"
    noncanonical_root.mkdir()
    _init_guard_repo(noncanonical_root)
    for path, payload in zip(relative_paths, payloads, strict=True):
        binary = noncanonical_root / path
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(payload)
    (noncanonical_root / BINARY_ALLOWLIST_NAME).write_text(
        json.dumps(
            {"entries": valid_entries, "schemaVersion": BINARY_ALLOWLIST_SCHEMA},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(
        ["git", "add", BINARY_ALLOWLIST_NAME, *relative_paths],
        cwd=noncanonical_root,
        check=True,
    )
    assert any(
        violation.endswith(": binary-allowlist-invalid")
        for violation in find_violations(noncanonical_root)
    )
    (noncanonical_root / BINARY_ALLOWLIST_NAME).write_text(
        "[{}]\n",
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(
        ["git", "add", BINARY_ALLOWLIST_NAME],
        cwd=noncanonical_root,
        check=True,
    )
    assert any(
        violation.endswith(": binary-allowlist-invalid")
        for violation in find_violations(noncanonical_root)
    )


def test_binary_allowlist_rejects_executable_magic_mismatch_and_truncation(tmp_path):
    source_png = (
        ROOT / "fixtures/recorded-video/sources/person-man-front.png"
    ).read_bytes()
    png_parts = [source_png[:8]]
    offset = 8
    while offset < len(source_png):
        length = int.from_bytes(source_png[offset : offset + 4], "big")
        chunk_end = offset + 12 + length
        if source_png[offset + 4 : offset + 8] != b"IDAT":
            png_parts.append(source_png[offset:chunk_end])
        offset = chunk_end
    corrupt_crc = bytearray(source_png)
    corrupt_crc[29] ^= 0x01
    disguised_payloads = {
        "pe": b"MZ\0pretend-png",
        "elf": b"\x7fELF\0pretend-png",
        "mach-o": b"\xfe\xed\xfa\xcf\0pretend-png",
        "shebang": b"#!/bin/sh\nexit 0\n",
        "extension-mismatch": b"\xff\xd8not-a-png\xff\xd9",
        "truncated": b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR",
        "no-idat": b"".join(png_parts),
        "corrupt-crc": bytes(corrupt_crc),
    }
    for name, payload in disguised_payloads.items():
        root = tmp_path / name
        root.mkdir()
        _init_guard_repo(root)
        relative_path = "fixtures/recorded-video/disguised.png"
        disguised = root / relative_path
        disguised.parent.mkdir(parents=True)
        disguised.write_bytes(payload)
        _write_binary_allowlist(
            root,
            [_recorded_fixture_entry(relative_path, payload)],
        )
        subprocess.run(["git", "add", relative_path], cwd=root, check=True)

        assert f"{disguised}: binary-format-invalid" in find_violations(root)


def test_jpeg_format_validator_requires_eoi_at_end():
    jpeg = (
        b"\xff\xd8"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
        b"\x01\xff\xd9"
    )

    assert _valid_jpeg(jpeg)
    assert not _valid_jpeg(jpeg + b"MZ")
    assert not _valid_jpeg(b"\xff\xd8MZ\xff\xd9")


def test_binary_allowlist_assets_decode_with_production_opencv_loaders():
    import cv2

    manifest = json.loads((ROOT / BINARY_ALLOWLIST_NAME).read_text("utf-8"))
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            capture = cv2.VideoCapture(str(path))
            try:
                opened, frame = capture.read()
                assert opened and frame is not None, path
            finally:
                capture.release()
        elif suffix == ".png":
            assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None, path
        elif suffix == ".onnx":
            assert not cv2.dnn.readNetFromONNX(str(path)).empty(), path
        elif suffix == ".caffemodel":
            definition = path.with_name(
                path.name.replace("_net.caffemodel", "_deploy.prototxt")
            )
            assert not cv2.dnn.readNetFromCaffe(
                str(definition),
                str(path),
            ).empty(), path
        else:
            raise AssertionError(f"missing production decoder for {path}")
