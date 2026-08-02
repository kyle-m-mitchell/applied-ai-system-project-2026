"""Checksum and archive defenses used before untrusted data reaches the ETL."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
COPY_CHUNK_SIZE = 1024 * 1024


class ChecksumMismatchError(ValueError):
    """Raised when an artifact differs from its pinned SHA-256 digest."""


class UnsafeArchiveError(ValueError):
    """Raised before extracting a suspicious or unexpectedly large ZIP."""


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a large file in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(COPY_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Validate a pinned lowercase/uppercase SHA-256 and return the actual digest."""
    if not SHA256_PATTERN.fullmatch(expected):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if not secrets_compare(actual, expected.lower()):
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {Path(path).name}: expected {expected.lower()}, got {actual}"
        )
    return actual


def secrets_compare(left: str, right: str) -> bool:
    """Constant-time comparison kept local to avoid accepting partial digests."""
    return hmac.compare_digest(left, right)


def _safe_member_path(member: zipfile.ZipInfo) -> PurePosixPath:
    name = member.filename.replace("\\", "/")
    if "\x00" in name:
        raise UnsafeArchiveError("ZIP member contains a NUL byte")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise UnsafeArchiveError(f"unsafe ZIP member path: {member.filename!r}")
    if path.parts and ":" in path.parts[0]:
        raise UnsafeArchiveError(f"drive-qualified ZIP member path: {member.filename!r}")

    unix_mode = member.external_attr >> 16
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        raise UnsafeArchiveError(f"symbolic links are not allowed: {member.filename!r}")
    # Some ZIP writers store permissions (0600) without file-type bits. Treat
    # that as an ordinary file; reject only an explicitly encoded special type.
    if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
        raise UnsafeArchiveError(f"special files are not allowed: {member.filename!r}")
    if member.flag_bits & 0x1:
        raise UnsafeArchiveError(f"encrypted ZIP members are not supported: {member.filename!r}")
    return path


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    allowed_files: frozenset[str] | None = None,
    selected_files: frozenset[str] | None = None,
    max_members: int = 10_000,
    max_member_bytes: int = 4 * 1024**3,
    max_total_bytes: int = 8 * 1024**3,
    max_compression_ratio: float = 1_000.0,
) -> tuple[Path, ...]:
    """Validate and atomically extract a ZIP without traversal or ZIP-bomb writes.

    The destination must not already exist: callers build into a fresh staging
    directory and then explicitly choose what to retain. Members are streamed
    one at a time, CRC-checked by :mod:`zipfile`, and never extracted through
    ``ZipFile.extractall``.
    """
    archive = Path(archive_path)
    target = Path(destination)
    if expected_sha256 is not None:
        verify_sha256(archive, expected_sha256)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite extraction destination: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > max_members:
                raise UnsafeArchiveError(
                    f"ZIP contains {len(members)} members; maximum is {max_members}"
                )

            seen: set[PurePosixPath] = set()
            total = 0
            validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for member in members:
                relative = _safe_member_path(member)
                if relative in seen:
                    raise UnsafeArchiveError(f"duplicate ZIP member: {member.filename!r}")
                seen.add(relative)
                if allowed_files is not None and not member.is_dir():
                    normalized = relative.as_posix()
                    basename = relative.name
                    if normalized not in allowed_files and basename not in allowed_files:
                        raise UnsafeArchiveError(f"unexpected ZIP member: {member.filename!r}")
                if member.file_size > max_member_bytes:
                    raise UnsafeArchiveError(f"ZIP member is too large: {member.filename!r}")
                total += member.file_size
                if total > max_total_bytes:
                    raise UnsafeArchiveError("ZIP uncompressed size exceeds the configured limit")
                if member.file_size and not member.compress_size:
                    raise UnsafeArchiveError(f"invalid compressed size: {member.filename!r}")
                if member.compress_size:
                    ratio = member.file_size / member.compress_size
                    if ratio > max_compression_ratio:
                        raise UnsafeArchiveError(
                            f"ZIP member compression ratio is suspicious: {member.filename!r}"
                        )
                validated.append((member, relative))

            for member, relative in validated:
                if selected_files is not None and not member.is_dir():
                    normalized = relative.as_posix()
                    if normalized not in selected_files and relative.name not in selected_files:
                        continue
                output = staging.joinpath(*relative.parts)
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, output.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=COPY_CHUNK_SIZE)
                extracted.append(target.joinpath(*relative.parts))

        os.replace(staging, target)
        return tuple(extracted)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
