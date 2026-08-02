"""Verified full-catalog resolution with an honest bundled-lite fallback."""

from __future__ import annotations

import hashlib
import gzip
import os
import secrets
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Callable, Protocol
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from src.etl.integrity import ChecksumMismatchError, SHA256_PATTERN, sha256_file
from src.etl.manifest import CatalogManifest
from src.fma_store import CatalogDatabaseError, Fts5UnavailableError, validate_fma_database


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 2 * 1024**3
DEFAULT_MAX_DATABASE_BYTES = 4 * 1024**3


class CatalogUnavailableError(RuntimeError):
    """No verified full or lite catalog can be opened."""


class _Response(Protocol):
    headers: object

    def read(self, amount: int = -1) -> bytes: ...
    def __enter__(self) -> "_Response": ...
    def __exit__(self, *args: object) -> None: ...


@dataclass(frozen=True, slots=True)
class ArtifactCandidate:
    database_path: Path
    manifest_path: Path
    source: str


@dataclass(frozen=True, slots=True)
class ResolvedCatalogArtifact:
    database_path: Path
    manifest_path: Path
    manifest: CatalogManifest
    source: str
    warnings: tuple[str, ...] = ()

    @property
    def is_fallback(self) -> bool:
        return self.manifest.edition == "lite"


def validate_catalog_artifact(candidate: ArtifactCandidate) -> CatalogManifest:
    manifest = CatalogManifest.read(candidate.manifest_path)
    actual = sha256_file(candidate.database_path)
    if actual != manifest.artifact_sha256:
        raise ChecksumMismatchError(
            f"{candidate.source} database checksum does not match its manifest"
        )
    if (
        manifest.artifact_bytes is not None
        and candidate.database_path.stat().st_size != manifest.artifact_bytes
    ):
        raise ChecksumMismatchError(f"{candidate.source} database size does not match its manifest")
    count = validate_fma_database(candidate.database_path, require_fts=True)
    if count != manifest.accepted_count:
        raise CatalogDatabaseError(
            f"{candidate.source} database count does not match its manifest"
        )
    return manifest


def _default_opener(request: Request, timeout: float) -> _Response:
    return urlopen(request, timeout=timeout)  # type: ignore[return-value]


def download_verified_artifact(
    url: str,
    destination: str | Path,
    expected_sha256: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    opener: Callable[[Request, float], _Response] = _default_opener,
) -> Path:
    """Stream a pinned HTTPS asset to a same-directory file, then atomically swap."""
    parsed = urlsplit(url)
    if parsed.scheme.casefold() != "https" or not parsed.netloc:
        raise ValueError("catalog release URL must be an absolute HTTPS URL")
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("release checksum must be a full SHA-256 digest")
    if max_bytes < 1:
        raise ValueError("maximum download size must be positive")

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".download", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    received = 0
    try:
        request = Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "Cadence-FMA-Catalog/1.0",
            },
        )
        with os.fdopen(descriptor, "wb") as sink, opener(request, timeout) as response:
            raw_length = getattr(response.headers, "get", lambda _name: None)(
                "Content-Length"
            )
            if raw_length is not None:
                try:
                    content_length = int(raw_length)
                except (TypeError, ValueError) as exc:
                    raise ValueError("release response has an invalid Content-Length") from exc
                if content_length < 0 or content_length > max_bytes:
                    raise ValueError("release artifact exceeds the configured size limit")
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                received += len(chunk)
                if received > max_bytes:
                    raise ValueError("release artifact exceeds the configured size limit")
                digest.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if digest.hexdigest() != expected_sha256.lower():
            raise ChecksumMismatchError("downloaded release artifact failed SHA-256 verification")
        os.replace(temporary, target)
        target.chmod(0o644)
        return target
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def package_catalog_gzip(
    database_path: str | Path,
    manifest_path: str | Path,
    distribution_path: str | Path,
    *,
    release_manifest_path: str | Path | None = None,
    compresslevel: int = 9,
) -> CatalogManifest:
    """Create a deterministic ``mtime=0`` gzip and its distribution manifest."""
    database = Path(database_path)
    manifest = CatalogManifest.read(manifest_path)
    if sha256_file(database) != manifest.artifact_sha256:
        raise ChecksumMismatchError("database does not match the source manifest")
    if not 0 <= compresslevel <= 9:
        raise ValueError("gzip compresslevel must be between 0 and 9")
    target = Path(distribution_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_output, database.open("rb") as source:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=compresslevel,
                mtime=0,
            ) as compressed:
                while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
                    compressed.write(chunk)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary, target)
        target.chmod(0o644)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    release_manifest = replace(
        manifest,
        artifact_bytes=database.stat().st_size,
        distribution_sha256=sha256_file(target),
        distribution_compression="gzip",
        distribution_bytes=target.stat().st_size,
    )
    release_manifest.write(release_manifest_path or manifest_path)
    return release_manifest


def download_verified_gzip_catalog(
    url: str,
    destination: str | Path,
    *,
    distribution_sha256: str,
    artifact_sha256: str,
    timeout: float = 30.0,
    max_download_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_database_bytes: int = DEFAULT_MAX_DATABASE_BYTES,
    opener: Callable[[Request, float], _Response] = _default_opener,
) -> Path:
    """Download a pinned gzip, bounded-decompress it, verify DB SHA, atomically install."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    compressed = target.parent / f".{target.name}-{secrets.token_hex(8)}.gz"
    descriptor, output_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".inflate", dir=target.parent
    )
    output = Path(output_name)
    try:
        download_verified_artifact(
            url,
            compressed,
            distribution_sha256,
            timeout=timeout,
            max_bytes=max_download_bytes,
            opener=opener,
        )
        digest = hashlib.sha256()
        expanded = 0
        with gzip.open(compressed, "rb") as source, os.fdopen(descriptor, "wb") as sink:
            while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
                expanded += len(chunk)
                if expanded > max_database_bytes:
                    raise ValueError("expanded catalog exceeds the configured size limit")
                digest.update(chunk)
                sink.write(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if digest.hexdigest() != artifact_sha256.lower():
            raise ChecksumMismatchError("expanded catalog failed database SHA-256 verification")
        os.replace(output, target)
        target.chmod(0o644)
        return target
    except (gzip.BadGzipFile, EOFError) as exc:
        raise ValueError("release asset is not a valid complete gzip stream") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        compressed.unlink(missing_ok=True)
        output.unlink(missing_ok=True)


class CatalogArtifactResolver:
    """Resolve local full → verified release cache → committed FMA-lite."""

    def __init__(
        self,
        *,
        local_full: ArtifactCandidate | None,
        bundled_lite: ArtifactCandidate,
        release_url: str | None = None,
        release_manifest_path: str | Path | None = None,
        release_cache_path: str | Path | None = None,
        timeout: float = 30.0,
        max_download_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_database_bytes: int = DEFAULT_MAX_DATABASE_BYTES,
        opener: Callable[[Request, float], _Response] = _default_opener,
    ) -> None:
        release_values = (release_url, release_manifest_path, release_cache_path)
        if any(value is not None for value in release_values) and not all(
            value is not None for value in release_values
        ):
            raise ValueError(
                "release_url, release_manifest_path, and release_cache_path must be supplied together"
            )
        self.local_full = local_full
        self.bundled_lite = bundled_lite
        self.release_url = release_url
        self.release_manifest_path = (
            Path(release_manifest_path) if release_manifest_path is not None else None
        )
        self.release_cache_path = (
            Path(release_cache_path) if release_cache_path is not None else None
        )
        self.timeout = timeout
        self.max_download_bytes = max_download_bytes
        self.max_database_bytes = max_database_bytes
        self.opener = opener

    @staticmethod
    def _try_candidate(
        candidate: ArtifactCandidate, warnings: list[str]
    ) -> ResolvedCatalogArtifact | None:
        try:
            manifest = validate_catalog_artifact(candidate)
        except (
            OSError,
            ValueError,
            CatalogDatabaseError,
            Fts5UnavailableError,
            ChecksumMismatchError,
        ) as exc:
            warnings.append(f"{candidate.source} unavailable: {type(exc).__name__}")
            return None
        return ResolvedCatalogArtifact(
            database_path=candidate.database_path,
            manifest_path=candidate.manifest_path,
            manifest=manifest,
            source=candidate.source,
            warnings=tuple(warnings),
        )

    def resolve(self) -> ResolvedCatalogArtifact:
        warnings: list[str] = []
        if self.local_full is not None:
            resolved = self._try_candidate(self.local_full, warnings)
            if resolved is not None:
                return resolved

        if self.release_url is not None:
            assert self.release_manifest_path is not None
            assert self.release_cache_path is not None
            try:
                release_manifest = CatalogManifest.read(self.release_manifest_path)
                if (
                    release_manifest.distribution_compression != "gzip"
                    or release_manifest.distribution_sha256 is None
                ):
                    raise ValueError("release manifest does not describe a gzip distribution")
                release_candidate = ArtifactCandidate(
                    database_path=self.release_cache_path,
                    manifest_path=self.release_manifest_path,
                    source="release-cache",
                )
                if self.release_cache_path.exists():
                    cached = self._try_candidate(release_candidate, warnings)
                    if cached is not None:
                        return cached
                download_verified_gzip_catalog(
                    self.release_url,
                    self.release_cache_path,
                    distribution_sha256=release_manifest.distribution_sha256,
                    artifact_sha256=release_manifest.artifact_sha256,
                    timeout=self.timeout,
                    max_download_bytes=self.max_download_bytes,
                    max_database_bytes=self.max_database_bytes,
                    opener=self.opener,
                )
                downloaded = self._try_candidate(release_candidate, warnings)
                if downloaded is not None:
                    return downloaded
            except (OSError, ValueError, ChecksumMismatchError) as exc:
                warnings.append(f"release download unavailable: {type(exc).__name__}")

        fallback = self._try_candidate(self.bundled_lite, warnings)
        if fallback is None:
            detail = "; ".join(warnings) or "no candidates were configured"
            raise CatalogUnavailableError(f"no verified FMA catalog is available ({detail})")
        if fallback.manifest.edition != "lite":
            raise CatalogUnavailableError("the bundled fallback manifest is not edition 'lite'")
        return fallback
