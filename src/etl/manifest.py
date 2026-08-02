"""Canonical build manifests for reproducible FMA catalog artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.etl.integrity import SHA256_PATTERN, sha256_file


MANIFEST_VERSION = "1"


def canonical_json_bytes(value: object) -> bytes:
    """Encode stable UTF-8 JSON used for hashes and checked-in evidence."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class CatalogManifest:
    """Facts needed to identify, verify, and truthfully describe one catalog."""

    catalog_id: str
    artifact_id: str
    edition: str
    schema_version: str
    etl_version: str
    artifact_sha256: str
    accepted_count: int
    quarantined_count: int
    artifact_bytes: int | None = None
    distribution_sha256: str | None = None
    distribution_compression: str | None = None
    distribution_bytes: int | None = None
    source_sha256: Mapping[str, str] = field(default_factory=dict)
    field_coverage: Mapping[str, float] = field(default_factory=dict)
    licenses: tuple[str, ...] = ()
    attribution: str = "Free Music Archive (FMA) metadata"
    supported_filters: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ()
    retrieval_methods: tuple[str, ...] = ("sqlite_fts5", "structured_sql")
    context_guides: bool = False
    research: bool = True
    calibration_status: str = "experimental"

    def __post_init__(self) -> None:
        if self.catalog_id != "fma":
            raise ValueError("FMA manifest catalog_id must be 'fma'")
        if self.edition not in {"full", "lite"}:
            raise ValueError("manifest edition must be 'full' or 'lite'")
        if not SHA256_PATTERN.fullmatch(self.artifact_sha256):
            raise ValueError("manifest artifact_sha256 is invalid")
        distribution_values = (
            self.distribution_sha256,
            self.distribution_compression,
            self.distribution_bytes,
        )
        if any(value is not None for value in distribution_values) and not all(
            value is not None for value in distribution_values
        ):
            raise ValueError("distribution checksum, compression, and size must be set together")
        if self.distribution_sha256 is not None:
            if not SHA256_PATTERN.fullmatch(self.distribution_sha256):
                raise ValueError("manifest distribution_sha256 is invalid")
            if self.distribution_compression != "gzip":
                raise ValueError("the only supported catalog distribution compression is gzip")
            if self.distribution_bytes is None or self.distribution_bytes < 1:
                raise ValueError("manifest distribution size must be positive")
        if self.artifact_bytes is not None and self.artifact_bytes < 1:
            raise ValueError("manifest artifact size must be positive")
        if self.accepted_count < 0 or self.quarantined_count < 0:
            raise ValueError("manifest counts cannot be negative")
        for name, digest in self.source_sha256.items():
            if not name or not SHA256_PATTERN.fullmatch(digest):
                raise ValueError("manifest source checksums must be named SHA-256 values")
        for name, coverage in self.field_coverage.items():
            if not name or not 0.0 <= float(coverage) <= 1.0:
                raise ValueError("field coverage values must be within [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["manifest_version"] = MANIFEST_VERSION
        # ``asdict`` preserves tuples. JSON has only arrays; normalizing here
        # means loaded and newly built manifests compare structurally.
        return json.loads(canonical_json_bytes(payload))

    def write(self, path: str | Path, *, checksum_path: str | Path | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(self.as_dict()))
        destination.chmod(0o644)
        sidecar = Path(checksum_path) if checksum_path else destination.with_suffix(
            destination.suffix + ".sha256"
        )
        sidecar.write_text(f"{sha256_file(destination)}  {destination.name}\n", encoding="ascii")
        sidecar.chmod(0o644)

    @classmethod
    def read(cls, path: str | Path) -> "CatalogManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("manifest_version", None) != MANIFEST_VERSION:
            raise ValueError("unsupported catalog manifest version")
        for name in (
            "licenses",
            "supported_filters",
            "supported_features",
            "retrieval_methods",
        ):
            payload[name] = tuple(payload.get(name, ()))
        return cls(**payload)
