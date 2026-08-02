"""Offline deterministic catalog builders, lazily exported.

The deployed SQLite runtime imports ``src.etl.manifest`` for a tiny JSON
contract. Python executes this package initializer first, so it must never
eagerly import pandas-backed ``src.etl.fma``.
"""

from __future__ import annotations

from typing import Any


_FMA_EXPORTS = {
    "ETL_VERSION",
    "FmaBuildProfile",
    "FmaBuildResult",
    "FmaSourcePaths",
    "build_fma_catalog",
    "prepare_model_matrix",
}
_INTEGRITY_EXPORTS = {
    "ChecksumMismatchError",
    "UnsafeArchiveError",
    "safe_extract_zip",
    "sha256_file",
    "verify_sha256",
}
__all__ = sorted(_FMA_EXPORTS | _INTEGRITY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _FMA_EXPORTS:
        from src.etl import fma

        return getattr(fma, name)
    if name in _INTEGRITY_EXPORTS:
        from src.etl import integrity

        return getattr(integrity, name)
    raise AttributeError(name)
