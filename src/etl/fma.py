"""Deterministic FMA metadata normalization and catalog build orchestration.

The official FMA files deliberately use two- and three-row pandas headers.
This module names those columns explicitly instead of flattening them with
positional guesses. Large ``tracks.csv`` files are consumed in chunks, which
keeps repeated artist biographies from multiplying memory use during a build.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import pandas as pd

from src.etl.integrity import sha256_file
from src.etl.manifest import CatalogManifest, canonical_json_bytes
from src.mood import compute_mood_profile


ETL_VERSION = "fma-etl-v2"
CATALOG_SCHEMA_VERSION = "fma-sqlite-v2"
PANDAS_REQUIRED_VERSION = "3.0.3"
CATALOG_ID = "fma"
LITE_SIZE = 300
LITE_SELECTION_SALT = "cadence-fma-lite-v1"
OFFICIAL_FMA_METADATA_SHA256 = (
    "d9527a5297a65da31c5676484d5047c3e2b8a8060ce72a46e26158be736bf265"
)

TARGET_FEATURES: tuple[str, ...] = (
    "energy",
    "valence",
    "acousticness",
    "danceability",
    "tempo_bpm",
    "instrumentalness",
)
UNIT_FEATURES = frozenset(set(TARGET_FEATURES) - {"tempo_bpm"})
ECHONEST_COLUMN = {
    "energy": ("echonest", "audio_features", "energy"),
    "valence": ("echonest", "audio_features", "valence"),
    "acousticness": ("echonest", "audio_features", "acousticness"),
    "danceability": ("echonest", "audio_features", "danceability"),
    "tempo_bpm": ("echonest", "audio_features", "tempo"),
    "instrumentalness": ("echonest", "audio_features", "instrumentalness"),
}

TRACK_REQUIRED_COLUMNS = (
    ("track", "title"),
    ("artist", "name"),
)
TRACK_OPTIONAL_COLUMNS = (
    ("track", "genre_top"),
    ("track", "genres"),
    ("track", "genres_all"),
    ("track", "information"),
    ("track", "tags"),
    ("track", "license"),
    ("track", "date_created"),
    ("album", "date_released"),
    ("album", "information"),
    ("album", "tags"),
    ("artist", "bio"),
    ("artist", "tags"),
    ("artist", "website"),
    ("artist", "wikipedia_page"),
)

_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CONTROL_TRANSLATION = dict.fromkeys(
    code for code in range(32) if code not in {9, 10, 13}
)


class FmaSchemaError(ValueError):
    """Raised when an input no longer matches the official FMA column layout."""


@dataclass(frozen=True, slots=True)
class FmaSourcePaths:
    tracks: Path
    genres: Path
    echonest: Path | None = None
    features: Path | None = None
    predictions: Path | None = None

    def checksums(self) -> dict[str, str]:
        sources = {"tracks.csv": self.tracks, "genres.csv": self.genres}
        if self.echonest is not None:
            sources["echonest.csv"] = self.echonest
        if self.features is not None:
            sources["features.csv"] = self.features
        if self.predictions is not None:
            sources["predictions.jsonl"] = self.predictions
        return {name: sha256_file(path) for name, path in sorted(sources.items())}


@dataclass(frozen=True, slots=True)
class FmaBuildProfile:
    edition: str = "full"
    lite_size: int = LITE_SIZE

    def __post_init__(self) -> None:
        if self.edition not in {"full", "lite"}:
            raise ValueError("FMA build edition must be 'full' or 'lite'")
        if self.lite_size < 1:
            raise ValueError("lite_size must be positive")


@dataclass(frozen=True, slots=True)
class FeatureDatum:
    value: float
    origin: str
    method_version: str
    confidence: float
    interval_low: float | None = None
    interval_high: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionDatum:
    track_id: int
    feature: str
    value: float | None
    confidence: float | None
    interval_low: float | None
    interval_high: float | None
    model_version: str
    released: bool


@dataclass(frozen=True, slots=True)
class NormalizedFmaTrack:
    """Loss-aware intermediate record written into SQLite without Pydantic."""

    track_id: int
    title: str
    artist: str
    primary_genre: str | None
    genres: tuple[str, ...]
    track_tags: tuple[str, ...]
    album_tags: tuple[str, ...]
    artist_tags: tuple[str, ...]
    track_information: str | None
    album_information: str | None
    artist_biography: str | None
    era: str | None
    license: str | None
    source_url: str | None
    track_url: str | None
    artist_url: str | None
    album_url: str | None
    features: Mapping[str, FeatureDatum]
    mood_profile: Mapping[str, Any] | None
    feature_terms: tuple[str, ...]
    lineage: tuple[Mapping[str, Any], ...]
    has_echonest: bool = False

    @property
    def catalog_id(self) -> str:
        return CATALOG_ID

    @property
    def external_id(self) -> str:
        return f"fma:{self.track_id}"


@dataclass(frozen=True, slots=True)
class QuarantinedRow:
    track_id: int | None
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FmaBuildResult:
    database_path: Path
    manifest_path: Path
    quarantine_path: Path
    manifest: CatalogManifest


@dataclass(frozen=True, slots=True)
class ModelMatrixResult:
    output_path: Path
    row_count: int
    feature_count: int
    sha256: str


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style"}:
            self._ignored_depth += 1
        elif tag.casefold() in {"p", "br", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.casefold() in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, type(pd.NA))) else False


@lru_cache(maxsize=100_000)
def _sanitize_cached(raw: str) -> str | None:
    parser = _PlainText()
    try:
        parser.feed(raw)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        # Broken HTML is metadata quality trouble, not a reason to reject an
        # otherwise identifiable track. Treat it as ordinary plain text.
        text = raw
    text = unicodedata.normalize("NFC", text).translate(_CONTROL_TRANSLATION)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text or None


def normalize_text(value: object) -> str | None:
    """Convert source text/HTML to normalized visible text without inventing copy."""
    if _is_missing(value):
        return None
    return _sanitize_cached(str(value))


def _parse_terms(value: object) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ()
    if not isinstance(parsed, (list, tuple)):
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        text = normalize_text(item)
        if text is None:
            continue
        term = text.casefold()
        if len(term) <= 120 and term not in seen:
            seen.add(term)
            normalized.append(term)
    return tuple(normalized)


def _parse_ids(value: object) -> tuple[int, ...]:
    if _is_missing(value):
        return ()
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ()
    if not isinstance(parsed, (list, tuple)):
        return ()
    result: list[int] = []
    for item in parsed:
        if isinstance(item, bool):
            continue
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
    return tuple(result)


def _safe_url(value: object) -> str | None:
    text = normalize_text(value)
    if text is None or len(text) > 2048:
        return None
    try:
        parsed = urlsplit(text)
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return text


def _positive_track_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _feature_value(feature: str, value: object) -> float | None:
    if isinstance(value, bool) or _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if feature in UNIT_FEATURES:
        return number if 0.0 <= number <= 1.0 else None
    if feature == "tempo_bpm":
        return number if 50.0 <= number <= 200.0 else None
    return None


def _era(album_date: object, track_date: object) -> str | None:
    for value in (album_date, track_date):
        text = normalize_text(value)
        if text is None:
            continue
        match = _YEAR_PATTERN.search(text)
        if match:
            year = int(match.group(1))
            return f"{year // 10 * 10}s"
    return None


def load_genre_map(path: str | Path) -> dict[int, str]:
    frame = pd.read_csv(path, index_col=0, keep_default_na=True)
    if "title" not in frame.columns:
        raise FmaSchemaError("genres.csv is missing the explicit 'title' column")
    result: dict[int, str] = {}
    for raw_id, raw_title in frame["title"].items():
        genre_id = _positive_track_id(raw_id)
        title = normalize_text(raw_title)
        if genre_id is not None and title is not None:
            result[genre_id] = title.casefold()
    if not result:
        raise FmaSchemaError("genres.csv did not contain any valid genre identities")
    return result


def _require_columns(frame: pd.DataFrame, expected: Sequence[tuple[str, ...]], label: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        formatted = ", ".join(".".join(column) for column in missing)
        raise FmaSchemaError(f"{label} is missing required columns: {formatted}")


def load_echonest(path: str | Path | None) -> dict[int, dict[str, FeatureDatum]]:
    if path is None:
        return {}
    frame = pd.read_csv(path, index_col=0, header=[0, 1, 2], low_memory=False)
    _require_columns(frame, tuple(ECHONEST_COLUMN.values()), "echonest.csv")
    result: dict[int, dict[str, FeatureDatum]] = {}
    for raw_id, row in frame.iterrows():
        track_id = _positive_track_id(raw_id)
        if track_id is None:
            continue
        values: dict[str, FeatureDatum] = {}
        for feature, column in ECHONEST_COLUMN.items():
            value = _feature_value(feature, row[column])
            if value is not None:
                values[feature] = FeatureDatum(
                    value=value,
                    origin="echonest_computed",
                    method_version="fma-echonest-audio-features",
                    confidence=1.0,
                )
        result[track_id] = values
    return result


def _librosa_column_name(column: tuple[object, object, object]) -> str:
    parts: list[str] = []
    for value in column:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
        if not normalized:
            raise FmaSchemaError(f"features.csv has an empty header component: {column!r}")
        parts.append(normalized)
    return "librosa__" + "__".join(parts)


def _load_artist_map(tracks_path: Path, *, chunk_size: int) -> dict[int, str]:
    artists: dict[int, str] = {}
    chunks = pd.read_csv(
        tracks_path,
        index_col=0,
        header=[0, 1],
        chunksize=chunk_size,
        low_memory=False,
        keep_default_na=True,
    )
    checked = False
    for frame in chunks:
        if not checked:
            _require_columns(frame, TRACK_REQUIRED_COLUMNS, "tracks.csv")
            checked = True
        for raw_id, raw_artist in frame[("artist", "name")].items():
            track_id = _positive_track_id(raw_id)
            artist = normalize_text(raw_artist)
            if track_id is not None and artist is not None:
                if track_id in artists:
                    raise FmaSchemaError(f"tracks.csv contains duplicate track ID {track_id}")
                artists[track_id] = artist
    if not checked:
        raise FmaSchemaError("tracks.csv has headers but no data rows")
    return artists


def prepare_model_matrix(
    paths: FmaSourcePaths,
    output_path: str | Path,
    *,
    chunk_size: int = 1_000,
    expected_feature_count: int = 518,
) -> ModelMatrixResult:
    """Stream the explicit 3-row Librosa table into the modeling CSV contract.

    Output columns are ``track_id``, ``artist``, 518 stable
    ``librosa__feature__statistic__number`` inputs, then the six Echo Nest
    targets. Raw Librosa vectors are an offline build artifact and are never
    copied into the serving SQLite database.
    """
    if paths.features is None:
        raise ValueError("features.csv is required to prepare the model matrix")
    if paths.echonest is None:
        raise ValueError("echonest.csv is required to prepare model targets")
    if expected_feature_count < 1:
        raise ValueError("expected_feature_count must be positive")
    artists = _load_artist_map(paths.tracks, chunk_size=chunk_size)
    echonest = load_echonest(paths.echonest)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    row_count = 0
    feature_names: tuple[str, ...] | None = None
    previous_track_id = 0
    wrote_header = False
    try:
        chunks = pd.read_csv(
            paths.features,
            index_col=0,
            header=[0, 1, 2],
            chunksize=chunk_size,
            low_memory=False,
            keep_default_na=True,
        )
        for frame in chunks:
            current_names = tuple(_librosa_column_name(column) for column in frame.columns)
            if feature_names is None:
                feature_names = current_names
                if len(feature_names) != expected_feature_count:
                    raise FmaSchemaError(
                        f"features.csv must contain {expected_feature_count} Librosa columns; "
                        f"found {len(feature_names)}"
                    )
                if len(feature_names) != len(set(feature_names)):
                    raise FmaSchemaError("features.csv headers do not flatten to unique names")
            elif current_names != feature_names:
                raise FmaSchemaError("features.csv column order changed between chunks")

            raw = frame.copy()
            raw.columns = feature_names
            numeric = raw.apply(pd.to_numeric, errors="coerce")
            invalid = raw.notna() & numeric.isna()
            if invalid.any(axis=None):
                raise FmaSchemaError("features.csv contains a non-numeric Librosa value")
            finite = numeric.map(lambda value: _is_missing(value) or math.isfinite(float(value)))
            if not finite.all(axis=None):
                raise FmaSchemaError("features.csv contains a non-finite Librosa value")

            output_rows: list[dict[str, object]] = []
            for raw_id, values in numeric.iterrows():
                track_id = _positive_track_id(raw_id)
                if track_id is None:
                    raise FmaSchemaError("features.csv contains an invalid track ID")
                if track_id <= previous_track_id:
                    raise FmaSchemaError("features.csv track IDs must be strictly increasing")
                previous_track_id = track_id
                artist = artists.get(track_id)
                if artist is None:
                    continue  # not a valid catalog identity; ETL quarantine owns the reason
                row: dict[str, object] = {"track_id": track_id, "artist": artist}
                row.update(values.to_dict())
                targets = echonest.get(track_id, {})
                for feature in TARGET_FEATURES:
                    datum = targets.get(feature)
                    row[feature] = datum.value if datum is not None else None
                output_rows.append(row)
            if not output_rows:
                continue
            output_frame = pd.DataFrame.from_records(
                output_rows,
                columns=("track_id", "artist", *feature_names, *TARGET_FEATURES),
            )
            output_frame.to_csv(
                temporary,
                mode="a",
                header=not wrote_header,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
                na_rep="",
                float_format="%.10g",
            )
            wrote_header = True
            row_count += len(output_frame)
        if feature_names is None:
            raise FmaSchemaError("features.csv has headers but no data rows")
        if not wrote_header:
            raise FmaSchemaError("no feature rows matched a valid track identity")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return ModelMatrixResult(
        output_path=destination,
        row_count=row_count,
        feature_count=len(feature_names),
        sha256=sha256_file(destination),
    )


def _optional_number(value: object, *, unit: bool = False) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (unit and not 0.0 <= number <= 1.0):
        return None
    return number


def load_predictions(path: str | Path | None) -> dict[tuple[int, str], PredictionDatum]:
    if path is None:
        return {}
    predictions: dict[tuple[int, str], PredictionDatum] = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid prediction JSON on line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"prediction line {line_number} must be an object")
            required = {
                "track_id", "feature", "value", "confidence", "interval_low",
                "interval_high", "model_version", "released",
            }
            if set(payload) != required:
                raise ValueError(
                    f"prediction line {line_number} must contain exactly {sorted(required)}"
                )
            track_id = _positive_track_id(payload["track_id"])
            feature = payload["feature"]
            released = payload["released"]
            if track_id is None or feature not in TARGET_FEATURES or not isinstance(released, bool):
                raise ValueError(f"prediction line {line_number} has invalid identity fields")
            model_version = normalize_text(payload["model_version"])
            if model_version is None:
                raise ValueError(f"prediction line {line_number} has no model version")
            value = _feature_value(feature, payload["value"])
            confidence = _optional_number(payload["confidence"], unit=True)
            low = _optional_number(payload["interval_low"])
            high = _optional_number(payload["interval_high"])
            if released and (value is None or confidence is None or low is None or high is None):
                raise ValueError(f"released prediction line {line_number} is incomplete or invalid")
            if low is not None and high is not None and low > high:
                raise ValueError(f"prediction line {line_number} interval is reversed")
            if released and not (low <= value <= high):  # type: ignore[operator]
                raise ValueError(f"prediction line {line_number} value is outside its interval")
            key = (track_id, feature)
            if key in predictions:
                raise ValueError(f"duplicate prediction for track {track_id} feature {feature}")
            predictions[key] = PredictionDatum(
                track_id=track_id,
                feature=feature,
                value=value,
                confidence=confidence,
                interval_low=low,
                interval_high=high,
                model_version=model_version,
                released=released,
            )
    return predictions


def _selected_features(
    track_id: int,
    echonest: Mapping[int, Mapping[str, FeatureDatum]],
    predictions: Mapping[tuple[int, str], PredictionDatum],
) -> tuple[dict[str, FeatureDatum], bool]:
    direct = echonest.get(track_id, {})
    selected = dict(direct)
    for feature in TARGET_FEATURES:
        if feature in selected:
            continue
        prediction = predictions.get((track_id, feature))
        if prediction is None or not prediction.released or prediction.value is None:
            continue
        selected[feature] = FeatureDatum(
            value=prediction.value,
            origin="model_estimated",
            method_version=prediction.model_version,
            confidence=prediction.confidence,  # validated above
            interval_low=prediction.interval_low,
            interval_high=prediction.interval_high,
        )
    return selected, bool(direct)


def _feature_terms(features: Mapping[str, FeatureDatum], mood_label: str | None) -> tuple[str, ...]:
    terms: list[str] = []
    for feature in ("energy", "valence", "danceability", "acousticness", "instrumentalness"):
        datum = features.get(feature)
        if datum is None:
            continue
        if datum.value >= 2.0 / 3.0:
            terms.extend((f"high {feature}",))
        elif datum.value <= 1.0 / 3.0:
            terms.extend((f"low {feature}",))
    tempo = features.get("tempo_bpm")
    if tempo is not None:
        if tempo.value >= 140.0:
            terms.append("fast tempo")
        elif tempo.value <= 80.0:
            terms.append("slow tempo")
    if mood_label is not None:
        terms.append(mood_label)
    return tuple(terms)


def _lineage(
    features: Mapping[str, FeatureDatum], *, has_era: bool, has_mood: bool
) -> tuple[Mapping[str, Any], ...]:
    entries: list[dict[str, Any]] = [
        {"field_name": name, "origin": "fma_metadata", "source_fields": [source]}
        for name, source in (
            ("title", "track.title"),
            ("artist", "artist.name"),
            ("genres", "track.genres_all+genres.csv"),
        )
    ]
    for feature in TARGET_FEATURES:
        datum = features.get(feature)
        if datum is None:
            continue
        entries.append(
            {
                "field_name": feature,
                "origin": datum.origin,
                "source_fields": [
                    f"echonest.audio_features.{feature.removesuffix('_bpm')}"
                    if datum.origin == "echonest_computed"
                    else "features.csv:librosa_518"
                ],
                "method_version": datum.method_version,
                "confidence": datum.confidence,
                "interval_low": datum.interval_low,
                "interval_high": datum.interval_high,
            }
        )
    if has_era:
        entries.append(
            {
                "field_name": "era",
                "origin": "deterministic_derived",
                "source_fields": ["album.date_released", "track.date_created"],
                "method_version": "calendar-decade-v1",
            }
        )
    if has_mood:
        entries.append(
            {
                "field_name": "mood_profile",
                "origin": "deterministic_derived",
                "source_fields": ["energy", "valence"],
                "method_version": "cadence-va-quadrant-v1",
            }
        )
    return tuple(entries)


def iter_normalized_tracks(
    paths: FmaSourcePaths,
    *,
    chunk_size: int = 5_000,
) -> Iterator[NormalizedFmaTrack | QuarantinedRow]:
    """Yield validated normalized rows, keeping malformed identities in quarantine."""
    # Fresh ML/ETL installs are pinned exactly in requirements-ml.txt. Accept a
    # later pandas 3.0 patch in an already-provisioned developer environment;
    # rejecting security/bug-fix patches would make fixture validation brittle.
    if tuple(int(part) for part in pd.__version__.split(".")[:2]) != (3, 0):
        raise RuntimeError(
            f"FMA ETL requires pandas 3.0.x (pinned to {PANDAS_REQUIRED_VERSION}); "
            f"found {pd.__version__}"
        )
    genre_map = load_genre_map(paths.genres)
    echonest = load_echonest(paths.echonest)
    predictions = load_predictions(paths.predictions)
    seen_ids: set[int] = set()

    chunks = pd.read_csv(
        paths.tracks,
        index_col=0,
        header=[0, 1],
        chunksize=chunk_size,
        low_memory=False,
        keep_default_na=True,
    )
    checked_schema = False
    for frame in chunks:
        if not checked_schema:
            _require_columns(frame, TRACK_REQUIRED_COLUMNS, "tracks.csv")
            checked_schema = True
        for raw_id, row in frame.iterrows():
            track_id = _positive_track_id(raw_id)
            title = normalize_text(row[("track", "title")])
            artist = normalize_text(row[("artist", "name")])
            reasons: list[str] = []
            if track_id is None:
                reasons.append("invalid_track_id")
            elif track_id in seen_ids:
                reasons.append("duplicate_track_id")
            if title is None:
                reasons.append("missing_title")
            if artist is None:
                reasons.append("missing_artist")
            if reasons:
                yield QuarantinedRow(track_id=track_id, reasons=tuple(reasons))
                continue
            assert track_id is not None and title is not None and artist is not None
            seen_ids.add(track_id)

            def optional(column: tuple[str, str]) -> object:
                return row[column] if column in frame.columns else None

            primary = normalize_text(optional(("track", "genre_top")))
            primary = primary.casefold() if primary else None
            genre_ids = _parse_ids(optional(("track", "genres_all")))
            if not genre_ids:
                genre_ids = _parse_ids(optional(("track", "genres")))
            genres: list[str] = []
            if primary is not None:
                genres.append(primary)
            for genre_id in genre_ids:
                name = genre_map.get(genre_id)
                if name is not None and name not in genres:
                    genres.append(name)

            features, has_echonest = _selected_features(track_id, echonest, predictions)
            energy = features.get("energy")
            valence = features.get("valence")
            mood = compute_mood_profile(
                energy.value if energy else None,
                valence.value if valence else None,
                energy_confidence=energy.confidence if energy else None,
                valence_confidence=valence.confidence if valence else None,
            )
            mood_payload = mood.as_profile_kwargs() if mood else None
            era = _era(
                optional(("album", "date_released")),
                optional(("track", "date_created")),
            )
            artist_url = _safe_url(optional(("artist", "website"))) or _safe_url(
                optional(("artist", "wikipedia_page"))
            )
            yield NormalizedFmaTrack(
                track_id=track_id,
                title=title,
                artist=artist,
                primary_genre=primary,
                genres=tuple(genres),
                track_tags=_parse_terms(optional(("track", "tags"))),
                album_tags=_parse_terms(optional(("album", "tags"))),
                artist_tags=_parse_terms(optional(("artist", "tags"))),
                track_information=normalize_text(optional(("track", "information"))),
                album_information=normalize_text(optional(("album", "information"))),
                artist_biography=normalize_text(optional(("artist", "bio"))),
                era=era,
                license=normalize_text(optional(("track", "license"))),
                # FMA does not supply a canonical track/album URL in tracks.csv.
                # Synthesizing one would violate the source-evidence boundary.
                source_url=None,
                track_url=None,
                artist_url=artist_url,
                album_url=None,
                features=features,
                mood_profile=mood_payload,
                feature_terms=_feature_terms(
                    features, mood_payload.get("label") if mood_payload else None
                ),
                lineage=_lineage(features, has_era=era is not None, has_mood=mood is not None),
                has_echonest=has_echonest,
            )

    if not checked_schema:
        raise FmaSchemaError("tracks.csv has headers but no data rows")


def select_lite_tracks(
    tracks: Iterable[NormalizedFmaTrack], *, size: int = LITE_SIZE
) -> tuple[NormalizedFmaTrack, ...]:
    """Stable-hash, genre round-robin selection from complete Echo Nest rows."""
    eligible = [
        track
        for track in tracks
        if track.primary_genre is not None
        and track.has_echonest
        and all(feature in track.features for feature in TARGET_FEATURES)
    ]
    groups: dict[str, list[NormalizedFmaTrack]] = {}
    for track in eligible:
        groups.setdefault(track.primary_genre, []).append(track)
    for group in groups.values():
        group.sort(
            key=lambda track: (
                hashlib.sha256(
                    f"{LITE_SELECTION_SALT}:{track.track_id}".encode("ascii")
                ).digest(),
                track.track_id,
            )
        )

    selected: list[NormalizedFmaTrack] = []
    positions = {name: 0 for name in groups}
    names = sorted(groups)
    while len(selected) < min(size, len(eligible)):
        progressed = False
        for name in names:
            position = positions[name]
            group = groups[name]
            if position >= len(group):
                continue
            selected.append(group[position])
            positions[name] += 1
            progressed = True
            if len(selected) == min(size, len(eligible)):
                break
        if not progressed:
            break
    return tuple(sorted(selected, key=lambda track: track.track_id))


def _coverage(tracks: Sequence[NormalizedFmaTrack]) -> dict[str, float]:
    total = len(tracks)
    if not total:
        return {}
    fields: dict[str, int] = {
        "primary_genre": sum(track.primary_genre is not None for track in tracks),
        "track_information": sum(track.track_information is not None for track in tracks),
        "album_information": sum(track.album_information is not None for track in tracks),
        "artist_biography": sum(track.artist_biography is not None for track in tracks),
        "license": sum(track.license is not None for track in tracks),
        "mood_profile": sum(track.mood_profile is not None for track in tracks),
    }
    for feature in TARGET_FEATURES:
        fields[feature] = sum(feature in track.features for track in tracks)
        fields[f"{feature}_estimated"] = sum(
            track.features.get(feature) is not None
            and track.features[feature].origin == "model_estimated"
            for track in tracks
        )
    return {name: round(count / total, 8) for name, count in sorted(fields.items())}


def build_fma_catalog(
    paths: FmaSourcePaths,
    database_path: str | Path,
    *,
    profile: FmaBuildProfile = FmaBuildProfile(),
    manifest_path: str | Path | None = None,
    quarantine_path: str | Path | None = None,
) -> FmaBuildResult:
    """Build a deterministic SQLite artifact, manifest, and quarantine report."""
    from src.fma_store import build_fma_sqlite  # runtime never imports pandas

    database = Path(database_path)
    manifest_destination = (
        Path(manifest_path) if manifest_path else database.with_suffix(".manifest.json")
    )
    quarantine_destination = (
        Path(quarantine_path) if quarantine_path else database.with_suffix(".quarantine.jsonl")
    )

    accepted: list[NormalizedFmaTrack] = []
    quarantined: list[QuarantinedRow] = []
    for item in iter_normalized_tracks(paths):
        if isinstance(item, QuarantinedRow):
            quarantined.append(item)
        else:
            accepted.append(item)
    accepted.sort(key=lambda track: track.track_id)
    selected = (
        select_lite_tracks(accepted, size=profile.lite_size)
        if profile.edition == "lite"
        else tuple(accepted)
    )
    if not selected:
        raise ValueError(f"FMA {profile.edition} profile did not select any valid tracks")

    database.parent.mkdir(parents=True, exist_ok=True)
    build_fma_sqlite(selected, database)
    quarantine_destination.parent.mkdir(parents=True, exist_ok=True)
    quarantine_payload = b"".join(
        canonical_json_bytes(
            {"track_id": row.track_id, "reasons": list(row.reasons)}
        )
        for row in sorted(
            quarantined,
            key=lambda row: (row.track_id is None, row.track_id or 0, row.reasons),
        )
    )
    quarantine_destination.write_bytes(quarantine_payload)

    manifest = CatalogManifest(
        catalog_id=CATALOG_ID,
        artifact_id=f"fma-{profile.edition}-{ETL_VERSION}-{sha256_file(database)[:12]}",
        edition=profile.edition,
        schema_version=CATALOG_SCHEMA_VERSION,
        etl_version=ETL_VERSION,
        artifact_sha256=sha256_file(database),
        accepted_count=len(selected),
        quarantined_count=len(quarantined),
        artifact_bytes=database.stat().st_size,
        source_sha256=paths.checksums(),
        field_coverage=_coverage(selected),
        licenses=("FMA metadata: CC BY 4.0", "Track audio: per-artist license"),
        supported_filters=(),
        supported_features=TARGET_FEATURES,
        calibration_status="experimental",
    )
    manifest.write(manifest_destination)
    return FmaBuildResult(
        database_path=database,
        manifest_path=manifest_destination,
        quarantine_path=quarantine_destination,
        manifest=manifest,
    )
