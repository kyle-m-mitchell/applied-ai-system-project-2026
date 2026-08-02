"""Read-only SQLite store for the large FMA catalog.

The runtime module is standard-library only. It returns small candidate sets and
materializes tracks only by ID, rather than constructing roughly 106,000
Pydantic models at application startup.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence
from urllib.parse import quote

from src.etl.manifest import CatalogManifest


APPLICATION_ID = 0x43414435  # "CAD5"
USER_VERSION = 2
DEFAULT_TEXT_LIMIT = 200
MAX_SEARCH_LIMIT = 1_000
TARGET_FEATURES: tuple[str, ...] = (
    "energy",
    "valence",
    "acousticness",
    "danceability",
    "tempo_bpm",
    "instrumentalness",
)
UNIT_FEATURES = frozenset(set(TARGET_FEATURES) - {"tempo_bpm"})
_QUERY_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


class Fts5UnavailableError(RuntimeError):
    """The current SQLite build cannot create or query FTS5 indexes."""


class CatalogDatabaseError(RuntimeError):
    """A catalog artifact is corrupt or has an unsupported schema."""


@dataclass(frozen=True, slots=True)
class StoredFeature:
    value: float
    origin: str
    method_version: str
    confidence: float
    interval_low: float | None = None
    interval_high: float | None = None


@dataclass(frozen=True, slots=True)
class StoredTrack:
    id: int
    catalog_id: str
    external_id: str
    title: str
    artist: str
    genre: str | None
    genres: tuple[str, ...]
    mood: None
    mood_profile: Mapping[str, Any] | None
    energy: float | None
    valence: float | None
    acousticness: float | None
    danceability: float | None
    tempo_bpm: float | None
    instrumentalness: float | None
    description: None
    tags: tuple[str, ...]
    album_tags: tuple[str, ...]
    artist_tags: tuple[str, ...]
    contexts: tuple[str, ...]
    instruments: tuple[str, ...]
    instrumental: None
    explicit: None
    era: str | None
    track_information: str | None
    album_information: str | None
    artist_biography: str | None
    license: str | None
    source_url: str | None
    track_url: str | None
    artist_url: str | None
    album_url: str | None
    lineage: tuple[Mapping[str, Any], ...]
    feature_data: Mapping[str, StoredFeature]

    @property
    def track_ref(self) -> tuple[str, int, str]:
        return self.catalog_id, self.id, self.external_id

    def as_contract_payload(self) -> dict[str, Any]:
        """Return shared-contract-shaped data without importing Pydantic here."""
        return {
            field: getattr(self, field)
            for field in (
                "id", "catalog_id", "external_id", "title", "artist", "genre",
                "genres", "mood", "mood_profile", "energy", "valence",
                "acousticness", "danceability", "tempo_bpm", "instrumentalness",
                "description", "tags", "album_tags", "artist_tags", "contexts",
                "instruments", "instrumental", "explicit", "era",
                "track_information", "album_information", "artist_biography",
                "license", "source_url", "track_url", "artist_url", "album_url",
                "lineage",
            )
        }

    def to_contract(self) -> Any:
        """Lazily validate against the current shared ``CatalogTrack`` contract."""
        from src.contracts import CatalogTrack

        return CatalogTrack.model_validate(self.as_contract_payload())


@dataclass(frozen=True, slots=True)
class CatalogStoreHit:
    track_id: int
    score: float
    reasons: tuple[str, ...] = ()
    raw_rank: float | None = None


@dataclass(frozen=True, slots=True)
class StructuredFeatureGoal:
    feature: str
    relation: str
    strength: float = 1.0
    target: float | None = None
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.feature not in TARGET_FEATURES:
            raise ValueError(f"unsupported FMA structured feature: {self.feature}")
        if self.relation not in {
            "prefer_high", "prefer_low", "near", "at_least", "at_most", "range"
        }:
            raise ValueError(f"unsupported structured relation: {self.relation}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("goal strength must be within [0, 1]")
        if self.relation in {"near", "at_least", "at_most"} and self.target is None:
            raise ValueError(f"{self.relation} requires target")
        if self.relation == "range" and (self.low is None or self.high is None):
            raise ValueError("range requires low and high")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("range low must be <= high")


@dataclass(frozen=True, slots=True)
class CatalogVocabulary:
    genres: tuple[str, ...]
    mood_labels: tuple[str, ...]
    feature_names: tuple[str, ...]


class CatalogStore(Protocol):
    """Duck-typed serving seam shared by lazy catalog implementations."""

    @property
    def count(self) -> int: ...
    def vocabulary(self) -> CatalogVocabulary: ...
    def text_search(self, query: str, *, limit: int = DEFAULT_TEXT_LIMIT) -> tuple[CatalogStoreHit, ...]: ...
    def structured_search(
        self, *, genre: str | None = None, goals: Sequence[Any] = (), limit: int = DEFAULT_TEXT_LIMIT
    ) -> tuple[CatalogStoreHit, ...]: ...
    def get_tracks(self, track_ids: Sequence[int]) -> tuple[StoredTrack, ...]: ...


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sequence_text(values: Iterable[str]) -> str:
    return " ".join(dict.fromkeys(value for value in values if value))


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size = 4096;
        PRAGMA auto_vacuum = NONE;
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA locking_mode = EXCLUSIVE;
        PRAGMA application_id = 1128350773;
        PRAGMA user_version = 2;

        CREATE TABLE catalog_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE tracks (
            track_id INTEGER PRIMARY KEY CHECK(track_id > 0),
            external_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL CHECK(length(title) > 0),
            artist TEXT NOT NULL CHECK(length(artist) > 0),
            primary_genre TEXT,
            era TEXT,
            track_information TEXT,
            album_information TEXT,
            artist_biography TEXT,
            license TEXT,
            source_url TEXT,
            track_url TEXT,
            artist_url TEXT,
            album_url TEXT,
            energy REAL CHECK(energy BETWEEN 0.0 AND 1.0),
            energy_origin TEXT,
            energy_method TEXT,
            energy_confidence REAL CHECK(energy_confidence BETWEEN 0.0 AND 1.0),
            energy_interval_low REAL,
            energy_interval_high REAL,
            valence REAL CHECK(valence BETWEEN 0.0 AND 1.0),
            valence_origin TEXT,
            valence_method TEXT,
            valence_confidence REAL CHECK(valence_confidence BETWEEN 0.0 AND 1.0),
            valence_interval_low REAL,
            valence_interval_high REAL,
            acousticness REAL CHECK(acousticness BETWEEN 0.0 AND 1.0),
            acousticness_origin TEXT,
            acousticness_method TEXT,
            acousticness_confidence REAL CHECK(acousticness_confidence BETWEEN 0.0 AND 1.0),
            acousticness_interval_low REAL,
            acousticness_interval_high REAL,
            danceability REAL CHECK(danceability BETWEEN 0.0 AND 1.0),
            danceability_origin TEXT,
            danceability_method TEXT,
            danceability_confidence REAL CHECK(danceability_confidence BETWEEN 0.0 AND 1.0),
            danceability_interval_low REAL,
            danceability_interval_high REAL,
            tempo_bpm REAL CHECK(tempo_bpm BETWEEN 50.0 AND 200.0),
            tempo_bpm_origin TEXT,
            tempo_bpm_method TEXT,
            tempo_bpm_confidence REAL CHECK(tempo_bpm_confidence BETWEEN 0.0 AND 1.0),
            tempo_bpm_interval_low REAL,
            tempo_bpm_interval_high REAL,
            instrumentalness REAL CHECK(instrumentalness BETWEEN 0.0 AND 1.0),
            instrumentalness_origin TEXT,
            instrumentalness_method TEXT,
            instrumentalness_confidence REAL CHECK(instrumentalness_confidence BETWEEN 0.0 AND 1.0),
            instrumentalness_interval_low REAL,
            instrumentalness_interval_high REAL,
            mood_upbeat REAL,
            mood_calm REAL,
            mood_intense REAL,
            mood_somber REAL,
            mood_label TEXT,
            mood_confidence REAL,
            mood_method TEXT,
            mood_experimental INTEGER CHECK(mood_experimental IN (0, 1)),
            feature_terms TEXT NOT NULL DEFAULT '',
            lineage_json TEXT NOT NULL,
            has_echonest INTEGER NOT NULL CHECK(has_echonest IN (0, 1))
        );

        CREATE TABLE track_genres (
            track_id INTEGER NOT NULL REFERENCES tracks(track_id),
            position INTEGER NOT NULL CHECK(position >= 0),
            genre TEXT NOT NULL,
            PRIMARY KEY(track_id, position),
            UNIQUE(track_id, genre)
        ) WITHOUT ROWID;
        CREATE INDEX track_genres_by_genre ON track_genres(genre, track_id);

        CREATE TABLE track_tags (
            track_id INTEGER NOT NULL REFERENCES tracks(track_id),
            scope TEXT NOT NULL CHECK(scope IN ('track', 'album', 'artist')),
            position INTEGER NOT NULL CHECK(position >= 0),
            tag TEXT NOT NULL,
            PRIMARY KEY(track_id, scope, position),
            UNIQUE(track_id, scope, tag)
        ) WITHOUT ROWID;
        CREATE INDEX track_tags_by_tag ON track_tags(tag, track_id);

        CREATE VIRTUAL TABLE tracks_fts USING fts5(
            title,
            artist,
            genres,
            track_tags,
            album_tags,
            artist_tags,
            track_information,
            album_information,
            artist_biography,
            feature_terms,
            content='',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def _track_values(track: Any) -> list[Any]:
    values: list[Any] = [
        track.track_id,
        track.external_id,
        track.title,
        track.artist,
        track.primary_genre,
        track.era,
        track.track_information,
        track.album_information,
        track.artist_biography,
        track.license,
        track.source_url,
        track.track_url,
        track.artist_url,
        track.album_url,
    ]
    for feature in TARGET_FEATURES:
        datum = track.features.get(feature)
        values.extend(
            (
                datum.value if datum else None,
                datum.origin if datum else None,
                datum.method_version if datum else None,
                datum.confidence if datum else None,
                datum.interval_low if datum else None,
                datum.interval_high if datum else None,
            )
        )
    mood = track.mood_profile or {}
    values.extend(
        (
            mood.get("upbeat"),
            mood.get("calm"),
            mood.get("intense"),
            mood.get("somber"),
            mood.get("label"),
            mood.get("confidence"),
            mood.get("method_version"),
            int(bool(mood.get("experimental"))) if mood else None,
            _sequence_text(track.feature_terms),
            _canonical_json(list(track.lineage)),
            int(track.has_echonest),
        )
    )
    return values


def build_fma_sqlite(tracks: Sequence[Any], destination: str | Path) -> None:
    """Atomically create a normalized, deterministic SQLite/FTS5 catalog."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
        except sqlite3.OperationalError as exc:
            connection.close()
            if "fts5" in str(exc).casefold():
                raise Fts5UnavailableError("SQLite was built without FTS5 support") from exc
            raise

        try:
            columns = [
                "track_id", "external_id", "title", "artist", "primary_genre", "era",
                "track_information", "album_information", "artist_biography", "license",
                "source_url", "track_url", "artist_url", "album_url",
            ]
            for feature in TARGET_FEATURES:
                columns.extend(
                    (
                        feature, f"{feature}_origin", f"{feature}_method",
                        f"{feature}_confidence", f"{feature}_interval_low",
                        f"{feature}_interval_high",
                    )
                )
            columns.extend(
                (
                    "mood_upbeat", "mood_calm", "mood_intense", "mood_somber",
                    "mood_label", "mood_confidence", "mood_method", "mood_experimental",
                    "feature_terms", "lineage_json", "has_echonest",
                )
            )
            ordered = sorted(tracks, key=lambda item: item.track_id)
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                f"INSERT INTO tracks ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                (_track_values(track) for track in ordered),
            )
            genre_rows: list[tuple[int, int, str]] = []
            tag_rows: list[tuple[int, str, int, str]] = []
            fts_rows: list[tuple[Any, ...]] = []
            for track in ordered:
                genre_rows.extend(
                    (track.track_id, position, genre)
                    for position, genre in enumerate(track.genres)
                )
                for scope, tags in (
                    ("track", track.track_tags),
                    ("album", track.album_tags),
                    ("artist", track.artist_tags),
                ):
                    tag_rows.extend(
                        (track.track_id, scope, position, tag)
                        for position, tag in enumerate(tags)
                    )
                fts_rows.append(
                    (
                        track.track_id,
                        track.title,
                        track.artist,
                        _sequence_text(track.genres),
                        _sequence_text(track.track_tags),
                        _sequence_text(track.album_tags),
                        _sequence_text(track.artist_tags),
                        track.track_information or "",
                        track.album_information or "",
                        track.artist_biography or "",
                        _sequence_text(track.feature_terms),
                    )
                )
            connection.executemany(
                "INSERT INTO track_genres(track_id,position,genre) VALUES (?,?,?)",
                genre_rows,
            )
            connection.executemany(
                "INSERT INTO track_tags(track_id,scope,position,tag) VALUES (?,?,?,?)",
                tag_rows,
            )
            connection.executemany(
                "INSERT INTO tracks_fts(rowid,title,artist,genres,track_tags,album_tags,"
                "artist_tags,track_information,album_information,artist_biography,feature_terms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                fts_rows,
            )
            connection.executemany(
                "INSERT INTO catalog_metadata(key,value) VALUES (?,?)",
                (
                    ("catalog_id", "fma"),
                    ("schema_version", f"fma-sqlite-v{USER_VERSION}"),
                    ("track_count", str(len(ordered))),
                ),
            )
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()
        os.replace(temporary, target)
        target.chmod(0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def validate_fma_database(path: str | Path, *, require_fts: bool = True) -> int:
    """Run cheap integrity/schema checks and return the stored track count."""
    database = Path(path)
    if not database.is_file():
        raise CatalogDatabaseError(f"catalog database does not exist: {database}")
    try:
        connection = _readonly_connection(database)
        try:
            if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise CatalogDatabaseError("catalog database has the wrong application ID")
            if connection.execute("PRAGMA user_version").fetchone()[0] != USER_VERSION:
                raise CatalogDatabaseError("unsupported catalog database schema version")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CatalogDatabaseError("catalog database failed SQLite quick_check")
            if require_fts:
                found = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks_fts'"
                ).fetchone()
                if found is None:
                    raise Fts5UnavailableError("catalog has no FTS5 index")
                connection.execute("SELECT rowid FROM tracks_fts LIMIT 1").fetchone()
            count = connection.execute("SELECT count(*) FROM tracks").fetchone()[0]
            stored = connection.execute(
                "SELECT value FROM catalog_metadata WHERE key='track_count'"
            ).fetchone()
            if stored is None or int(stored[0]) != count:
                raise CatalogDatabaseError("catalog track count metadata does not match")
            return int(count)
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise CatalogDatabaseError(f"invalid SQLite catalog: {database.name}") from exc


def _fts_query(raw_query: str) -> str:
    if not isinstance(raw_query, str):
        raise TypeError("catalog text query must be a string")
    tokens = [match.group(0).casefold() for match in _QUERY_TOKEN.finditer(raw_query[:2_000])]
    tokens = list(dict.fromkeys(token for token in tokens if len(token) >= 2))[:32]
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"*' for token in tokens)


def _bounded_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_SEARCH_LIMIT}")
    return limit


class FmaCatalogStore:
    """Small-query interface over a read-only FMA SQLite artifact."""

    def __init__(self, database_path: str | Path, manifest_path: str | Path | None = None):
        self.database_path = Path(database_path)
        self._count = validate_fma_database(self.database_path)
        self.manifest = CatalogManifest.read(manifest_path) if manifest_path else None
        if self.manifest is not None and self.manifest.accepted_count != self._count:
            raise CatalogDatabaseError("manifest and database track counts differ")

    @property
    def count(self) -> int:
        return self._count

    @property
    def descriptor(self) -> Any | None:
        """Lazily adapt the distribution manifest to the shared descriptor."""
        if self.manifest is None:
            return None
        from src.contracts import CatalogCapabilities, CatalogDescriptor

        combined_source = hashlib.sha256(
            "|".join(
                f"{name}:{digest}" for name, digest in sorted(self.manifest.source_sha256.items())
            ).encode("ascii")
        ).hexdigest()
        return CatalogDescriptor(
            catalog_id=self.manifest.catalog_id,
            artifact_id=self.manifest.artifact_id,
            edition=self.manifest.edition,
            schema_version=self.manifest.schema_version,
            etl_version=self.manifest.etl_version,
            source_checksum=combined_source,
            artifact_checksum=self.manifest.artifact_sha256,
            accepted_count=self.manifest.accepted_count,
            quarantined_count=self.manifest.quarantined_count,
            licenses=self.manifest.licenses,
            attribution=(self.manifest.attribution,),
            field_coverage=self.manifest.field_coverage,
            capabilities=CatalogCapabilities(
                supported_filters=self.manifest.supported_filters,
                supported_features=self.manifest.supported_features,
                retrieval_methods=self.manifest.retrieval_methods,
                context_guides=self.manifest.context_guides,
                research=self.manifest.research,
            ),
            calibration_status=self.manifest.calibration_status,
        )

    def _connect(self) -> sqlite3.Connection:
        return _readonly_connection(self.database_path)

    def valid_track_ids(self) -> frozenset[int]:
        with self._connect() as connection:
            return frozenset(row[0] for row in connection.execute("SELECT track_id FROM tracks"))

    def vocabulary(self) -> CatalogVocabulary:
        with self._connect() as connection:
            genres = tuple(
                row[0] for row in connection.execute("SELECT DISTINCT genre FROM track_genres ORDER BY genre")
            )
            moods = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT mood_label FROM tracks WHERE mood_label IS NOT NULL ORDER BY mood_label"
                )
            )
        return CatalogVocabulary(genres=genres, mood_labels=moods, feature_names=TARGET_FEATURES)

    def text_search(self, query: str, *, limit: int = DEFAULT_TEXT_LIMIT) -> tuple[CatalogStoreHit, ...]:
        limit = _bounded_limit(limit)
        match = _fts_query(query)
        if not match:
            return ()
        with self._connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT rowid AS track_id,
                           bm25(tracks_fts, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5,
                                2.5, 1.0, 0.5, 2.0) AS text_rank
                    FROM tracks_fts
                    WHERE tracks_fts MATCH ?
                    ORDER BY text_rank ASC, rowid ASC
                    LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                raise Fts5UnavailableError("the selected catalog cannot execute FTS5 search") from exc
        hits: list[CatalogStoreHit] = []
        for row in rows:
            rank = float(row["text_rank"])
            # BM25 is negative in SQLite FTS5; this monotonic mapping is bounded
            # evidence strength, not a calibrated probability.
            score = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, rank))))
            hits.append(
                CatalogStoreHit(
                    track_id=int(row["track_id"]),
                    score=score,
                    reasons=("fts5",),
                    raw_rank=rank,
                )
            )
        return tuple(hits)

    @staticmethod
    def _coerce_goal(raw: Any) -> StructuredFeatureGoal:
        if isinstance(raw, StructuredFeatureGoal):
            return raw
        relation = getattr(raw, "relation")
        relation_value = getattr(relation, "value", relation)
        return StructuredFeatureGoal(
            feature=getattr(raw, "feature"),
            relation=str(relation_value),
            strength=float(getattr(raw, "strength", 1.0)),
            target=getattr(raw, "target", None),
            low=getattr(raw, "low", None),
            high=getattr(raw, "high", None),
        )

    @staticmethod
    def _goal_expression(goal: StructuredFeatureGoal) -> tuple[str, list[float]]:
        column = f"t.{goal.feature}"
        value = f"(({column} - 50.0) / 150.0)" if goal.feature == "tempo_bpm" else column
        parameters: list[float] = []
        if goal.relation == "prefer_high":
            return value, parameters
        if goal.relation == "prefer_low":
            return f"(1.0 - {value})", parameters
        if goal.relation == "near":
            target = (goal.target - 50.0) / 150.0 if goal.feature == "tempo_bpm" else goal.target
            parameters.append(float(target))
            return f"max(0.0, 1.0 - abs({value} - ?))", parameters
        if goal.relation == "at_least":
            target = (goal.target - 50.0) / 150.0 if goal.feature == "tempo_bpm" else goal.target
            parameters.extend((float(target), float(target)))
            return f"CASE WHEN {value} >= ? THEN 1.0 ELSE max(0.0, 1.0 - (? - {value})) END", parameters
        if goal.relation == "at_most":
            target = (goal.target - 50.0) / 150.0 if goal.feature == "tempo_bpm" else goal.target
            parameters.extend((float(target), float(target)))
            return f"CASE WHEN {value} <= ? THEN 1.0 ELSE max(0.0, 1.0 - ({value} - ?)) END", parameters
        low = (goal.low - 50.0) / 150.0 if goal.feature == "tempo_bpm" else goal.low
        high = (goal.high - 50.0) / 150.0 if goal.feature == "tempo_bpm" else goal.high
        parameters.extend((float(low), float(high), float(low), float(high)))
        return (
            f"CASE WHEN {value} BETWEEN ? AND ? THEN 1.0 "
            f"WHEN {value} < ? THEN max(0.0, 1.0 - (? - {value})) "
            f"ELSE max(0.0, 1.0 - ({value} - ?)) END",
            # The expression has five placeholders; low is repeated twice and
            # high is repeated twice around the BETWEEN pair.
            [float(low), float(high), float(low), float(low), float(high)],
        )

    def structured_search(
        self,
        *,
        genre: str | None = None,
        goals: Sequence[Any] = (),
        limit: int = DEFAULT_TEXT_LIMIT,
    ) -> tuple[CatalogStoreHit, ...]:
        limit = _bounded_limit(limit)
        normalized_genre = " ".join(genre.split()).casefold() if genre else None
        normalized_goals = tuple(self._coerce_goal(goal) for goal in goals)
        if normalized_genre is None and not normalized_goals:
            return ()

        achieved: list[str] = []
        possible: list[str] = []
        params: list[Any] = []
        reasons: list[str] = []
        if normalized_genre is not None:
            achieved.append(
                "4.0 * CASE WHEN EXISTS (SELECT 1 FROM track_genres g "
                "WHERE g.track_id=t.track_id AND g.genre=?) THEN 1.0 ELSE 0.0 END"
            )
            possible.append("4.0")
            params.append(normalized_genre)
            reasons.append("genre")
        for goal in normalized_goals:
            component, component_params = self._goal_expression(goal)
            weight = 0.75 * goal.strength
            confidence = f"coalesce(t.{goal.feature}_confidence, 1.0)"
            achieved.append(
                f"CASE WHEN t.{goal.feature} IS NULL THEN 0.0 "
                f"ELSE {weight!r} * {confidence} * ({component}) END"
            )
            possible.append(
                f"CASE WHEN t.{goal.feature} IS NULL THEN 0.0 ELSE {weight!r} END"
            )
            params.extend(component_params)
            reasons.append(goal.feature)

        achieved_sql = " + ".join(achieved)
        possible_sql = " + ".join(possible)
        sql = f"""
            WITH signals AS (
                SELECT t.track_id,
                       ({achieved_sql}) AS achieved,
                       ({possible_sql}) AS possible
                FROM tracks t
            )
            SELECT track_id, achieved / possible AS structured_score
            FROM signals
            WHERE possible > 0.0
            ORDER BY structured_score DESC, track_id ASC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return tuple(
            CatalogStoreHit(
                track_id=int(row["track_id"]),
                score=max(0.0, min(1.0, float(row["structured_score"]))),
                reasons=tuple(reasons),
            )
            for row in rows
        )

    def get_tracks(self, track_ids: Sequence[int]) -> tuple[StoredTrack, ...]:
        if not track_ids:
            return ()
        if len(track_ids) > MAX_SEARCH_LIMIT:
            raise ValueError(f"get_tracks accepts at most {MAX_SEARCH_LIMIT} IDs")
        requested: list[int] = []
        for track_id in track_ids:
            if isinstance(track_id, bool) or not isinstance(track_id, int) or track_id <= 0:
                raise ValueError("track IDs must be positive integers")
            if track_id not in requested:
                requested.append(track_id)
        placeholders = ",".join("?" for _ in requested)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM tracks WHERE track_id IN ({placeholders})", requested
            ).fetchall()
            genre_rows = connection.execute(
                f"SELECT track_id,genre FROM track_genres WHERE track_id IN ({placeholders}) "
                "ORDER BY track_id,position",
                requested,
            ).fetchall()
            tag_rows = connection.execute(
                f"SELECT track_id,scope,tag FROM track_tags WHERE track_id IN ({placeholders}) "
                "ORDER BY track_id,scope,position",
                requested,
            ).fetchall()
        genres: dict[int, list[str]] = {track_id: [] for track_id in requested}
        tags: dict[int, dict[str, list[str]]] = {
            track_id: {"track": [], "album": [], "artist": []} for track_id in requested
        }
        for row in genre_rows:
            genres[int(row["track_id"])].append(row["genre"])
        for row in tag_rows:
            tags[int(row["track_id"])][row["scope"]].append(row["tag"])

        by_id: dict[int, StoredTrack] = {}
        for row in rows:
            track_id = int(row["track_id"])
            feature_data: dict[str, StoredFeature] = {}
            for feature in TARGET_FEATURES:
                if row[feature] is not None:
                    feature_data[feature] = StoredFeature(
                        value=float(row[feature]),
                        origin=row[f"{feature}_origin"],
                        method_version=row[f"{feature}_method"],
                        confidence=float(row[f"{feature}_confidence"]),
                        interval_low=row[f"{feature}_interval_low"],
                        interval_high=row[f"{feature}_interval_high"],
                    )
            mood = None
            if row["mood_upbeat"] is not None:
                mood = MappingProxyType(
                    {
                        "upbeat": float(row["mood_upbeat"]),
                        "calm": float(row["mood_calm"]),
                        "intense": float(row["mood_intense"]),
                        "somber": float(row["mood_somber"]),
                        "label": row["mood_label"],
                        "confidence": row["mood_confidence"],
                        "method_version": row["mood_method"],
                        "experimental": bool(row["mood_experimental"]),
                    }
                )
            by_id[track_id] = StoredTrack(
                id=track_id,
                catalog_id="fma",
                external_id=row["external_id"],
                title=row["title"],
                artist=row["artist"],
                genre=row["primary_genre"],
                genres=tuple(genres[track_id]),
                mood=None,
                mood_profile=mood,
                energy=row["energy"],
                valence=row["valence"],
                acousticness=row["acousticness"],
                danceability=row["danceability"],
                tempo_bpm=row["tempo_bpm"],
                instrumentalness=row["instrumentalness"],
                description=None,
                tags=tuple(tags[track_id]["track"]),
                album_tags=tuple(tags[track_id]["album"]),
                artist_tags=tuple(tags[track_id]["artist"]),
                contexts=(),
                instruments=(),
                instrumental=None,
                explicit=None,
                era=row["era"],
                track_information=row["track_information"],
                album_information=row["album_information"],
                artist_biography=row["artist_biography"],
                license=row["license"],
                source_url=row["source_url"],
                track_url=row["track_url"],
                artist_url=row["artist_url"],
                album_url=row["album_url"],
                lineage=tuple(json.loads(row["lineage_json"])),
                feature_data=MappingProxyType(feature_data),
            )
        return tuple(by_id[track_id] for track_id in requested if track_id in by_id)

    def get_contract_tracks(self, track_ids: Sequence[int]) -> tuple[Any, ...]:
        return tuple(track.to_contract() for track in self.get_tracks(track_ids))
