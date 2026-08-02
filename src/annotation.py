"""Local-only human annotation primitives for experimental FMA mood work.

This module deliberately contains no Streamlit or network code.  It creates a
stable, genre-stratified audit set and exposes only a small identity/context
view to a rater; model predictions never appear in that view.  Labels remain a
separate local research artifact and cannot promote a production model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit


Quadrant = Literal["upbeat", "calm", "intense", "somber"]
RaterRole = Literal["primary", "audit"]
QUADRANTS: tuple[Quadrant, ...] = ("upbeat", "calm", "intense", "somber")
RATER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
CATALOG_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SAMPLE_SEED = "cadence-fma-mood-audit-v1"


def _clean_text(value: object, *, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    # Keep printable text only and collapse whitespace. FMA text is untrusted
    # display data, even in a local terminal harness.
    cleaned = " ".join("".join(char for char in value if char.isprintable()).split())
    return cleaned[:maximum]


def _safe_http_url(value: object) -> str | None:
    text = _clean_text(value, maximum=1000)
    if not text:
        return None
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        return None
    return text


def _positive_track_id(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("track_id must be a positive integer")
    try:
        track_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("track_id must be a positive integer") from exc
    if track_id <= 0 or str(track_id) != str(value).strip():
        # Do not silently turn 4.8 into track 4 or otherwise alter identity.
        raise ValueError("track_id must be a positive integer")
    return track_id


@dataclass(frozen=True, slots=True)
class AnnotationItem:
    """Prediction-free record shown to a human rater."""

    catalog_id: str
    track_id: int
    title: str
    artist: str
    genre: str
    external_id: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not CATALOG_ID_PATTERN.fullmatch(self.catalog_id):
            raise ValueError("catalog_id must be a canonical catalog identifier")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError("track_id must be a positive integer")
        _positive_track_id(self.track_id)
        for name, value, maximum in (
            ("title", self.title, 200),
            ("artist", self.artist, 200),
            ("genre", self.genre, 80),
        ):
            if not value or _clean_text(value, maximum=maximum) != value:
                raise ValueError(f"{name} must be normalized nonempty text")
        if self.external_id is not None and (
            not self.external_id
            or _clean_text(self.external_id, maximum=160) != self.external_id
        ):
            raise ValueError("external_id must be normalized text when provided")
        if self.source_url is not None and _safe_http_url(self.source_url) != self.source_url:
            raise ValueError("source_url must be a safe HTTP(S) URL")

    @property
    def key(self) -> str:
        return f"{self.catalog_id}:{self.track_id}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MoodAnnotation:
    """One independent human judgment; confidence uses an explicit 1–5 scale."""

    catalog_id: str
    track_id: int
    rater_id: str
    role: RaterRole
    valence: float
    arousal: float
    quadrant: Quadrant
    confidence: int
    recorded_at: str

    def __post_init__(self) -> None:
        if not self.catalog_id or len(self.catalog_id) > 80:
            raise ValueError("catalog_id must contain 1 to 80 characters")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError("track_id must be a positive integer")
        _positive_track_id(self.track_id)
        if not RATER_ID_PATTERN.fullmatch(self.rater_id):
            raise ValueError("rater_id must be a short pseudonymous identifier")
        if self.role not in ("primary", "audit"):
            raise ValueError("role must be primary or audit")
        for name, value in (("valence", self.valence), ("arousal", self.arousal)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number between 0 and 1")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be a real number between 0 and 1")
        if self.quadrant not in QUADRANTS:
            raise ValueError(f"quadrant must be one of {', '.join(QUADRANTS)}")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int)
            or self.confidence not in range(1, 6)
        ):
            raise ValueError("confidence must be an integer from 1 to 5")
        try:
            parsed = datetime.fromisoformat(self.recorded_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("recorded_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")

    @classmethod
    def create(
        cls,
        *,
        item: AnnotationItem,
        rater_id: str,
        role: RaterRole,
        valence: float,
        arousal: float,
        quadrant: Quadrant,
        confidence: int,
        now: datetime | None = None,
    ) -> MoodAnnotation:
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        return cls(
            catalog_id=item.catalog_id,
            track_id=item.track_id,
            rater_id=rater_id,
            role=role,
            valence=valence,
            arousal=arousal,
            quadrant=quadrant,
            confidence=confidence,
            recorded_at=timestamp,
        )

    @property
    def track_key(self) -> str:
        return f"{self.catalog_id}:{self.track_id}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnnotationReadiness:
    """Progress report for a possible *future* calibration decision.

    ``status`` is intentionally fixed to ``experimental``. Reaching the sample
    thresholds is evidence for a later reviewed promotion script; it does not
    let this harness promote itself.
    """

    status: Literal["experimental"]
    total_annotations: int
    unique_tracks: int
    primary_tracks: int
    audit_annotations: int
    independent_audit_pairs: int
    quadrant_agreement: float | None
    ready_for_future_review: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def annotation_item_from_record(record: Mapping[str, object]) -> AnnotationItem:
    """Project a catalog row into an explicit prediction-free annotation view."""
    raw_track_id = record.get("track_id", record.get("id"))
    track_id = _positive_track_id(raw_track_id)
    catalog_id = _clean_text(record.get("catalog_id", "fma"), maximum=80)
    title = _clean_text(record.get("title"), maximum=200)
    artist = _clean_text(record.get("artist"), maximum=200)
    if not catalog_id or not title or not artist:
        raise ValueError("annotation records require catalog_id, title, and artist")

    raw_genre = record.get("genre", record.get("genre_top", "unknown"))
    if isinstance(raw_genre, Sequence) and not isinstance(raw_genre, (str, bytes)):
        raw_genre = next(iter(raw_genre), "unknown")
    genre = _clean_text(raw_genre, maximum=80) or "unknown"
    external_id = _clean_text(record.get("external_id"), maximum=160) or None
    source_url = _safe_http_url(record.get("source_url", record.get("track_url")))

    # No splat/copy of the source mapping: fields such as mood_profile,
    # predicted_energy, and model_confidence cannot leak through accidentally.
    return AnnotationItem(
        catalog_id=catalog_id,
        track_id=track_id,
        title=title,
        artist=artist,
        genre=genre,
        external_id=external_id,
        source_url=source_url,
    )


def _stable_digest(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def select_annotation_sample(
    records: Iterable[Mapping[str, object]],
    *,
    size: int = 300,
    seed: str = _SAMPLE_SEED,
) -> tuple[AnnotationItem, ...]:
    """Select a deterministic, round-robin genre-stratified sample.

    Rows are ordered by a stable identity hash, never by their source order.
    Duplicate catalog/track identities are rejected because allowing them would
    silently shrink the number of independent tracks in the audit set.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    if not seed:
        raise ValueError("seed cannot be empty")

    groups: dict[str, list[AnnotationItem]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        item = annotation_item_from_record(record)
        if item.key in seen:
            raise ValueError(f"duplicate annotation identity: {item.key}")
        seen.add(item.key)
        groups[item.genre.lower()].append(item)

    for items in groups.values():
        items.sort(key=lambda item: _stable_digest(item.key, seed))
    strata = sorted(groups, key=lambda name: _stable_digest(name, seed))

    selected: list[AnnotationItem] = []
    offsets = {name: 0 for name in strata}
    while len(selected) < size:
        added = False
        for name in strata:
            offset = offsets[name]
            if offset < len(groups[name]):
                selected.append(groups[name][offset])
                offsets[name] += 1
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    return tuple(selected)


def append_annotation(path: str | Path, annotation: MoodAnnotation) -> None:
    """Append one validated label, refusing duplicate track/rater judgments."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        for existing in load_annotations(destination):
            if (
                existing.track_key == annotation.track_key
                and existing.rater_id == annotation.rater_id
            ):
                raise ValueError(
                    f"rater {annotation.rater_id!r} already labeled {annotation.track_key}"
                )
    line = json.dumps(annotation.to_dict(), sort_keys=True, separators=(",", ":"))
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_annotations(path: str | Path) -> tuple[MoodAnnotation, ...]:
    """Load and validate every JSONL annotation; malformed lines fail closed."""
    source = Path(path)
    if not source.exists():
        return ()
    annotations: list[MoodAnnotation] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                annotations.append(MoodAnnotation(**payload))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid annotation on line {line_number}") from exc
    return tuple(annotations)


def assess_annotation_readiness(
    annotations: Iterable[MoodAnnotation],
    *,
    target_primary_tracks: int = 300,
    target_independent_audits: int = 60,
) -> AnnotationReadiness:
    """Measure coverage/agreement without changing experimental status."""
    for name, threshold in (
        ("target_primary_tracks", target_primary_tracks),
        ("target_independent_audits", target_independent_audits),
    ):
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
            raise ValueError(f"{name} must be a positive integer")
    values = tuple(annotations)
    primary_by_track: dict[str, list[MoodAnnotation]] = defaultdict(list)
    audit_values: list[MoodAnnotation] = []
    for annotation in values:
        if annotation.role == "primary":
            primary_by_track[annotation.track_key].append(annotation)
        else:
            audit_values.append(annotation)

    paired: list[tuple[MoodAnnotation, MoodAnnotation]] = []
    for audit in audit_values:
        candidates = [
            primary
            for primary in primary_by_track.get(audit.track_key, ())
            if primary.rater_id != audit.rater_id
        ]
        if candidates:
            paired.append((candidates[0], audit))
    agreement = (
        sum(primary.quadrant == audit.quadrant for primary, audit in paired) / len(paired)
        if paired
        else None
    )
    unique_tracks = len({annotation.track_key for annotation in values})
    primary_tracks = len(primary_by_track)
    ready = (
        primary_tracks >= target_primary_tracks
        and len(paired) >= target_independent_audits
    )
    return AnnotationReadiness(
        status="experimental",
        total_annotations=len(values),
        unique_tracks=unique_tracks,
        primary_tracks=primary_tracks,
        audit_annotations=len(audit_values),
        independent_audit_pairs=len(paired),
        quadrant_agreement=agreement,
        ready_for_future_review=ready,
    )
