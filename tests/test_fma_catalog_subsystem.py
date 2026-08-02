"""Focused, network-free tests for the Phase 5 FMA catalog foundation."""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import closing
from pathlib import Path
from urllib.request import Request

import pytest

from src.catalog_artifacts import (
    ArtifactCandidate,
    CatalogArtifactResolver,
    CatalogUnavailableError,
    download_verified_artifact,
    download_verified_gzip_catalog,
    package_catalog_gzip,
)
from src.etl.fma import (
    FmaBuildProfile,
    FmaSourcePaths,
    build_fma_catalog,
    iter_normalized_tracks,
    prepare_model_matrix,
)
from src.etl.integrity import (
    ChecksumMismatchError,
    UnsafeArchiveError,
    safe_extract_zip,
    sha256_file,
)
from src.fma_store import FmaCatalogStore, StructuredFeatureGoal
from src.fma_retrieval import FmaRetriever
from src.contracts import FeatureGoal, FeatureRelation, MusicIntent


TRACK_COLUMNS = (
    ("track", "title"),
    ("artist", "name"),
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


def _write_multi_csv(
    path: Path, columns: tuple[tuple[str, ...], ...], rows: list[tuple[object, ...]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        depth = len(columns[0])
        for level in range(depth):
            writer.writerow([""] + [column[level] for column in columns])
        writer.writerow(["track_id"] + [""] * len(columns))
        writer.writerows(rows)


def _source_bundle(root: Path) -> FmaSourcePaths:
    tracks = root / "tracks.csv"
    _write_multi_csv(
        tracks,
        TRACK_COLUMNS,
        [
            (
                1, "Night Signal", "Alpha", "Rock", "[1]", "[1, 2]",
                "<p>Midnight guitars</p><script>ignore me</script>", "['night', 'guitar']",
                "CC BY", "2014-01-01", "2014-02-02", "<p>A nocturnal album</p>",
                "['dark']", "<p>Alpha biography</p>", "['independent']",
                "https://alpha.example", "",
            ),
            (
                2, "Morning Field", "Beta", "Folk", "[2]", "[2]", "",
                "['morning']", "CC BY-NC", "2008-03-01", "", "", "[]",
                "<p>Beta biography</p>", "['acoustic']", "", "https://example.org/beta",
            ),
            (
                3, "Estimated Pulse", "Gamma", "Rock", "[1]", "[1]",
                "A synthetic-feature fixture, not authored mood", "[]", "", "2020-01-01",
                "", "", "[]", "", "[]", "", "",
            ),
            (
                4, "No Artist", "", "Rock", "[1]", "[1]", "", "[]", "", "", "",
                "", "[]", "", "[]", "", "",
            ),
            (
                5, "", "Delta", "Folk", "[2]", "[2]", "", "[]", "", "", "",
                "", "[]", "", "[]", "", "",
            ),
        ],
    )
    genres = root / "genres.csv"
    genres.write_text(
        "genre_id,#tracks,parent,title,top_level\n"
        "1,2,0,Rock,1\n"
        "2,2,0,Folk,2\n",
        encoding="utf-8",
    )
    echo_columns = tuple(
        ("echonest", "audio_features", name)
        for name in (
            "acousticness", "danceability", "energy", "instrumentalness",
            "liveness", "speechiness", "tempo", "valence",
        )
    )
    echonest = root / "echonest.csv"
    _write_multi_csv(
        echonest,
        echo_columns,
        [
            (1, 0.1, 0.8, 0.9, 0.2, 0.1, 0.1, 150.0, 0.8),
            (2, 0.9, 0.2, 0.1, 0.8, 0.1, 0.1, 70.0, 0.7),
        ],
    )
    predictions = root / "predictions.jsonl"
    lines = [
        # Echo Nest must win over this released prediction for track 1.
        '{"track_id":1,"feature":"energy","value":0.05,"confidence":0.9,'
        '"interval_low":0.0,"interval_high":0.1,"model_version":"fixture-v1","released":true}',
    ]
    predicted = {
        "energy": (0.99, 0.2, 0.8, 1.0),
        "valence": (0.2, 0.3, 0.0, 0.4),
        "acousticness": (0.3, 0.5, 0.1, 0.5),
        "danceability": (0.8, 0.5, 0.6, 1.0),
        "tempo_bpm": (145.0, 0.5, 130.0, 160.0),
        "instrumentalness": (0.7, 0.5, 0.5, 0.9),
    }
    for feature, (value, confidence, low, high) in predicted.items():
        lines.append(
            "{" + f'"track_id":3,"feature":"{feature}","value":{value},'
            f'"confidence":{confidence},"interval_low":{low},"interval_high":{high},'
            '"model_version":"fixture-v1","released":true}'
        )
    predictions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FmaSourcePaths(
        tracks=tracks,
        genres=genres,
        echonest=echonest,
        predictions=predictions,
    )


def _with_feature_fixture(paths: FmaSourcePaths, root: Path) -> FmaSourcePaths:
    features = root / "features.csv"
    _write_multi_csv(
        features,
        (
            ("chroma_cens", "mean", "01"),
            ("spectral_centroid", "std", "01"),
        ),
        [
            (1, 0.1, 10.0),
            (2, 0.2, 20.0),
            (3, 0.3, 30.0),
        ],
    )
    return FmaSourcePaths(
        tracks=paths.tracks,
        genres=paths.genres,
        echonest=paths.echonest,
        features=features,
        predictions=paths.predictions,
    )


@pytest.fixture()
def built_full(tmp_path: Path):
    sources = _source_bundle(tmp_path)
    return build_fma_catalog(sources, tmp_path / "full.sqlite")


def test_multilevel_parser_preserves_scope_lineage_and_unknowns(tmp_path: Path):
    items = list(iter_normalized_tracks(_source_bundle(tmp_path), chunk_size=2))
    tracks = [item for item in items if hasattr(item, "title")]
    quarantined = [item for item in items if not hasattr(item, "title")]

    assert len(tracks) == 3
    assert len(quarantined) == 2
    first = tracks[0]
    assert first.genres == ("rock", "folk")
    assert first.track_information == "Midnight guitars"
    assert first.album_information == "A nocturnal album"
    assert first.artist_biography == "Alpha biography"
    assert first.track_tags == ("night", "guitar")
    assert first.features["energy"].value == 0.9  # direct source beats model
    assert first.features["energy"].origin == "echonest_computed"
    assert first.track_url is None  # no synthetic FMA URL
    assert first.mood_profile is not None
    assert first.mood_profile["experimental"] is True
    third = tracks[2]
    assert third.features["energy"].origin == "model_estimated"
    assert third.features["energy"].confidence == 0.2


def test_build_is_byte_deterministic_and_manifest_is_verifiable(tmp_path: Path):
    sources = _source_bundle(tmp_path)
    first = build_fma_catalog(sources, tmp_path / "one.sqlite")
    second = build_fma_catalog(sources, tmp_path / "two.sqlite")

    assert sha256_file(first.database_path) == sha256_file(second.database_path)
    assert first.manifest.artifact_sha256 == sha256_file(first.database_path)
    assert first.manifest.accepted_count == 3
    assert first.manifest.quarantined_count == 2
    assert first.manifest.field_coverage["energy_estimated"] == pytest.approx(1 / 3)
    assert first.manifest_path.with_suffix(".json.sha256").is_file()


def test_model_matrix_flattens_explicit_three_row_librosa_header(tmp_path: Path):
    sources = _with_feature_fixture(_source_bundle(tmp_path), tmp_path)
    one = prepare_model_matrix(
        sources, tmp_path / "matrix-one.csv", expected_feature_count=2, chunk_size=2
    )
    two = prepare_model_matrix(
        sources, tmp_path / "matrix-two.csv", expected_feature_count=2, chunk_size=1
    )

    assert one.row_count == 3
    assert one.feature_count == 2
    assert one.sha256 == two.sha256
    header = one.output_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [
        "track_id",
        "artist",
        "librosa__chroma_cens__mean__01",
        "librosa__spectral_centroid__std__01",
        "energy",
        "valence",
        "acousticness",
        "danceability",
        "tempo_bpm",
        "instrumentalness",
    ]
    rows = list(csv.DictReader(one.output_path.open(encoding="utf-8")))
    assert rows[0]["artist"] == "Alpha"
    assert rows[0]["energy"] == "0.9"
    assert rows[2]["energy"] == ""


def test_model_matrix_rejects_feature_header_count_drift(tmp_path: Path):
    sources = _with_feature_fixture(_source_bundle(tmp_path), tmp_path)
    with pytest.raises(ValueError, match="518 Librosa columns"):
        prepare_model_matrix(sources, tmp_path / "bad.csv")


def test_lite_profile_is_balanced_echo_only_and_deterministic(tmp_path: Path):
    sources = _source_bundle(tmp_path)
    result = build_fma_catalog(
        sources,
        tmp_path / "lite.sqlite",
        profile=FmaBuildProfile(edition="lite", lite_size=2),
    )
    store = FmaCatalogStore(result.database_path, result.manifest_path)

    assert store.count == 2
    tracks = store.get_tracks(sorted(store.valid_track_ids()))
    assert {track.genre for track in tracks} == {"rock", "folk"}
    assert all(track.feature_data["energy"].origin == "echonest_computed" for track in tracks)


def test_store_searches_scoped_text_and_structured_values(built_full):
    store = FmaCatalogStore(built_full.database_path, built_full.manifest_path)

    assert store.text_search('midnight" OR *')[0].track_id == 1
    assert store.text_search("biography")[0].track_id in {1, 2}
    high = store.structured_search(
        goals=(StructuredFeatureGoal(feature="energy", relation="prefer_high"),),
        limit=3,
    )
    assert [hit.track_id for hit in high] == [1, 3, 2]
    # Track 3 has a higher point estimate but its low calibrated confidence keeps
    # it below the direct Echo Nest value.
    assert high[0].score > high[1].score
    rock = store.structured_search(genre="ROCK", limit=3)
    assert {hit.track_id for hit in rock[:2]} == {1, 3}


def test_store_materializes_requested_order_and_read_only_connection(built_full):
    store = FmaCatalogStore(built_full.database_path, built_full.manifest_path)
    tracks = store.get_tracks([3, 1, 999])

    assert [track.id for track in tracks] == [3, 1]
    assert tracks[0].mood is None
    assert tracks[0].explicit is None
    assert tracks[0].instrumental is None
    assert tracks[0].description is None
    assert tracks[0].feature_data["energy"].origin == "model_estimated"
    assert store.descriptor.catalog_id == "fma"
    assert store.descriptor.accepted_count == 3
    with closing(store._connect()) as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM tracks")


def test_sqlite_runtime_import_does_not_import_pandas():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'pandas' or name.startswith('pandas.'):
        raise AssertionError('runtime imported pandas')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import src.fma_store
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_fma_retriever_returns_namespaced_lazy_hits_and_rrf_union(built_full):
    retriever = FmaRetriever(FmaCatalogStore(built_full.database_path, built_full.manifest_path))
    text = retriever.search("midnight guitars", k=1)
    assert text.hits[0].track.id == 1
    assert text.hits[0].source_id == "catalog:fma:1"
    assert "track_information" in text.hits[0].fields_used

    result = retriever.search_with_intent(
        MusicIntent(
            query="night rock energetic",
            genre="rock",
            feature_goals=(
                FeatureGoal(
                    feature="energy",
                    relation=FeatureRelation.PREFER_HIGH,
                    cue_id="test-energy-high",
                ),
            ),
            limit=3,
        )
    )
    assert result.hits[0].track.id == 1
    assert {hit.track.id for hit in result.hits} == {1, 2, 3}
    assert result.hits[0].fusion_version == "rrf:text=0.4,structured=0.6,k=60;fma-v1"
    assert all(hit.source_id.startswith("catalog:fma:") for hit in result.hits)

    unsupported = retriever.search("rock", instrumental_only=True)
    assert unsupported.hits == ()
    assert unsupported.filters_applied == ("instrumental_only:unsupported_unknown",)


def test_safe_zip_extraction_checks_digest_and_rejects_traversal(tmp_path: Path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fma_metadata/tracks.csv", "data")
        bundle.writestr("fma_metadata/features.csv", "large-unused-fixture")
    destination = tmp_path / "safe"
    extracted = safe_extract_zip(
        archive,
        destination,
        expected_sha256=sha256_file(archive),
        selected_files=frozenset({"tracks.csv"}),
    )
    assert extracted == (destination / "fma_metadata" / "tracks.csv",)
    assert extracted[0].read_text() == "data"
    assert not (destination / "fma_metadata" / "features.csv").exists()
    with pytest.raises(ChecksumMismatchError):
        safe_extract_zip(archive, tmp_path / "wrong", expected_sha256="0" * 64)

    malicious = tmp_path / "malicious.zip"
    with zipfile.ZipFile(malicious, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(malicious, tmp_path / "escaped")
    assert not (tmp_path / "escape.txt").exists()


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_resolver_uses_local_full_then_verified_release_then_lite(tmp_path: Path, built_full):
    lite = build_fma_catalog(
        _source_bundle(tmp_path),
        tmp_path / "lite.sqlite",
        profile=FmaBuildProfile(edition="lite", lite_size=2),
    )
    local = ArtifactCandidate(built_full.database_path, built_full.manifest_path, "local-full")
    fallback = ArtifactCandidate(lite.database_path, lite.manifest_path, "bundled-lite")
    assert CatalogArtifactResolver(local_full=local, bundled_lite=fallback).resolve().source == "local-full"

    distribution = tmp_path / "full.sqlite.gz"
    release_manifest = tmp_path / "release.manifest.json"
    packaged = package_catalog_gzip(
        built_full.database_path,
        built_full.manifest_path,
        distribution,
        release_manifest_path=release_manifest,
    )
    payload = distribution.read_bytes()
    cache = tmp_path / "downloaded.sqlite"
    resolved = CatalogArtifactResolver(
        local_full=None,
        bundled_lite=fallback,
        release_url="https://example.invalid/fma.sqlite.gz",
        release_manifest_path=release_manifest,
        release_cache_path=cache,
        opener=lambda _request, _timeout: _FakeResponse(payload),
    ).resolve()
    assert resolved.source == "release-cache"
    assert sha256_file(cache) == built_full.manifest.artifact_sha256

    corrupt = CatalogArtifactResolver(
        local_full=ArtifactCandidate(tmp_path / "missing.sqlite", built_full.manifest_path, "bad"),
        bundled_lite=fallback,
        release_url="https://example.invalid/fma.sqlite.gz",
        release_manifest_path=release_manifest,
        release_cache_path=tmp_path / "corrupt.sqlite",
        opener=lambda _request, _timeout: _FakeResponse(b"corrupt"),
    ).resolve()
    assert corrupt.source == "bundled-lite"
    assert corrupt.is_fallback
    assert corrupt.warnings


def test_release_gzip_is_deterministic_and_expansion_is_bounded(tmp_path: Path, built_full):
    first = tmp_path / "one.sqlite.gz"
    second = tmp_path / "two.sqlite.gz"
    first_manifest = tmp_path / "one.manifest.json"
    second_manifest = tmp_path / "two.manifest.json"
    one = package_catalog_gzip(
        built_full.database_path,
        built_full.manifest_path,
        first,
        release_manifest_path=first_manifest,
    )
    two = package_catalog_gzip(
        built_full.database_path,
        built_full.manifest_path,
        second,
        release_manifest_path=second_manifest,
    )
    assert one.distribution_sha256 == two.distribution_sha256
    assert first.read_bytes()[4:8] == b"\x00\x00\x00\x00"  # gzip MTIME

    with pytest.raises(ValueError, match="expanded catalog exceeds"):
        download_verified_gzip_catalog(
            "https://example.invalid/fma.sqlite.gz",
            tmp_path / "bounded.sqlite",
            distribution_sha256=one.distribution_sha256,
            artifact_sha256=one.artifact_sha256,
            max_database_bytes=built_full.database_path.stat().st_size - 1,
            opener=lambda _request, _timeout: _FakeResponse(first.read_bytes()),
        )


def test_download_rejects_non_https_without_opening(tmp_path: Path):
    called = False

    def opener(_request: Request, _timeout: float):
        nonlocal called
        called = True
        return _FakeResponse(b"")

    with pytest.raises(ValueError, match="HTTPS"):
        download_verified_artifact(
            "http://example.invalid/catalog.sqlite",
            tmp_path / "catalog.sqlite",
            hashlib.sha256(b"").hexdigest(),
            opener=opener,
        )
    assert not called


def test_resolver_fails_closed_when_even_lite_is_invalid(tmp_path: Path, built_full):
    resolver = CatalogArtifactResolver(
        local_full=None,
        bundled_lite=ArtifactCandidate(
            tmp_path / "missing.sqlite", built_full.manifest_path, "bundled-lite"
        ),
    )
    with pytest.raises(CatalogUnavailableError, match="no verified"):
        resolver.resolve()
