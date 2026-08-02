"""Offline specialized feature modeling for the FMA catalog.

The serving application does not import this module.  An offline build joins
FMA's Librosa statistics with the smaller Echo Nest overlap, trains one model
per target, applies release and per-row abstention gates, and writes plain JSONL
predictions for the catalog ETL to consume.

Important boundaries:

* artists, not tracks, are split 70/15/15 to prevent artist leakage;
* Echo Nest targets are machine-computed reference values, not human truth;
* each target has its own release decision;
* an unreleased or uncertain prediction is ``null``, never a guessed default;
* no pickle/joblib model is loaded by the product runtime.

Pandas and scikit-learn are imported lazily.  This keeps the normal Cadence CLI
and Streamlit environment lean; install ``requirements-ml.txt`` only for an
offline model build.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal


SplitName = Literal["train", "calibration", "test"]
MODEL_FAMILY_VERSION: Final = "cadence-fma-hgb-v1"
PREDICTION_SCHEMA_VERSION: Final = 1
DEFAULT_SPLIT_SEED: Final = "cadence-fma-artist-split-v1"
MAX_MISSING_FEATURE_FRACTION: Final = 0.20


class ModelingDataError(ValueError):
    """The prepared training table cannot support an honest model build."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Release policy for one Echo Nest-compatible target."""

    name: str
    low: float
    high: float
    retained_mae_limit: float
    minimum_improvement: float = 0.05
    minimum_interval_coverage: float = 0.75
    maximum_interval_coverage: float = 0.90
    minimum_retained_coverage: float = 0.30

    @property
    def span(self) -> float:
        return self.high - self.low


FEATURE_SPECS: Final[Mapping[str, FeatureSpec]] = MappingProxyType(
    {
        name: FeatureSpec(name=name, low=0.0, high=1.0, retained_mae_limit=0.15)
        for name in (
            "energy",
            "valence",
            "acousticness",
            "danceability",
            "instrumentalness",
        )
    }
    | {
        "tempo_bpm": FeatureSpec(
            name="tempo_bpm", low=50.0, high=200.0, retained_mae_limit=15.0
        )
    }
)


@dataclass(frozen=True, slots=True)
class FeaturePrediction:
    """Exact JSONL wire shape consumed by the FMA catalog builder."""

    track_id: int
    feature: str
    value: float | None
    confidence: float | None
    interval_low: float | None
    interval_high: float | None
    model_version: str
    released: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelingRun:
    """Serializable report plus baked per-track predictions."""

    report: Mapping[str, object]
    predictions: tuple[FeaturePrediction, ...]


@dataclass(slots=True)
class _FittedTarget:
    spec: FeatureSpec
    feature_columns: tuple[str, ...]
    point_model: Any
    lower_model: Any
    upper_model: Any
    location: Any
    scale: Any
    ood_threshold: float
    width_threshold: float | None
    model_version: str
    global_released: bool
    report: dict[str, object]


def _stable_digest(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _normal_artist(value: object) -> str:
    if not isinstance(value, str):
        raise ModelingDataError("artist values must be nonempty strings")
    normalized = " ".join(value.split()).casefold()
    if not normalized:
        raise ModelingDataError("artist values must be nonempty strings")
    return normalized


def _split_counts(group_count: int) -> tuple[int, int, int]:
    """Allocate groups as closely as possible to 70/15/15, with no empty split."""
    if group_count < 3:
        raise ModelingDataError("at least three distinct artists are required")
    train = round(group_count * 0.70)
    calibration = round(group_count * 0.15)
    train = max(1, train)
    calibration = max(1, calibration)
    test = group_count - train - calibration
    if test < 1:
        train -= 1 - test
        test = 1
    return train, calibration, test


def artist_group_split_assignments(
    artists: Iterable[object], *, seed: str = DEFAULT_SPLIT_SEED
) -> tuple[SplitName, ...]:
    """Return deterministic 70/15/15 assignments with no artist leakage."""
    if not seed:
        raise ValueError("split seed cannot be empty")
    normalized = tuple(_normal_artist(value) for value in artists)
    groups = sorted(set(normalized), key=lambda value: _stable_digest(value, seed))
    train_count, calibration_count, _ = _split_counts(len(groups))
    assignment: dict[str, SplitName] = {}
    for index, artist in enumerate(groups):
        if index < train_count:
            assignment[artist] = "train"
        elif index < train_count + calibration_count:
            assignment[artist] = "calibration"
        else:
            assignment[artist] = "test"
    return tuple(assignment[artist] for artist in normalized)


def _require_ml() -> tuple[Any, ...]:
    """Import offline dependencies with a targeted installation error."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise RuntimeError(
            "offline modeling dependencies are missing; install requirements-ml.txt"
        ) from exc
    return (
        np,
        pd,
        TransformedTargetRegressor,
        DummyRegressor,
        HistGradientBoostingRegressor,
        SimpleImputer,
        Ridge,
        mean_absolute_error,
        r2_score,
        make_pipeline,
        StandardScaler,
    )


def _positive_track_id(value: object) -> int:
    if isinstance(value, bool):
        raise ModelingDataError("track IDs must be positive integers")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingDataError("track IDs must be positive integers") from exc
    try:
        same_numeric_value = float(number) == float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        same_numeric_value = False
    if number <= 0 or not same_numeric_value:
        raise ModelingDataError("track IDs must be positive integers")
    return number


def _model_version(
    target: str,
    feature_columns: Sequence[str],
    *,
    split_seed: str,
    random_seed: int,
) -> str:
    payload = json.dumps(
        {
            "family": MODEL_FAMILY_VERSION,
            "target": target,
            "features": list(feature_columns),
            "split": split_seed,
            "random_seed": random_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{MODEL_FAMILY_VERSION}-{target}-{suffix}"


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _improvement(model_mae: float, baseline_mae: float) -> float:
    if baseline_mae <= 0.0:
        # Keep reports strict-JSON serializable even in the degenerate case of
        # a perfect baseline.
        return 0.0 if model_mae <= 0.0 else -1.0
    return (baseline_mae - model_mae) / baseline_mae


def _fit_ood_reference(np: Any, numeric_frame: Any) -> tuple[Any, Any]:
    values = numeric_frame.to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        location = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        fallback = np.nanstd(values, axis=0)
    location = np.where(np.isfinite(location), location, 0.0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return location, scale


def _ood_scores(np: Any, numeric_frame: Any, location: Any, scale: Any) -> tuple[Any, Any]:
    values = numeric_frame.to_numpy(dtype=float)
    missing_fraction = np.mean(~np.isfinite(values), axis=1)
    filled = np.where(np.isfinite(values), values, location)
    robust_z = np.abs((filled - location) / scale)
    # A mean clipped robust distance stays interpretable with 518 dimensions and
    # does not mark a row OOD merely because one harmless statistic is unusual.
    scores = np.mean(np.minimum(robust_z, 20.0), axis=1)
    return scores, missing_fraction


def _calibrate_width_threshold(
    np: Any,
    y_true: Any,
    point: Any,
    lower: Any,
    upper: Any,
    *,
    mae_limit: float,
    eligible: Any,
) -> float | None:
    widths = np.maximum(0.0, upper - lower)
    valid = eligible & np.isfinite(widths) & np.isfinite(point) & np.isfinite(y_true)
    candidates = np.unique(widths[valid])
    if candidates.size == 0:
        return None
    chosen: float | None = None
    for threshold in candidates:
        retained = valid & (widths <= threshold)
        if not np.any(retained):
            continue
        mae = float(np.mean(np.abs(y_true[retained] - point[retained])))
        if mae <= mae_limit:
            # Candidates are ascending; retaining this value makes it the widest
            # accepted interval so far.
            chosen = float(threshold)
    return chosen


def _raw_predictions(np: Any, model: Any, frame: Any) -> Any:
    """Return estimator output without hiding invalid values by clipping.

    Domain validation is an abstention gate below. Clipping would turn an
    impossible prediction into plausible-looking evidence.
    """
    return np.asarray(model.predict(frame), dtype=float)


def _fit_target(
    frame: Any,
    *,
    feature_columns: tuple[str, ...],
    spec: FeatureSpec,
    artist_column: str,
    split_column: str,
    split_seed: str,
    random_seed: int,
) -> _FittedTarget:
    (
        np,
        pd,
        _TransformedTargetRegressor,
        DummyRegressor,
        HistGradientBoostingRegressor,
        SimpleImputer,
        Ridge,
        mean_absolute_error,
        r2_score,
        make_pipeline,
        StandardScaler,
    ) = _require_ml()

    raw_target = frame[spec.name]
    target = pd.to_numeric(raw_target, errors="coerce")
    raw_present = raw_target.notna()
    if raw_target.dtype == object:
        raw_present &= raw_target.astype(str).str.strip().ne("")
    invalid_target = raw_present & (
        target.isna() | ~target.between(spec.low, spec.high)
    )
    if invalid_target.any():
        raise ModelingDataError(
            f"{spec.name} contains {int(invalid_target.sum())} corrupt/out-of-range targets"
        )
    observed = target.notna() & target.between(spec.low, spec.high)
    subsets: dict[str, Any] = {
        name: frame.loc[observed & (frame[split_column] == name)]
        for name in ("train", "calibration", "test")
    }
    if any(len(part) < 5 for part in subsets.values()):
        counts = {name: len(part) for name, part in subsets.items()}
        raise ModelingDataError(
            f"{spec.name} needs at least five observed rows in every split; got {counts}"
        )
    if subsets["train"][artist_column].nunique() < 3:
        raise ModelingDataError(f"{spec.name} needs at least three training artists")

    def numeric_features(part: Any) -> Any:
        return part.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")

    x_train = numeric_features(subsets["train"])
    x_calibration = numeric_features(subsets["calibration"])
    x_test = numeric_features(subsets["test"])
    y_train = target.loc[subsets["train"].index].to_numpy(dtype=float)
    y_calibration = target.loc[subsets["calibration"].index].to_numpy(dtype=float)
    y_test = target.loc[subsets["test"].index].to_numpy(dtype=float)

    point_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=150,
            max_leaf_nodes=31,
            learning_rate=0.06,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=random_seed,
        ),
    )
    lower_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            max_iter=150,
            max_leaf_nodes=31,
            learning_rate=0.06,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=random_seed + 1,
        ),
    )
    upper_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            max_iter=150,
            max_leaf_nodes=31,
            learning_rate=0.06,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=random_seed + 2,
        ),
    )
    dummy_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        DummyRegressor(strategy="median"),
    )
    ridge_model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    for model in (point_model, lower_model, upper_model, dummy_model, ridge_model):
        model.fit(x_train, y_train)

    test_point = _raw_predictions(np, point_model, x_test)
    test_lower = _raw_predictions(np, lower_model, x_test)
    test_upper = _raw_predictions(np, upper_model, x_test)
    interval_low = np.minimum(test_lower, test_upper)
    interval_high = np.maximum(test_lower, test_upper)
    dummy_test = _raw_predictions(np, dummy_model, x_test)
    ridge_test = _raw_predictions(np, ridge_model, x_test)

    point_mae = float(mean_absolute_error(y_test, test_point))
    dummy_mae = float(mean_absolute_error(y_test, dummy_test))
    ridge_mae = float(mean_absolute_error(y_test, ridge_test))
    r2 = float(r2_score(y_test, test_point)) if len(y_test) >= 2 else 0.0
    interval_coverage = float(
        np.mean((y_test >= interval_low) & (y_test <= interval_high))
    )

    location, scale = _fit_ood_reference(np, x_train)
    calibration_ood, calibration_missing = _ood_scores(
        np, x_calibration, location, scale
    )
    finite_calibration_ood = calibration_ood[np.isfinite(calibration_ood)]
    ood_threshold = (
        float(np.quantile(finite_calibration_ood, 0.99))
        if finite_calibration_ood.size
        else 0.0
    )
    calibration_point = _raw_predictions(np, point_model, x_calibration)
    calibration_lower = _raw_predictions(np, lower_model, x_calibration)
    calibration_upper = _raw_predictions(np, upper_model, x_calibration)
    calibration_low = np.minimum(calibration_lower, calibration_upper)
    calibration_high = np.maximum(calibration_lower, calibration_upper)
    calibration_eligible = (
        (calibration_ood <= ood_threshold)
        & (calibration_missing <= MAX_MISSING_FEATURE_FRACTION)
        & (calibration_point >= spec.low)
        & (calibration_point <= spec.high)
        & (calibration_low >= spec.low)
        & (calibration_high <= spec.high)
    )
    width_threshold = _calibrate_width_threshold(
        np,
        y_calibration,
        calibration_point,
        calibration_low,
        calibration_high,
        mae_limit=spec.retained_mae_limit,
        eligible=calibration_eligible,
    )

    missing_target = target.isna()
    missing_frame = frame.loc[missing_target]
    if len(missing_frame):
        x_missing = numeric_features(missing_frame)
        missing_point = _raw_predictions(np, point_model, x_missing)
        missing_lower = _raw_predictions(np, lower_model, x_missing)
        missing_upper = _raw_predictions(np, upper_model, x_missing)
        missing_low = np.minimum(missing_lower, missing_upper)
        missing_high = np.maximum(missing_lower, missing_upper)
        missing_width = missing_high - missing_low
        missing_ood, missing_fraction = _ood_scores(np, x_missing, location, scale)
        retained = (
            np.isfinite(missing_point)
            & np.isfinite(missing_low)
            & np.isfinite(missing_high)
            & (missing_ood <= ood_threshold)
            & (missing_fraction <= MAX_MISSING_FEATURE_FRACTION)
            & (missing_point >= spec.low)
            & (missing_point <= spec.high)
            & (missing_low >= spec.low)
            & (missing_high <= spec.high)
        )
        if width_threshold is None:
            retained &= False
        else:
            retained &= missing_width <= width_threshold
        retained_coverage = float(np.mean(retained))
    else:
        retained_coverage = 0.0

    dummy_improvement = _improvement(point_mae, dummy_mae)
    ridge_improvement = _improvement(point_mae, ridge_mae)
    gates = {
        "beats_dummy_by_5pct": dummy_improvement >= spec.minimum_improvement,
        "beats_ridge_by_5pct": ridge_improvement >= spec.minimum_improvement,
        "interval_coverage_75_to_90pct": (
            spec.minimum_interval_coverage
            <= interval_coverage
            <= spec.maximum_interval_coverage
        ),
        "retains_at_least_30pct": retained_coverage >= spec.minimum_retained_coverage,
        "calibrated_width_available": width_threshold is not None,
    }
    global_released = all(gates.values())
    version = _model_version(
        spec.name,
        feature_columns,
        split_seed=split_seed,
        random_seed=random_seed,
    )
    report: dict[str, object] = {
        "feature": spec.name,
        "model_version": version,
        "status": "released" if global_released else "experimental_unreleased",
        "reference_target": "echonest_computed",
        "split_unit": "artist",
        "split_rows": {name: len(part) for name, part in subsets.items()},
        "split_artists": {
            name: int(part[artist_column].nunique()) for name, part in subsets.items()
        },
        "observed_rows": int(observed.sum()),
        "missing_target_rows": int(missing_target.sum()),
        "test_metrics": {
            "mae": point_mae,
            "r2": r2,
            "dummy_mae": dummy_mae,
            "ridge_mae": ridge_mae,
            "improvement_over_dummy": dummy_improvement,
            "improvement_over_ridge": ridge_improvement,
            "interval_coverage": interval_coverage,
        },
        "calibration": {
            "retained_mae_limit": spec.retained_mae_limit,
            "maximum_interval_width": width_threshold,
            "ood_score_threshold": ood_threshold,
            "maximum_missing_feature_fraction": MAX_MISSING_FEATURE_FRACTION,
        },
        "retained_missing_coverage": retained_coverage,
        "release_gates": gates,
    }
    return _FittedTarget(
        spec=spec,
        feature_columns=feature_columns,
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
        location=location,
        scale=scale,
        ood_threshold=ood_threshold,
        width_threshold=width_threshold,
        model_version=version,
        global_released=global_released,
        report=report,
    )


def _predict_missing(
    fitted: _FittedTarget,
    frame: Any,
    *,
    track_id_column: str,
) -> tuple[FeaturePrediction, ...]:
    np, pd, *_ = _require_ml()
    target = pd.to_numeric(frame[fitted.spec.name], errors="coerce")
    missing = frame.loc[target.isna()]
    if missing.empty:
        return ()
    numeric = missing.loc[:, fitted.feature_columns].apply(pd.to_numeric, errors="coerce")
    point = _raw_predictions(np, fitted.point_model, numeric)
    lower_raw = _raw_predictions(np, fitted.lower_model, numeric)
    upper_raw = _raw_predictions(np, fitted.upper_model, numeric)
    lower = np.minimum(lower_raw, upper_raw)
    upper = np.maximum(lower_raw, upper_raw)
    widths = upper - lower
    ood, missing_fraction = _ood_scores(
        np, numeric, fitted.location, fitted.scale
    )
    individual_ok = (
        np.isfinite(point)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (ood <= fitted.ood_threshold)
        & (missing_fraction <= MAX_MISSING_FEATURE_FRACTION)
        & (point >= fitted.spec.low)
        & (point <= fitted.spec.high)
        & (lower >= fitted.spec.low)
        & (upper <= fitted.spec.high)
    )
    if fitted.width_threshold is None:
        individual_ok &= False
    else:
        individual_ok &= widths <= fitted.width_threshold

    track_ids = tuple(_positive_track_id(value) for value in missing[track_id_column])
    if len(track_ids) != len(set(track_ids)):
        raise ModelingDataError("track IDs must be unique")

    predictions: list[FeaturePrediction] = []
    for index, track_id in enumerate(track_ids):
        released = bool(fitted.global_released and individual_ok[index])
        if released:
            confidence = max(0.0, min(1.0, 1.0 - float(widths[index]) / fitted.spec.span))
            predictions.append(
                FeaturePrediction(
                    track_id=track_id,
                    feature=fitted.spec.name,
                    value=float(point[index]),
                    confidence=confidence,
                    interval_low=float(lower[index]),
                    interval_high=float(upper[index]),
                    model_version=fitted.model_version,
                    released=True,
                )
            )
        else:
            predictions.append(
                FeaturePrediction(
                    track_id=track_id,
                    feature=fitted.spec.name,
                    value=None,
                    confidence=None,
                    interval_low=None,
                    interval_high=None,
                    model_version=fitted.model_version,
                    released=False,
                )
            )
    return tuple(predictions)


def train_feature_models(
    frame: Any,
    *,
    feature_columns: Sequence[str],
    targets: Sequence[str] | None = None,
    track_id_column: str = "track_id",
    artist_column: str = "artist",
    split_seed: str = DEFAULT_SPLIT_SEED,
    random_seed: int = 20260802,
) -> ModelingRun:
    """Train/evaluate targets and return deterministic report + baked predictions.

    ``frame`` is a prepared pandas DataFrame. Feature columns should be the
    normalized FMA Librosa statistics; target columns use the names in
    ``FEATURE_SPECS`` and contain Echo Nest values where available and NA
    elsewhere.
    """
    _np, pd, *_ = _require_ml()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ModelingDataError("prepared table column names must be unique")
    selected_features = tuple(feature_columns)
    if not selected_features or len(selected_features) != len(set(selected_features)):
        raise ModelingDataError("feature_columns must be nonempty and unique")
    selected_targets = tuple(targets or FEATURE_SPECS)
    unknown_targets = set(selected_targets) - set(FEATURE_SPECS)
    if unknown_targets:
        raise ModelingDataError(f"unknown targets: {sorted(unknown_targets)}")
    leaked_targets = set(selected_features) & set(FEATURE_SPECS)
    if leaked_targets:
        raise ModelingDataError(
            f"Echo Nest target columns cannot be model inputs: {sorted(leaked_targets)}"
        )
    forbidden_identity_inputs = {track_id_column, artist_column} & set(selected_features)
    if forbidden_identity_inputs:
        raise ModelingDataError(
            f"identity columns cannot be model inputs: {sorted(forbidden_identity_inputs)}"
        )
    required_columns = {
        track_id_column,
        artist_column,
        *selected_features,
        *selected_targets,
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ModelingDataError(f"prepared table is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ModelingDataError("prepared table cannot be empty")

    work = frame.copy()
    track_ids = tuple(_positive_track_id(value) for value in work[track_id_column])
    if len(track_ids) != len(set(track_ids)):
        raise ModelingDataError("track IDs must be unique")
    assignments = artist_group_split_assignments(work[artist_column], seed=split_seed)
    split_column = "__cadence_artist_split"
    while split_column in work.columns:
        split_column += "_"
    work[split_column] = assignments

    reports: dict[str, object] = {}
    predictions: list[FeaturePrediction] = []
    for offset, target in enumerate(selected_targets):
        spec = FEATURE_SPECS[target]
        try:
            fitted = _fit_target(
                work,
                feature_columns=selected_features,
                spec=spec,
                artist_column=artist_column,
                split_column=split_column,
                split_seed=split_seed,
                random_seed=random_seed + offset * 10,
            )
        except ModelingDataError as exc:
            reports[target] = {
                "feature": target,
                "status": "unavailable",
                "released": False,
                "reason": str(exc),
            }
            continue
        reports[target] = fitted.report
        predictions.extend(
            _predict_missing(fitted, work, track_id_column=track_id_column)
        )

    split_counts = {
        name: int(sum(assignment == name for assignment in assignments))
        for name in ("train", "calibration", "test")
    }
    artist_assignments: dict[str, str] = {}
    for artist, assignment in zip(work[artist_column], assignments, strict=True):
        artist_assignments[_normal_artist(artist)] = assignment
    report: dict[str, object] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY_VERSION,
        "reference_target": "echonest_computed",
        "feature_count": len(selected_features),
        "row_count": len(work),
        "artist_count": len(artist_assignments),
        "split_policy": {
            "unit": "artist",
            "requested_ratio": {"train": 0.70, "calibration": 0.15, "test": 0.15},
            "seed": split_seed,
            "row_counts": split_counts,
            "artist_counts": {
                name: sum(value == name for value in artist_assignments.values())
                for name in ("train", "calibration", "test")
            },
        },
        "features": reports,
    }
    ordered = tuple(sorted(predictions, key=lambda row: (row.track_id, row.feature)))
    return ModelingRun(report=report, predictions=ordered)


def write_prediction_jsonl(path: str | Path, predictions: Iterable[FeaturePrediction]) -> None:
    """Write deterministic JSONL with the strict ETL prediction schema."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(predictions, key=lambda row: (row.track_id, row.feature))
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(destination)


def write_model_report(path: str | Path, report: Mapping[str, object]) -> None:
    """Write a stable, human-readable model release report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
