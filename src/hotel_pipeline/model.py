"""Stage 6 - Predict.

Trains a calibrated cancellation-risk model and exposes it for scoring single
bookings.

Three decisions matter more here than the choice of algorithm:

**Split on time, never at random.** The model is trained on arrivals before
2017 and tested on 2017 onward. A random split would let the model learn from
bookings that, at the moment of prediction, had not happened yet - and booking
behaviour genuinely drifts across this series.

**The feature tiers are enforced, not remembered.** Everything the model may see
is declared in ``features.py`` and checked before training. See that module for
why `room_changed` would otherwise wreck this.

**Calibration, not just ranking.** The point of a cancellation model is to sum
probabilities into expected arrivals for overbooking. That needs P(cancel) to be
a real probability, so the model is isotonically calibrated and scored on Brier
and log loss, not only on AUC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features as F
from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger
from .transform import SEASONS

log = get_logger(__name__)

TEST_FROM = "2017-01-01"   # arrivals from here are held out entirely
RANDOM_SEED = 42

RISK_BANDS = [
    (0.20, "Low"),
    (0.40, "Moderate"),
    (0.60, "Elevated"),
    (1.01, "High"),
]


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Cast categoricals to a fixed vocabulary learned at fit time.

    High-cardinality columns (country has 178 values) are capped at the most
    frequent `max_categories`; everything else, including values never seen in
    training, folds into a single "other" level. Without this, a country that
    first appears at scoring time would silently become a missing value.
    """

    def __init__(self, columns: list[str], max_categories: int = 30,
                 other: str = "__other__"):
        self.columns = columns
        self.max_categories = max_categories
        self.other = other

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {}
        for col in self.columns:
            counts = X[col].astype("object").value_counts()
            keep = list(counts.head(self.max_categories).index)
            self.categories_[col] = keep + [self.other]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, levels in self.categories_.items():
            values = X[col].astype("object")
            values = values.where(values.isin(levels[:-1]), self.other)
            X[col] = pd.Categorical(values, categories=levels)
        return X


def build_gboost(numeric: list[str], categorical: list[str]) -> Pipeline:
    """Gradient boosting with native categorical splits."""
    return Pipeline([
        ("cast", CategoryCaster(categorical)),
        ("clf", HistGradientBoostingClassifier(
            categorical_features="from_dtype",
            learning_rate=0.08,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=RANDOM_SEED,
        )),
    ])


def build_logistic(numeric: list[str], categorical: list[str]) -> Pipeline:
    """A linear baseline that a revenue manager can be walked through."""
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25),
             categorical),
        ])),
        ("clf", LogisticRegression(max_iter=3000, C=0.5,
                                   random_state=RANDOM_SEED)),
    ])


# --- Metrics --------------------------------------------------------------

def evaluate(y_true, y_prob, label: str) -> dict:
    """Ranking and calibration quality, plus the decile lift table."""
    metrics = {
        "label": label,
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "mean_predicted": float(np.mean(y_prob)),
    }
    # Calibration error: how far the predicted probability sits from the
    # observed frequency, averaged over ten equal-width bins.
    true_p, pred_p = calibration_curve(y_true, y_prob, n_bins=10,
                                       strategy="quantile")
    metrics["calibration_mae"] = float(np.mean(np.abs(true_p - pred_p)))
    metrics["calibration_curve"] = {
        "predicted": [float(v) for v in pred_p],
        "observed": [float(v) for v in true_p],
    }

    frame = pd.DataFrame({"y": y_true, "p": y_prob})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), 10,
                              labels=range(1, 11))
    lift = (frame.groupby("decile", observed=True)
            .agg(bookings=("y", "size"), cancelled=("y", "sum"),
                 mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
            .reset_index())
    lift["lift"] = lift["actual_rate"] / max(frame["y"].mean(), 1e-9)
    metrics["deciles"] = lift.to_dict(orient="records")
    return metrics


def _log_metrics(m: dict) -> None:
    log.info("  %-28s ROC-AUC %.4f  PR-AUC %.4f  Brier %.4f  "
             "calib MAE %.4f", m["label"], m["roc_auc"], m["pr_auc"],
             m["brier"], m["calibration_mae"])


# --- The model object -----------------------------------------------------

@dataclass
class CancellationModel:
    """A fitted, calibrated model plus everything needed to score with it."""

    pipeline: object
    numeric: list[str]
    categorical: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + self.categorical

    # --- scoring ----------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(cancel) for a frame that already has the feature columns."""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature column(s): {missing}")
        return self.pipeline.predict_proba(df[self.feature_names])[:, 1]

    def predict_one(self, **booking) -> dict:
        """Score a single booking described in plain terms.

        Only the fields you know are needed; the rest fall back to the
        defaults in `BOOKING_DEFAULTS`, and derived features (total nights,
        party type, season) are computed the same way the pipeline computes
        them for training data.
        """
        row = prepare_booking(**booking)
        probability = float(self.predict_proba(row)[0])
        band = next(name for ceiling, name in RISK_BANDS
                    if probability < ceiling)
        base = self.metadata.get("test", {}).get("base_rate", 0.37)
        supplied = {k for k, v in booking.items() if v is not None}
        return {
            "probability": round(probability, 4),
            "percent": round(probability * 100, 1),
            "risk_band": band,
            "base_rate": round(base, 4),
            "lift_vs_base": round(probability / base, 2) if base else None,
            # Defaults are the dataset's *modal* booking, which is itself
            # high-risk (Portuguese, Online TA, 90-day lead). Naming the
            # defaulted fields keeps a scary number from looking like it came
            # from the caller's own inputs.
            "fields_supplied": sorted(supplied),
            "fields_defaulted": sorted(set(BOOKING_DEFAULTS) - supplied),
            "inputs_used": {k: _plain(v) for k, v in
                            row.iloc[0].to_dict().items()},
        }

    # --- persistence ------------------------------------------------------
    def save(self, path: Path) -> Path:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "metadata": self.metadata,
        }, path, compress=3)
        log.info("Model saved -> %s (%.1f MB)", path.name,
                 path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path | None = None,
             cfg: Config = DEFAULT_CONFIG) -> "CancellationModel":
        import joblib
        path = path or cfg.paths.model
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Run `python run_pipeline.py "
                f"--stage model` first."
            )
        blob = joblib.load(path)
        return cls(blob["pipeline"], blob["numeric"], blob["categorical"],
                   blob["metadata"])


def _plain(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# --- Building a single booking -------------------------------------------

# A single coherent booking, not a bag of independent column modes. The modal
# values disagree with each other - a Friday three-night stay cannot have one
# weekend night - so these are pinned to one real date, Friday 21 July 2017
# (ISO week 29), and the weekday/weekend split follows from that calendar. The
# browser app derives the same split from the same date, so both interfaces
# score the untouched default identically.
BOOKING_DEFAULTS = {
    "hotel": "City Hotel",
    "lead_time": 90,
    "arrival_month": 7,
    "arrival_dow": "Friday",
    "arrival_date_week_number": 29,
    "weekend_nights": 2,
    "week_nights": 1,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "adr": 100.0,
    "meal": "Bed & Breakfast",
    "country": "PRT",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "special_requests": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "has_agent": True,
    "has_company": False,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
}


def prepare_booking(**booking) -> pd.DataFrame:
    """Turn plain booking facts into the model's feature row.

    Derived features are computed exactly as `transform.py` computes them for
    training data - if the two ever drifted apart, the model would be scored on
    a different definition than it was trained on.
    """
    unknown = set(booking) - set(BOOKING_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown booking field(s): {sorted(unknown)}. "
            f"Valid fields: {sorted(BOOKING_DEFAULTS)}"
        )
    b = {**BOOKING_DEFAULTS, **{k: v for k, v in booking.items()
                                if v is not None}}

    weekend, week = int(b["weekend_nights"]), int(b["week_nights"])
    total_nights = weekend + week
    adults, children, babies = int(b["adults"]), int(b["children"]), int(b["babies"])
    total_guests = adults + children + babies
    if total_guests <= 0:
        raise ValueError("A booking needs at least one guest.")

    if children + babies > 0:
        party = "Family"
    elif total_guests == 1:
        party = "Solo"
    elif total_guests == 2:
        party = "Couple"
    else:
        party = "Group"

    if week > 0 and weekend > 0:
        stay = "Mixed"
    elif week > 0:
        stay = "Weekday only"
    elif weekend > 0:
        stay = "Weekend only"
    else:
        stay = "No overnight stay"

    month = int(b["arrival_month"])
    adr = float(b["adr"])

    row = {
        "lead_time": int(b["lead_time"]),
        "arrival_month_num": month,
        "arrival_date_week_number": int(b["arrival_date_week_number"]),
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,
        "total_nights": total_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "total_guests": total_guests,
        "adr": adr,
        "booking_value": round(adr * total_nights, 2),
        "previous_cancellations": int(b["previous_cancellations"]),
        "previous_bookings_not_canceled": int(b["previous_bookings_not_canceled"]),
        "total_of_special_requests": int(b["special_requests"]),
        "is_repeated_guest": int(b["is_repeated_guest"]),
        "booking_changes": int(b["booking_changes"]),
        "days_in_waiting_list": int(b["days_in_waiting_list"]),
        "hotel": b["hotel"],
        "meal": b["meal"],
        "country": b["country"],
        "market_segment": b["market_segment"],
        "distribution_channel": b["distribution_channel"],
        "reserved_room_type": b["reserved_room_type"],
        "deposit_type": b["deposit_type"],
        "customer_type": b["customer_type"],
        "season": SEASONS[month],
        "arrival_dow": b["arrival_dow"],
        "party_type": party,
        "stay_type": stay,
        "has_agent": bool(b["has_agent"]),
        "has_company": bool(b["has_company"]),
    }
    return pd.DataFrame([row])


# --- Training -------------------------------------------------------------

def train(cfg: Config = DEFAULT_CONFIG, include_pre_arrival: bool = False
          ) -> tuple[CancellationModel, dict]:
    """Train, calibrate and evaluate. Returns the model and its metrics."""
    staged = pd.read_parquet(cfg.paths.staged_parquet)
    log.info("Training on %s staged bookings", f"{len(staged):,}")

    numeric, categorical = F.feature_columns(include_pre_arrival)
    F.assert_no_leakage(numeric + categorical)
    audit = F.audit(staged.columns)
    log.info("  %d feature(s); %d blocked as leakage; %d excluded by design",
             len(numeric) + len(categorical), len(audit["blocked_as_leakage"]),
             len(audit["excluded_by_design"]))
    if audit["unclassified"]:
        log.warning("  %d unclassified column(s): %s",
                    len(audit["unclassified"]), audit["unclassified"])

    # Split on arrival date: a model deployed on the cutoff only knows the
    # outcome of stays that have already happened.
    is_test = staged["arrival_date"] >= TEST_FROM
    train_df, test_df = staged[~is_test], staged[is_test]
    X_train = train_df[numeric + categorical]
    X_test = test_df[numeric + categorical]
    y_train = train_df[F.TARGET].astype(int).to_numpy()
    y_test = test_df[F.TARGET].astype(int).to_numpy()
    log.info("  train %s rows (arrivals < %s, %.1f%% cancelled)",
             f"{len(X_train):,}", TEST_FROM, 100 * y_train.mean())
    log.info("  test  %s rows (arrivals >= %s, %.1f%% cancelled)",
             f"{len(X_test):,}", TEST_FROM, 100 * y_test.mean())

    results: dict[str, dict] = {}

    # Baseline: everyone gets the training base rate. Any model that cannot
    # beat this on Brier is not earning its keep.
    baseline = np.full(len(y_test), y_train.mean())
    results["baseline"] = {
        "label": "baseline (train base rate)",
        "brier": float(brier_score_loss(y_test, baseline)),
        "log_loss": float(log_loss(y_test, baseline)),
        "roc_auc": 0.5, "pr_auc": float(y_test.mean()),
    }
    log.info("  %-28s Brier %.4f (no skill)", "baseline",
             results["baseline"]["brier"])

    log.info("Fitting logistic regression")
    logistic = build_logistic(numeric, categorical).fit(X_train, y_train)
    results["logistic"] = evaluate(
        y_test, logistic.predict_proba(X_test)[:, 1], "logistic regression"
    )
    _log_metrics(results["logistic"])

    log.info("Fitting gradient boosting")
    gboost = build_gboost(numeric, categorical).fit(X_train, y_train)
    results["gboost"] = evaluate(
        y_test, gboost.predict_proba(X_test)[:, 1], "gradient boosting"
    )
    _log_metrics(results["gboost"])

    # Calibrate with internal cross-validation on the training data only, so
    # the test set stays untouched by any fitting decision.
    log.info("Calibrating (isotonic, 5-fold on train)")
    calibrated = CalibratedClassifierCV(
        build_gboost(numeric, categorical), method="isotonic", cv=5
    ).fit(X_train, y_train)
    results["calibrated"] = evaluate(
        y_test, calibrated.predict_proba(X_test)[:, 1],
        "gradient boosting (calibrated)"
    )
    _log_metrics(results["calibrated"])

    # Pick on Brier across every candidate: it is the metric that matters for
    # summing probabilities into expected arrivals. Ranking quality (AUC) is
    # reported too, because a model used to target interventions rather than
    # to forecast volume would be chosen on that instead.
    candidates = {"logistic": logistic, "gboost": gboost,
                  "calibrated": calibrated}
    best_name = min(candidates, key=lambda k: results[k]["brier"])
    best = candidates[best_name]
    log.info("Selected: %s (best Brier of %d candidates)",
             results[best_name]["label"], len(candidates))
    ranker = max(candidates, key=lambda k: results[k]["roc_auc"])
    if ranker != best_name:
        log.info("  note: %s ranks better (ROC-AUC %.4f vs %.4f) but is less "
                 "well calibrated", results[ranker]["label"],
                 results[ranker]["roc_auc"], results[best_name]["roc_auc"])

    # Permutation importance on held-out data: shuffle one column, see how
    # much Brier degrades. Model-agnostic, and unlike a coefficient it answers
    # "how much does the model actually lean on this" rather than "what is the
    # weight". Subsampled because it refits nothing but predicts n_features x
    # n_repeats times.
    log.info("Computing permutation importance")
    from sklearn.inspection import permutation_importance
    sample = min(12_000, len(X_test))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_test), size=sample, replace=False)
    importance = permutation_importance(
        best, X_test.iloc[idx], y_test[idx], n_repeats=3,
        scoring="neg_brier_score", random_state=RANDOM_SEED, n_jobs=1,
    )
    ranked = sorted(
        ({"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(X_test.columns, importance.importances_mean,
                            importance.importances_std)),
        key=lambda r: r["importance"], reverse=True,
    )
    results["importance"] = ranked
    log.info("  top 5: %s", ", ".join(r["feature"] for r in ranked[:5]))

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "selected": best_name,
        "feature_tier": "safe + pre-arrival" if include_pre_arrival else "safe",
        "n_features": len(numeric) + len(categorical),
        "numeric": numeric,
        "categorical": categorical,
        "test_from": TEST_FROM,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train": {"base_rate": float(y_train.mean())},
        "test": results[best_name],
        "all_models": {k: {m: v for m, v in r.items()
                           if m not in ("deciles", "calibration_curve")}
                       for k, r in results.items() if isinstance(r, dict)},
        "top_features": [r["feature"] for r in ranked[:10]],
        "leakage_blocked": audit["blocked_as_leakage"],
    }
    model = CancellationModel(best, numeric, categorical, metadata)
    model.save(cfg.paths.model)

    # The browser app runs the logistic form, so export it whether or not it
    # was the selected model - the app says which it is.
    log.info("Exporting logistic model for the web app")
    export_for_web(logistic, numeric, categorical, X_train.assign(
        **{F.TARGET: y_train}), metadata, cfg)
    from .app import build_app
    build_app(cfg)

    cfg.paths.model_metrics.write_text(
        json.dumps({"selected": best_name, "models": results,
                    "metadata": {k: v for k, v in metadata.items()
                                 if k not in ("test", "all_models")}},
                   indent=2, default=_plain),
        encoding="utf-8",
    )
    log.info("Model metrics -> %s", cfg.paths.model_metrics.name)
    return model, results


def export_for_web(pipeline: Pipeline, numeric: list[str],
                   categorical: list[str], train_df: pd.DataFrame,
                   metadata: dict, cfg: Config = DEFAULT_CONFIG) -> Path:
    """Export the logistic model as JSON so a browser can run it exactly.

    A logistic regression is a scaler, a one-hot encoder and a dot product, so
    the whole thing travels as a few hundred numbers and the scoring page needs
    no server. The parity check at the end is the point: if the exported form
    ever stops matching scikit-learn's own output, this raises rather than
    shipping a page that quietly disagrees with the pipeline.
    """
    ct = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    scaler = ct.named_transformers_["num"]
    ohe = ct.named_transformers_["cat"]
    names = list(ct.get_feature_names_out())
    coefs = clf.coef_[0]
    index = {name: i for i, name in enumerate(names)}

    numeric_spec = [
        {"name": col, "mean": float(scaler.mean_[i]),
         "scale": float(scaler.scale_[i]),
         "coef": float(coefs[index[f"num__{col}"]])}
        for i, col in enumerate(numeric)
    ]

    categorical_spec: dict[str, dict] = {}
    for position, col in enumerate(categorical):
        levels: dict[str, float] = {}
        for value in ohe.categories_[position]:
            key = f"cat__{col}_{value}"
            if key in index:
                levels[str(value)] = float(coefs[index[key]])
        infrequent_key = f"cat__{col}_infrequent_sklearn"
        infrequent = ohe.infrequent_categories_[position]
        categorical_spec[col] = {
            # Coefficient per level. A level that was folded into the
            # infrequent bucket is not listed and falls back to `infrequent`.
            "levels": levels,
            "infrequent": (float(coefs[index[infrequent_key]])
                           if infrequent_key in index else None),
            # handle_unknown="ignore": an unseen value contributes nothing.
            "folded": sorted(str(v) for v in (infrequent
                                              if infrequent is not None else [])),
            # Ordered by training frequency so the UI lists useful values first.
            "order": [str(v) for v in
                      train_df[col].astype(str).value_counts().index
                      if str(v) in levels],
        }

    payload = {
        "kind": "logistic",
        "intercept": float(clf.intercept_[0]),
        "numeric": numeric_spec,
        "categorical": categorical_spec,
        "defaults": {k: _plain(v) for k, v in BOOKING_DEFAULTS.items()},
        "risk_bands": [[c, n] for c, n in RISK_BANDS],
        "meta": {
            "trained_at": metadata.get("trained_at"),
            "is_selected_model": metadata.get("selected") == "logistic",
            "test_from": metadata.get("test_from"),
            "n_train": metadata.get("n_train"),
            "n_test": metadata.get("n_test"),
            "train_base_rate": metadata.get("train", {}).get("base_rate"),
            "test_base_rate": metadata.get("test", {}).get("base_rate"),
            "roc_auc": metadata.get("all_models", {})
                               .get("logistic", {}).get("roc_auc"),
            "brier": metadata.get("all_models", {})
                             .get("logistic", {}).get("brier"),
            "mean_predicted": metadata.get("all_models", {})
                                      .get("logistic", {}).get("mean_predicted"),
        },
    }

    # --- Parity check: re-score real rows using only the exported numbers ---
    sample = train_df.head(400)
    expected = pipeline.predict_proba(sample[numeric + categorical])[:, 1]
    got = np.array([_score_from_payload(payload, row)
                    for row in sample.to_dict(orient="records")])
    drift = float(np.max(np.abs(expected - got)))
    if drift > 1e-9:
        raise ValueError(
            f"Exported model disagrees with scikit-learn by up to {drift:.2e}. "
            f"Refusing to ship a web model that does not match the pipeline."
        )
    log.info("  parity with scikit-learn verified (max diff %.2e over %d rows)",
             drift, len(sample))

    path = cfg.paths.web_model
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"),
                               default=_plain), encoding="utf-8")
    log.info("Web model -> %s (%.0f KB)", path.name, path.stat().st_size / 1024)
    return path


def _score_from_payload(payload: dict, row: dict) -> float:
    """Reference implementation of the exported model, used to verify parity.

    Deliberately mirrors what the JavaScript does, arithmetic for arithmetic,
    so the check actually proves the browser maths is right.
    """
    z = payload["intercept"]
    for spec in payload["numeric"]:
        value = float(row[spec["name"]])
        z += spec["coef"] * (value - spec["mean"]) / spec["scale"]
    for col, spec in payload["categorical"].items():
        value = str(row[col])
        if value in spec["levels"]:
            z += spec["levels"][value]
        elif value in spec["folded"] and spec["infrequent"] is not None:
            z += spec["infrequent"]
        # else: unseen value contributes nothing (handle_unknown="ignore")
    return 1.0 / (1.0 + np.exp(-z))


def score_warehouse(model: CancellationModel,
                    cfg: Config = DEFAULT_CONFIG) -> int:
    """Write a risk score for every booking back into the warehouse."""
    import sqlite3
    from contextlib import closing

    staged = pd.read_parquet(cfg.paths.staged_parquet)
    probability = model.predict_proba(staged)
    scored = pd.DataFrame({
        "booking_id": staged["booking_id"],
        "cancel_probability": probability.round(4),
        "risk_band": pd.cut(
            probability, bins=[-0.01, 0.20, 0.40, 0.60, 1.01],
            labels=["Low", "Moderate", "Elevated", "High"],
        ).astype(str),
        "in_training_period": (staged["arrival_date"] < TEST_FROM).astype(int),
    })
    with closing(sqlite3.connect(cfg.paths.database)) as conn:
        scored.to_sql("fact_booking_scored", conn, index=False,
                      if_exists="replace")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_scored_booking "
                     "ON fact_booking_scored(booking_id)")
        conn.execute("DROP VIEW IF EXISTS v_booking_risk")
        conn.execute("""
            CREATE VIEW v_booking_risk AS
            SELECT b.*, s.cancel_probability, s.risk_band
            FROM v_bookings b
            JOIN fact_booking_scored s ON b.booking_id = s.booking_id
        """)
        conn.commit()
    log.info("Scored %s bookings -> fact_booking_scored (+ v_booking_risk)",
             f"{len(scored):,}")
    return len(scored)
