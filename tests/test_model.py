"""Tests for the cancellation model.

The leakage tests are the point of this file. A model trained on a leaked
column scores beautifully and predicts nothing, and that failure is invisible
in every metric - which is exactly why it needs a test rather than a comment.

These run offline against a small synthetic frame and never load the real
trained model, so they stay fast and do not require the model stage to have run.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_pipeline import features as F  # noqa: E402
from hotel_pipeline import model as M  # noqa: E402
from hotel_pipeline.config import Config, Paths  # noqa: E402


def synthetic(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A frame with a real signal: long lead times cancel more."""
    rng = np.random.default_rng(seed)
    lead = rng.integers(0, 400, n)
    # Cancellation probability rises with lead time, so the model has
    # something genuine to find rather than noise to memorise.
    p = 0.05 + 0.85 * (lead / 400)
    canceled = rng.random(n) < p
    numeric, categorical = F.feature_columns()
    frame = pd.DataFrame({"lead_time": lead, "is_canceled": canceled})

    for col in numeric:
        if col == "lead_time":
            continue
        frame[col] = rng.integers(0, 4, n)
    frame["adr"] = rng.uniform(40, 200, n).round(2)
    frame["arrival_month_num"] = rng.integers(1, 13, n)
    for col in categorical:
        frame[col] = rng.choice(["x", "y", "z"], n)
    frame["hotel"] = rng.choice(["City Hotel", "Resort Hotel"], n)
    # These two are genuinely boolean in the staged data. The fixture has to
    # match, or the encoder is fitted on strings and scored on bools.
    for col in ("has_agent", "has_company"):
        frame[col] = rng.random(n) < 0.7

    # Half the rows fall either side of the train/test cutoff.
    start = pd.Timestamp("2016-01-01")
    frame["arrival_date"] = start + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )
    return frame


class TestFeatureTiers(unittest.TestCase):
    def test_no_leaked_column_is_ever_a_feature(self):
        numeric, categorical = F.feature_columns(include_pre_arrival=True)
        for column in numeric + categorical:
            self.assertNotIn(column, F.LEAKED,
                             f"{column} is both a feature and leakage")

    def test_assert_no_leakage_raises_on_the_classic_trap(self):
        # room_changed is the one that looks like the best feature in the data
        # and is worthless: it is only known once the guest has arrived.
        with self.assertRaises(ValueError) as ctx:
            F.assert_no_leakage(["lead_time", "room_changed"])
        self.assertIn("room_changed", str(ctx.exception))

    def test_assert_no_leakage_names_every_offender(self):
        with self.assertRaises(ValueError) as ctx:
            F.assert_no_leakage(["reservation_status", "is_no_show"])
        message = str(ctx.exception)
        self.assertIn("reservation_status", message)
        self.assertIn("is_no_show", message)

    def test_clean_feature_set_passes(self):
        numeric, categorical = F.feature_columns()
        F.assert_no_leakage(numeric + categorical)  # must not raise

    def test_target_is_not_a_feature(self):
        numeric, categorical = F.feature_columns(include_pre_arrival=True)
        self.assertNotIn(F.TARGET, numeric + categorical)

    def test_pre_arrival_tier_is_opt_in(self):
        safe, _ = F.feature_columns(include_pre_arrival=False)
        both, _ = F.feature_columns(include_pre_arrival=True)
        self.assertNotIn("booking_changes", safe)
        self.assertIn("booking_changes", both)

    def test_parking_is_blocked_as_leakage(self):
        # Every one of the 7,409 bookings requesting a parking space completed
        # its stay. Perfect separation means the field is recorded on arrival,
        # not at booking - the same trap as room_changed.
        self.assertIn("required_car_parking_spaces", F.LEAKED)
        numeric, categorical = F.feature_columns(include_pre_arrival=True)
        self.assertNotIn("required_car_parking_spaces", numeric + categorical)
        with self.assertRaises(ValueError):
            F.assert_no_leakage(["required_car_parking_spaces"])

    def test_no_feature_perfectly_separates_the_target(self):
        """A feature with zero positives in a large stratum is leakage.

        This is the check that would have caught the parking column before it
        reached a model, rather than after someone noticed a 0.0% prediction.
        """
        df = synthetic(2000)
        numeric, categorical = F.feature_columns()
        for col in numeric:
            for value, group in df.groupby(col):
                if len(group) >= 50:
                    rate = group["is_canceled"].mean()
                    self.assertTrue(
                        0.0 < rate < 1.0,
                        f"{col}={value} perfectly separates the target "
                        f"({len(group)} rows, rate {rate})")

    def test_audit_classifies_every_staged_column(self):
        columns = (F.SAFE_NUMERIC + F.SAFE_CATEGORICAL + F.PRE_ARRIVAL_NUMERIC
                   + list(F.LEAKED) + list(F.EXCLUDED) + [F.TARGET])
        self.assertEqual(F.audit(columns)["unclassified"], [])

    def test_audit_flags_an_unknown_column(self):
        report = F.audit(["lead_time", "some_new_column"])
        self.assertEqual(report["unclassified"], ["some_new_column"])


class TestCategoryCaster(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 30 + ["rare"] * 2})
        self.caster = M.CategoryCaster(["c"], max_categories=2).fit(self.df)

    def test_rare_levels_fold_into_other(self):
        out = self.caster.transform(self.df)
        self.assertEqual(set(out["c"].dropna()), {"a", "b", "__other__"})

    def test_unseen_level_becomes_other_not_nan(self):
        # A country that first appears at scoring time must not silently
        # become a missing value.
        out = self.caster.transform(pd.DataFrame({"c": ["never_seen"]}))
        self.assertEqual(out["c"].iloc[0], "__other__")
        self.assertFalse(out["c"].isna().any())

    def test_output_is_categorical_dtype(self):
        out = self.caster.transform(self.df)
        self.assertEqual(out["c"].dtype.name, "category")


class TestPrepareBooking(unittest.TestCase):
    def test_defaults_produce_a_complete_feature_row(self):
        numeric, categorical = F.feature_columns(include_pre_arrival=True)
        row = M.prepare_booking()
        for col in numeric + categorical:
            self.assertIn(col, row.columns, f"{col} missing from booking row")
        self.assertEqual(len(row), 1)

    def test_derived_fields_match_the_training_definitions(self):
        row = M.prepare_booking(weekend_nights=2, week_nights=3, adults=2,
                                children=1, adr=100.0).iloc[0]
        self.assertEqual(row["total_nights"], 5)
        self.assertEqual(row["total_guests"], 3)
        self.assertEqual(row["booking_value"], 500.0)
        self.assertEqual(row["party_type"], "Family")   # children present
        self.assertEqual(row["stay_type"], "Mixed")

    def test_party_type_rules(self):
        self.assertEqual(M.prepare_booking(adults=1).iloc[0]["party_type"],
                         "Solo")
        self.assertEqual(M.prepare_booking(adults=2).iloc[0]["party_type"],
                         "Couple")
        self.assertEqual(M.prepare_booking(adults=4).iloc[0]["party_type"],
                         "Group")
        self.assertEqual(
            M.prepare_booking(adults=2, babies=1).iloc[0]["party_type"],
            "Family")

    def test_season_follows_the_arrival_month(self):
        self.assertEqual(M.prepare_booking(arrival_month=1).iloc[0]["season"],
                         "Winter")
        self.assertEqual(M.prepare_booking(arrival_month=8).iloc[0]["season"],
                         "Summer")

    def test_no_guests_is_rejected(self):
        with self.assertRaises(ValueError):
            M.prepare_booking(adults=0, children=0, babies=0)

    def test_unknown_field_is_rejected_not_ignored(self):
        # Silently dropping a typo'd field would score a different booking
        # than the caller described.
        with self.assertRaises(ValueError) as ctx:
            M.prepare_booking(leadtime=30)
        self.assertIn("leadtime", str(ctx.exception))

    def test_booking_row_carries_no_leaked_column(self):
        F.assert_no_leakage(M.prepare_booking().columns)


class TestTrainedModel(unittest.TestCase):
    """End-to-end on synthetic data, with calibration skipped for speed."""

    @classmethod
    def setUpClass(cls):
        df = synthetic()
        numeric, categorical = F.feature_columns()
        cls.numeric, cls.categorical = numeric, categorical
        X, y = df[numeric + categorical], df["is_canceled"].astype(int)
        pipeline = M.build_logistic(numeric, categorical).fit(X, y)
        cls.model = M.CancellationModel(
            pipeline, numeric, categorical,
            {"test": {"base_rate": float(y.mean())}},
        )
        cls.X, cls.y = X, y

    def test_probabilities_are_in_range(self):
        p = self.model.predict_proba(self.X)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertEqual(len(p), len(self.X))

    def test_model_learns_the_planted_signal(self):
        # Long lead times were generated to cancel more; the model must find it.
        short = self.model.predict_one(lead_time=5)["probability"]
        long = self.model.predict_one(lead_time=380)["probability"]
        self.assertGreater(long, short)

    def test_predict_one_reports_what_it_defaulted(self):
        result = self.model.predict_one(lead_time=100, country="GBR")
        self.assertEqual(result["fields_supplied"], ["country", "lead_time"])
        self.assertIn("adr", result["fields_defaulted"])
        self.assertNotIn("lead_time", result["fields_defaulted"])

    def test_predict_one_output_shape(self):
        result = self.model.predict_one(lead_time=50)
        self.assertIn(result["risk_band"],
                      {"Low", "Moderate", "Elevated", "High"})
        self.assertAlmostEqual(result["percent"],
                               result["probability"] * 100, places=1)

    def test_missing_feature_column_is_an_error(self):
        with self.assertRaises(ValueError):
            self.model.predict_proba(self.X.drop(columns=["lead_time"]))

    def test_round_trip_through_disk_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            self.model.save(cfg.paths.model)
            reloaded = M.CancellationModel.load(cfg.paths.model)
            np.testing.assert_allclose(
                self.model.predict_proba(self.X.head(50)),
                reloaded.predict_proba(self.X.head(50)),
            )

    def test_load_without_a_trained_model_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            with self.assertRaises(FileNotFoundError) as ctx:
                M.CancellationModel.load(cfg=cfg)
            self.assertIn("--stage model", str(ctx.exception))


class TestEvaluate(unittest.TestCase):
    def test_perfect_predictions_score_perfectly(self):
        y = np.array([0, 0, 1, 1] * 25)
        metrics = M.evaluate(y, y.astype(float), "perfect")
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.0)

    def test_constant_prediction_has_no_ranking_power(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        metrics = M.evaluate(y, np.full(400, y.mean()), "constant")
        self.assertAlmostEqual(metrics["roc_auc"], 0.5, places=2)

    def test_deciles_cover_every_row(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        metrics = M.evaluate(y, rng.random(500), "random")
        self.assertEqual(sum(d["bookings"] for d in metrics["deciles"]), 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
