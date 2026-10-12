"""Tests for the hotel booking pipeline.

Written as ``unittest`` cases so they run with either runner:

    python -m unittest discover -s tests -v      # no install needed
    pytest tests -v                              # if pytest is available

Every test builds its own tiny frame with the quirks of the real source baked
in, so the suite runs offline and in well under a second. The point is to pin
the *cleaning decisions* - those are the part of a pipeline that silently drifts.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hotel_pipeline import transform as T  # noqa: E402
from hotel_pipeline import validate as V  # noqa: E402
from hotel_pipeline import warehouse as W  # noqa: E402
from hotel_pipeline.analyze import derive_insights, load_queries  # noqa: E402
from hotel_pipeline.config import Config, Paths  # noqa: E402
from hotel_pipeline.extract import EXPECTED_COLUMNS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_raw(n_extra: int = 0) -> pd.DataFrame:
    """A miniature raw frame carrying every quirk the transform must handle."""
    rows = [
        # 0: ordinary city booking, not cancelled
        dict(hotel="City Hotel", is_canceled=0, lead_time=5,
             arrival_date_month="January", arrival_date_day_of_month=15,
             adults=2, children=0.0, babies=0, meal="BB", country="PRT",
             stays_in_weekend_nights=0, stays_in_week_nights=3, adr=100.0,
             reserved_room_type="A", assigned_room_type="A",
             reservation_status="Check-Out", is_repeated_guest=0),
        # 1: cancelled, long lead time, null children, null country
        dict(hotel="Resort Hotel", is_canceled=1, lead_time=400,
             arrival_date_month="August", arrival_date_day_of_month=2,
             adults=2, children=np.nan, babies=0, meal="Undefined",
             country=None, stays_in_weekend_nights=2, stays_in_week_nights=5,
             adr=250.0, reserved_room_type="A", assigned_room_type="C",
             reservation_status="Canceled", is_repeated_guest=0),
        # 2: zero occupants - must be dropped
        dict(hotel="City Hotel", is_canceled=0, lead_time=10,
             arrival_date_month="March", arrival_date_day_of_month=1,
             adults=0, children=0.0, babies=0, meal="BB", country="GBR",
             stays_in_weekend_nights=0, stays_in_week_nights=1, adr=80.0,
             reserved_room_type="A", assigned_room_type="A",
             reservation_status="Check-Out", is_repeated_guest=0),
        # 3: absurd ADR - must be winsorised, original preserved
        dict(hotel="City Hotel", is_canceled=0, lead_time=60,
             arrival_date_month="July", arrival_date_day_of_month=10,
             adults=1, children=2.0, babies=1, meal="HB", country="FRA",
             stays_in_weekend_nights=1, stays_in_week_nights=1, adr=5400.0,
             reserved_room_type="B", assigned_room_type="B",
             reservation_status="Check-Out", is_repeated_guest=1),
        # 4: negative ADR, zero nights, repeat guest
        dict(hotel="Resort Hotel", is_canceled=0, lead_time=0,
             arrival_date_month="December", arrival_date_day_of_month=31,
             adults=1, children=0.0, babies=0, meal="SC", country="ESP",
             stays_in_weekend_nights=0, stays_in_week_nights=0, adr=-6.38,
             reserved_room_type="A", assigned_room_type="A",
             reservation_status="No-Show", is_repeated_guest=1),
    ]
    # 5 & 6: an exact duplicate pair
    rows.append(dict(rows[0]))
    rows.extend(dict(rows[0]) for _ in range(n_extra))

    defaults = dict(
        arrival_date_year=2016, arrival_date_week_number=3,
        market_segment="Online TA", distribution_channel="TA/TO",
        previous_cancellations=0, previous_bookings_not_canceled=0,
        booking_changes=0, deposit_type="No Deposit", agent=240.0,
        company=np.nan, days_in_waiting_list=0, customer_type="Transient",
        required_car_parking_spaces=0, total_of_special_requests=1,
        reservation_status_date="2016-01-20",
    )
    df = pd.DataFrame([{**defaults, **row} for row in rows])
    return df[EXPECTED_COLUMNS]


class TestClean(unittest.TestCase):
    def setUp(self):
        self.raw = make_raw()
        self.clean = T.clean(self.raw, Config())

    def test_zero_guest_rows_are_dropped(self):
        self.assertEqual(len(self.clean), len(self.raw) - 1)
        self.assertTrue((self.clean["total_guests"] > 0).all())

    def test_input_frame_is_not_mutated(self):
        # A transform that edits its argument makes stage re-runs unreliable.
        self.assertTrue(self.raw["children"].isna().any())
        self.assertEqual(len(self.raw), 6)

    def test_null_children_become_zero(self):
        self.assertFalse(self.clean["children"].isna().any())
        self.assertEqual(self.clean["children"].dtype, np.dtype("int64"))

    def test_null_country_is_labelled_not_dropped(self):
        self.assertIn(T.UNKNOWN, set(self.clean["country"]))
        self.assertFalse(self.clean["country"].isna().any())

    def test_undefined_meal_collapses_into_self_catering(self):
        self.assertNotIn("Undefined", set(self.clean["meal_code"]))
        self.assertIn("Self Catering", set(self.clean["meal"]))

    def test_adr_is_winsorised_and_original_kept(self):
        self.assertLessEqual(self.clean["adr"].max(), 1000)
        self.assertGreaterEqual(self.clean["adr"].min(), 0)
        self.assertEqual(self.clean["adr_raw"].max(), 5400.0)
        self.assertEqual(self.clean["adr_raw"].min(), -6.38)

    def test_agent_null_becomes_a_flag_not_a_zero(self):
        # Zero-filling an ID would invent a real agent called 0.
        self.assertIn("has_company", self.clean.columns)
        self.assertFalse(self.clean["has_company"].any())
        self.assertTrue(self.clean["has_agent"].all())

    def test_arrival_date_is_constructed(self):
        first = self.clean.iloc[0]
        self.assertEqual(first["arrival_date"], pd.Timestamp("2016-01-15"))

    def test_duplicates_are_flagged_by_default(self):
        self.assertTrue(self.clean["is_duplicate_row"].any())
        # Flagged, not removed: the row is still there.
        self.assertEqual(len(self.clean), 5)

    def test_duplicates_can_be_dropped_by_config(self):
        dropped = T.clean(self.raw, Config(drop_exact_duplicates=True))
        self.assertFalse(dropped["is_duplicate_row"].any())
        self.assertEqual(len(dropped), 4)

    def test_zero_night_stays_are_flagged_not_dropped(self):
        self.assertTrue(self.clean["is_zero_night"].any())


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = T.transform(make_raw(), Config())

    def test_booking_id_is_unique_and_one_based(self):
        self.assertEqual(self.df["booking_id"].nunique(), len(self.df))
        self.assertEqual(self.df["booking_id"].min(), 1)

    def test_totals_agree_with_their_components(self):
        self.assertTrue((self.df["total_nights"] ==
                         self.df["stays_in_weekend_nights"] +
                         self.df["stays_in_week_nights"]).all())
        self.assertTrue((self.df["total_guests"] ==
                         self.df["adults"] + self.df["children"] +
                         self.df["babies"]).all())

    def test_revenue_never_double_counts(self):
        both = ((self.df["realized_revenue"] > 0) &
                (self.df["lost_revenue"] > 0)).sum()
        self.assertEqual(both, 0)

    def test_cancelled_bookings_earn_nothing(self):
        cancelled = self.df[self.df["is_canceled"]]
        self.assertGreater(len(cancelled), 0)
        self.assertEqual(cancelled["realized_revenue"].sum(), 0.0)
        self.assertTrue((cancelled["lost_revenue"] > 0).all())

    def test_booking_value_is_rate_times_nights(self):
        expected = (self.df["adr"] * self.df["total_nights"]).round(2)
        pd.testing.assert_series_equal(
            self.df["booking_value"], expected, check_names=False
        )

    def test_lead_time_buckets_are_ordered_and_complete(self):
        self.assertNotIn("nan", set(self.df["lead_time_bucket"]))
        row = self.df[self.df["lead_time"] == 400].iloc[0]
        self.assertEqual(row["lead_time_bucket"], "365+ days")
        row = self.df[self.df["lead_time"] == 0].iloc[0]
        self.assertEqual(row["lead_time_bucket"], "0-7 days")

    def test_party_type_prefers_family_over_size(self):
        family = self.df[self.df["is_family"]]
        self.assertTrue((family["party_type"] == "Family").all())

    def test_season_maps_to_northern_hemisphere(self):
        by_month = dict(zip(self.df["arrival_month_num"], self.df["season"]))
        self.assertEqual(by_month[1], "Winter")
        self.assertEqual(by_month[8], "Summer")
        self.assertEqual(by_month[12], "Winter")

    def test_room_change_is_detected(self):
        changed = self.df[self.df["room_changed"]]
        self.assertTrue((changed["reserved_room_type"] !=
                         changed["assigned_room_type"]).all())

    def test_no_rows_added_or_lost_by_feature_engineering(self):
        cleaned = T.clean(make_raw(), Config())
        self.assertEqual(len(T.engineer_features(cleaned, Config())),
                         len(cleaned))


class TestWarehouse(unittest.TestCase):
    def setUp(self):
        self.staged = T.transform(make_raw(), Config())
        self.tables = W.build_star(self.staged)

    def test_fact_has_one_row_per_booking(self):
        self.assertEqual(len(self.tables["fact_booking"]), len(self.staged))

    def test_dimension_keys_are_unique(self):
        for name, table in self.tables.items():
            if not name.startswith("dim_"):
                continue
            key = table.columns[0]
            self.assertEqual(table[key].nunique(), len(table),
                             f"{name}.{key} is not unique")

    def test_every_fact_row_resolves_to_a_dimension(self):
        fact = self.tables["fact_booking"]
        for _, (table, key, _attr) in W.DIMENSIONS.items():
            valid = set(self.tables[table][key])
            self.assertTrue(set(fact[key].dropna()).issubset(valid),
                            f"orphaned key in {key}")
            self.assertFalse(fact[key].isna().any(), f"{key} has nulls")

    def test_date_dimension_is_dense(self):
        dim = self.tables["dim_date"]
        self.assertEqual(dim["date_key"].nunique(), len(dim))
        dates = pd.to_datetime(dim["full_date"])
        # A gap-free calendar: n days between first and last, inclusive.
        self.assertEqual(len(dim), (dates.max() - dates.min()).days + 1)

    def test_arrival_keys_point_into_the_calendar(self):
        fact = self.tables["fact_booking"]
        calendar = set(self.tables["dim_date"]["date_key"])
        self.assertTrue(set(fact["arrival_date_key"]).issubset(calendar))

    def test_load_creates_queryable_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            W.load(self.tables, cfg)
            got = W.query("SELECT COUNT(*) AS n FROM v_bookings", cfg)
            # INNER JOINs: equal counts prove referential integrity end to end.
            self.assertEqual(int(got["n"].iloc[0]), len(self.staged))
            monthly = W.query("SELECT * FROM v_monthly_performance", cfg)
            self.assertGreater(len(monthly), 0)

    def test_load_rebuilds_rather_than_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            W.load(self.tables, cfg)
            W.load(self.tables, cfg)  # a second run must not double the rows
            got = W.query("SELECT COUNT(*) AS n FROM fact_booking", cfg)
            self.assertEqual(int(got["n"].iloc[0]), len(self.staged))


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, None]})

    def test_passing_check_is_not_a_failure(self):
        s = V.Suite("t")
        s.expect_unique(self.df, "a")
        self.assertEqual(s.failures, [])
        self.assertEqual(s.results[0].status, "PASS")

    def test_error_level_failure_raises(self):
        s = V.Suite("t")
        s.expect_unique(self.df, "b")
        self.assertEqual(len(s.failures), 1)
        with self.assertRaises(ValueError):
            s.raise_if_failed()

    def test_warning_does_not_raise(self):
        s = V.Suite("t")
        s.expect_no_nulls(self.df, ["b"], level=V.WARN)
        self.assertEqual(len(s.failures), 0)
        self.assertEqual(len(s.warnings), 1)
        s.raise_if_failed()  # must not raise

    def test_range_check_detects_outliers(self):
        s = V.Suite("t")
        s.expect_between(self.df, "a", 0, 2)
        self.assertEqual(len(s.failures), 1)

    def test_domain_check_detects_unexpected_values(self):
        s = V.Suite("t")
        s.expect_values_in_set(self.df, "a", {1, 2})
        self.assertEqual(len(s.failures), 1)

    def test_report_is_written(self):
        s = V.Suite("t")
        s.expect_unique(self.df, "a")
        with tempfile.TemporaryDirectory() as tmp:
            md, js = Path(tmp) / "q.md", Path(tmp) / "q.json"
            V.write_report([s], md, js)
            self.assertIn("uniqueness", md.read_text(encoding="utf-8"))
            self.assertIn("PASS", md.read_text(encoding="utf-8"))
            self.assertTrue(js.read_text(encoding="utf-8").startswith("{"))


class TestAnalysis(unittest.TestCase):
    def test_shipped_queries_all_parse_and_run(self):
        """Guards against a typo in a .sql file reaching a real run."""
        staged = T.transform(make_raw(n_extra=20), Config())
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            W.load(W.build_star(staged), cfg)
            queries = load_queries(PROJECT_ROOT / "sql")
            self.assertGreater(len(queries), 10)
            with closing(sqlite3.connect(cfg.paths.database)) as conn:
                for name, sql in queries.items():
                    with self.subTest(query=name):
                        result = pd.read_sql_query(sql, conn)
                        self.assertGreater(len(result), 0,
                                           f"{name} returned no rows")

    def test_insights_are_derived_from_the_tables(self):
        staged = T.transform(make_raw(n_extra=20), Config())
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(paths=Paths(root=Path(tmp)))
            cfg.paths.ensure()
            W.load(W.build_star(staged), cfg)
            queries = load_queries(PROJECT_ROOT / "sql")
            with closing(sqlite3.connect(cfg.paths.database)) as conn:
                results = {n: pd.read_sql_query(q, conn)
                           for n, q in queries.items()}
            insights = derive_insights(results)

        self.assertEqual(insights["total_bookings"], len(staged))
        self.assertGreaterEqual(insights["cancellation_rate_pct"], 0)
        self.assertLessEqual(insights["cancellation_rate_pct"], 100)
        # The headline split must reconcile with the underlying rows.
        self.assertAlmostEqual(insights["realized_revenue"],
                               staged["realized_revenue"].sum(), places=1)
        self.assertAlmostEqual(insights["lost_revenue"],
                               staged["lost_revenue"].sum(), places=1)
        for key in ("riskiest_segment", "safest_segment", "top_country"):
            self.assertIsInstance(insights[key], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
