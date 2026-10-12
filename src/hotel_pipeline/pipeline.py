"""The orchestrator: runs the stages in order, with a quality gate between them.

    extract -> [validate raw] -> transform -> [validate staged] -> load
            -> analyze -> report

The two validation suites ask different questions. The raw suite asks *is the
source still what we think it is* - it fails loudly when upstream changes shape,
before a single transform rule has run on bad assumptions. The staged suite asks
*did our own transform do what it claimed* - it is a regression test on the
cleaning rules, checking the nulls are gone and the derived columns are sane.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import analyze as analyze_stage
from . import extract as extract_stage
from . import report as report_stage
from . import transform as transform_stage
from . import validate as validate_stage
from . import warehouse as warehouse_stage
from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger, setup_logging
from .validate import ERROR, INFO, WARN, Suite

log = get_logger(__name__)


# --- Expectation suites ---------------------------------------------------

def raw_suite(df: pd.DataFrame) -> Suite:
    """Contract with the upstream source."""
    s = Suite("Raw source")
    s.expect_columns(df, extract_stage.EXPECTED_COLUMNS)
    s.expect_row_count_between(df, 100_000, 200_000)
    s.expect_values_in_set(df, "hotel", {"City Hotel", "Resort Hotel"})
    s.expect_values_in_set(df, "is_canceled", {0, 1})
    s.expect_values_in_set(
        df, "reservation_status", {"Check-Out", "Canceled", "No-Show"}
    )
    s.expect_values_in_set(
        df, "deposit_type", {"No Deposit", "Non Refund", "Refundable"}
    )
    s.expect_between(df, "lead_time", 0, 1000)
    s.expect_between(df, "arrival_date_year", 2014, 2020)
    s.expect_between(df, "adults", 0, 60)
    s.expect_no_nulls(df, ["hotel", "is_canceled", "lead_time", "adr"])

    # Known messiness: recorded so a change in its scale is visible, but not
    # treated as a failure - the transform stage is what resolves these.
    s.expect_null_rate_below(df, "children", 0.001, level=WARN)
    s.expect_null_rate_below(df, "country", 0.01, level=WARN)
    # Two rows in the source have absurd ADR values and always will. Gate on
    # the rate, not the presence, so `--strict` stays a usable CI signal.
    s.expect_outlier_rate_below(df, "adr", 0, 1000, max_rate=0.001, level=WARN)
    s.expect(
        "known issue: zero-occupant bookings", df,
        lambda d: (True, f"{int(((d.adults + d.children.fillna(0) + d.babies) == 0).sum()):,} rows"),
        "tracked, removed in staging", level=INFO,
    )
    s.expect(
        "known issue: exact duplicate rows", df,
        lambda d: (True, f"{int(d.duplicated().sum()):,} rows"),
        "tracked, flagged in staging", level=INFO,
    )
    return s


def staged_suite(df: pd.DataFrame, raw_rows: int) -> Suite:
    """Regression test on the transform rules."""
    s = Suite("Staged data")
    s.expect_unique(df, "booking_id")
    s.expect_row_count_between(df, int(raw_rows * 0.95), raw_rows)

    # The cleaning rules promised these would be resolved.
    s.expect_no_nulls(df, ["children", "country", "arrival_date",
                           "total_guests", "total_nights", "adr",
                           "lead_time_bucket", "season", "booking_value"])
    s.expect_between(df, "adr", 0, 1000)
    s.expect_between(df, "total_guests", 1, 60)
    s.expect_between(df, "total_nights", 0, 70)
    s.expect_between(df, "adr_per_guest", 0, 1000)
    s.expect_values_in_set(df, "season",
                           {"Winter", "Spring", "Summer", "Autumn"})
    s.expect_values_in_set(df, "party_type",
                           {"Solo", "Couple", "Family", "Group"})

    # Derived columns must agree with the columns they were derived from.
    s.expect(
        "consistency: total_nights = weekend + week nights", df,
        lambda d: (
            bool((d.total_nights ==
                  d.stays_in_weekend_nights + d.stays_in_week_nights).all()),
            "all rows agree" if (d.total_nights ==
                                 d.stays_in_weekend_nights + d.stays_in_week_nights).all()
            else "mismatched rows present",
        ),
        "exact match on every row",
    )
    s.expect(
        "consistency: total_guests = adults + children + babies", df,
        lambda d: (
            bool((d.total_guests == d.adults + d.children + d.babies).all()),
            "all rows agree" if (d.total_guests ==
                                 d.adults + d.children + d.babies).all()
            else "mismatched rows present",
        ),
        "exact match on every row",
    )
    s.expect(
        "consistency: revenue splits by cancellation", df,
        lambda d: (
            bool(((d.realized_revenue > 0) & (d.lost_revenue > 0)).sum() == 0),
            f"{int(((d.realized_revenue > 0) & (d.lost_revenue > 0)).sum())} "
            f"rows counted in both",
            ),
        "no booking is both realised and lost",
    )
    s.expect(
        "consistency: cancelled bookings earn nothing", df,
        lambda d: (
            bool(d.loc[d.is_canceled, "realized_revenue"].sum() == 0),
            f"cancelled realised revenue = "
            f"{d.loc[d.is_canceled, 'realized_revenue'].sum():.2f}",
        ),
        "exactly 0",
    )
    s.expect(
        "plausibility: cancellation rate", df,
        lambda d: (0.2 <= d.is_canceled.mean() <= 0.5,
                   f"{d.is_canceled.mean():.2%}"),
        "between 20% and 50%", level=WARN,
    )
    return s


def warehouse_suite(cfg: Config) -> Suite:
    """Referential integrity: no fact row may point at a missing dimension."""
    s = Suite("Warehouse")
    fact_rows = warehouse_stage.query(
        "SELECT COUNT(*) AS n FROM fact_booking", cfg
    )["n"].iloc[0]
    view_rows = warehouse_stage.query(
        "SELECT COUNT(*) AS n FROM v_bookings", cfg
    )["n"].iloc[0]

    # Every join in v_bookings is an INNER JOIN, so a row lost between the fact
    # table and the view is precisely an orphaned foreign key.
    s.check("integrity: no orphaned dimension keys", fact_rows == view_rows,
            f"{fact_rows:,} fact rows -> {view_rows:,} joined rows",
            "no rows lost in the join")

    dupes = warehouse_stage.query(
        "SELECT COUNT(*) AS n FROM (SELECT booking_id FROM fact_booking "
        "GROUP BY booking_id HAVING COUNT(*) > 1)", cfg
    )["n"].iloc[0]
    s.check("uniqueness: fact_booking.booking_id", dupes == 0,
            f"{dupes:,} duplicated key(s)", "0 duplicates")

    nulls = warehouse_stage.query(
        "SELECT COUNT(*) AS n FROM fact_booking WHERE arrival_date_key IS NULL "
        "OR hotel_key IS NULL OR country_key IS NULL", cfg
    )["n"].iloc[0]
    s.check("completeness: mandatory foreign keys", nulls == 0,
            f"{nulls:,} row(s) with a null key", "0 nulls")
    return s


# --- Runner ---------------------------------------------------------------

@dataclass
class RunResult:
    rows_raw: int
    rows_staged: int
    database: Path
    quality_report: Path
    analysis_report: Path
    duration_seconds: float
    warnings: int


def run(cfg: Config = DEFAULT_CONFIG, force_download: bool = False,
        strict: bool = False, with_model: bool = False) -> RunResult:
    """Execute the whole pipeline.

    ``strict`` promotes warnings to failures - the right setting for a
    scheduled run, where silently degrading data should stop the job.

    ``with_model`` adds the training stage. It is off by default because
    isotonic calibration refits the booster five times and takes several
    minutes, which is the wrong default for a run whose other five stages
    finish in well under a minute.
    """
    started = time.perf_counter()
    cfg.paths.ensure()
    log_file = setup_logging(cfg.paths.logs)
    log.info("=" * 72)
    log.info("Hotel booking pipeline - run starting")
    log.info("Log file: %s", log_file.name)
    log.info("=" * 72)

    # 1. Extract
    log.info("[1/5] Extract")
    raw = extract_stage.extract(cfg, force=force_download)

    # 2. Validate the source, then transform
    log.info("[2/5] Validate source")
    suite_raw = raw_suite(raw)
    suite_raw.log_summary()
    suite_raw.raise_if_failed()

    log.info("[3/5] Transform")
    staged = transform_stage.transform(raw, cfg)
    staged.to_parquet(cfg.paths.staged_parquet, index=False)
    log.info("Staged data -> %s", cfg.paths.staged_parquet.name)

    suite_staged = staged_suite(staged, raw_rows=len(raw))
    suite_staged.log_summary()
    suite_staged.raise_if_failed()

    # 3. Load
    log.info("[4/5] Load warehouse")
    tables = warehouse_stage.build_star(staged)
    warehouse_stage.load(tables, cfg)
    suite_wh = warehouse_suite(cfg)
    suite_wh.log_summary()
    suite_wh.raise_if_failed()

    suites = [suite_raw, suite_staged, suite_wh]
    validate_stage.write_report(suites, cfg.paths.quality_report,
                                cfg.paths.quality_json)
    warnings = sum(len(s.warnings) for s in suites)
    if strict and warnings:
        raise ValueError(
            f"strict mode: {warnings} warning(s) treated as failures. "
            f"See {cfg.paths.quality_report}."
        )

    # 4. Analyse
    total = 6 if with_model else 5
    log.info("[5/%d] Analyse", total)
    results, insights = analyze_stage.analyze(cfg)

    # 5. Train (opt-in), so the report can include the model's own evaluation
    if with_model:
        log.info("[6/%d] Train cancellation model", total)
        from . import model as model_stage
        trained, _ = model_stage.train(cfg)
        model_stage.score_warehouse(trained, cfg)

    log.info("Report")
    report_path = report_stage.report(results, insights, cfg)

    duration = time.perf_counter() - started
    log.info("=" * 72)
    log.info("Run complete in %.1fs - %s raw -> %s staged rows, %d warning(s)",
             duration, f"{len(raw):,}", f"{len(staged):,}", warnings)
    log.info("  warehouse : %s", cfg.paths.database)
    log.info("  quality   : %s", cfg.paths.quality_report)
    log.info("  report    : %s", report_path)
    log.info("=" * 72)

    return RunResult(
        rows_raw=len(raw), rows_staged=len(staged),
        database=cfg.paths.database, quality_report=cfg.paths.quality_report,
        analysis_report=report_path, duration_seconds=duration,
        warnings=warnings,
    )
