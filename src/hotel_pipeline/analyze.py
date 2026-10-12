"""Stage 4 - Analyse.

Runs every query in ``sql/`` against the warehouse and returns the results as a
dict of DataFrames.

The analysis lives in ``.sql`` files rather than in Python strings on purpose:
they are readable without running anything, they can be pasted straight into a
SQL client against the same database, and adding an analysis means adding a
file - no code change.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)


def load_queries(sql_dir: Path) -> dict[str, str]:
    """Read every ``.sql`` file in the directory, keyed by filename stem."""
    files = sorted(sql_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {sql_dir}")
    return {f.stem: f.read_text(encoding="utf-8") for f in files}


def run_analyses(cfg: Config = DEFAULT_CONFIG) -> dict[str, pd.DataFrame]:
    """Execute all analysis queries against the warehouse."""
    db_path = cfg.paths.database
    if not db_path.exists():
        raise FileNotFoundError(
            f"Warehouse not found at {db_path}. Run the load stage first."
        )

    queries = load_queries(cfg.paths.sql)
    results: dict[str, pd.DataFrame] = {}
    with closing(sqlite3.connect(db_path)) as conn:
        for name, sql in queries.items():
            try:
                results[name] = pd.read_sql_query(sql, conn)
            except Exception as exc:  # one bad query must not sink the run
                log.error("Query %s failed: %s", name, exc)
                continue
            log.info("  %-34s %4d row(s)", name, len(results[name]))
    log.info("Ran %d/%d analyses", len(results), len(queries))
    return results


def derive_insights(results: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Reduce the result tables to the handful of numbers the report leads with.

    Kept separate from the SQL so the narrative in the report is generated from
    the data every run, rather than being a hand-written claim that silently
    goes stale when the source updates.
    """
    kpi = results["overview_kpis"].iloc[0]
    insights: dict[str, object] = {
        "total_bookings": int(kpi["total_bookings"]),
        "cancellation_rate_pct": float(kpi["cancellation_rate_pct"]),
        "avg_daily_rate": float(kpi["avg_daily_rate"]),
        "avg_lead_time_days": float(kpi["avg_lead_time_days"]),
        "realized_revenue": float(kpi["realized_revenue"]),
        "lost_revenue": float(kpi["lost_revenue"]),
        "period_start": str(kpi["period_start"]),
        "period_end": str(kpi["period_end"]),
        "countries": int(kpi["countries"]),
    }
    total_demand = insights["realized_revenue"] + insights["lost_revenue"]
    insights["revenue_at_risk_pct"] = round(
        100 * insights["lost_revenue"] / total_demand, 2
    ) if total_demand else 0.0

    # Lead time: compare the shortest and longest booking windows.
    lead = results["cancellation_by_lead_time"]
    insights["lead_time_low"] = lead.iloc[0]["lead_time_bucket"]
    insights["lead_time_low_rate"] = float(lead.iloc[0]["cancellation_rate_pct"])
    insights["lead_time_high"] = lead.iloc[-1]["lead_time_bucket"]
    insights["lead_time_high_rate"] = float(lead.iloc[-1]["cancellation_rate_pct"])

    # Hotels: name each property's rate so the report can contrast them.
    hotels = results["performance_by_hotel"].set_index("hotel_name")
    insights["hotel_rates"] = {
        name: float(row["cancellation_rate_pct"])
        for name, row in hotels.iterrows()
    }
    insights["hotel_adr"] = {
        name: float(row["avg_daily_rate"]) for name, row in hotels.iterrows()
    }

    # Segments: the riskiest channel that still carries meaningful volume.
    seg = results["cancellation_by_segment"]
    material = seg[seg["bookings"] >= 0.01 * insights["total_bookings"]]
    riskiest = material.nlargest(1, "cancellation_rate_pct").iloc[0]
    insights["riskiest_segment"] = str(riskiest["market_segment"])
    insights["riskiest_segment_rate"] = float(riskiest["cancellation_rate_pct"])
    insights["riskiest_segment_share"] = float(riskiest["share_of_bookings_pct"])
    safest = material.nsmallest(1, "cancellation_rate_pct").iloc[0]
    insights["safest_segment"] = str(safest["market_segment"])
    insights["safest_segment_rate"] = float(safest["cancellation_rate_pct"])

    # Deposits: the finding that contradicts the obvious expectation.
    dep = results["cancellation_by_deposit"].set_index("deposit_type")
    insights["deposit_rates"] = {
        name: float(row["cancellation_rate_pct"]) for name, row in dep.iterrows()
    }

    # Special requests: the big drop is from zero to any; beyond that the
    # effect is flat-to-mild, so report the spread rather than a trend.
    req = results["special_requests_vs_cancellation"]
    insights["requests_zero_rate"] = float(req.iloc[0]["cancellation_rate_pct"])
    insights["requests_many_rate"] = float(req.iloc[-1]["cancellation_rate_pct"])

    # Seasonality: the peak and trough months by rate, resort side.
    adr = results["adr_by_month_hotel"]
    resort = adr[adr["hotel_name"] == "Resort Hotel"]
    if len(resort):
        insights["peak_adr_month"] = str(
            resort.loc[resort["avg_daily_rate"].idxmax(), "arrival_month"]
        )
        insights["trough_adr_month"] = str(
            resort.loc[resort["avg_daily_rate"].idxmin(), "arrival_month"]
        )

    # Top source market.
    top = results["top_countries"].iloc[0]
    insights["top_country"] = str(top["country_code"])
    insights["top_country_share"] = float(top["share_of_bookings_pct"])
    insights["top_country_rate"] = float(top["cancellation_rate_pct"])

    return insights


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def export(results: dict[str, pd.DataFrame], insights: dict[str, object],
           cfg: Config = DEFAULT_CONFIG) -> None:
    """Write every result table to CSV and the insights to JSON.

    CSV so the tables open in a spreadsheet without touching Python; JSON so a
    dashboard or a downstream job can read the headline numbers directly.
    """
    tables_dir = cfg.paths.reports / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    cfg.paths.metrics_json.write_text(
        json.dumps(insights, indent=2, default=_json_safe), encoding="utf-8"
    )
    log.info("Exported %d table(s) -> %s", len(results), tables_dir.name)
    log.info("Headline metrics -> %s", cfg.paths.metrics_json.name)


def analyze(cfg: Config = DEFAULT_CONFIG
            ) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Full stage 4: run queries, derive insights, export both."""
    log.info("Running analyses")
    results = run_analyses(cfg)
    insights = derive_insights(results)
    export(results, insights, cfg)
    return results, insights
