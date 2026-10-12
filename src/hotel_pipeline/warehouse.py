"""Stage 3 - Load.

Reshapes the staged flat table into a star schema and writes it to SQLite.

Why a star schema for a 119k-row table that fits in memory? Because the point of
this layer is not speed, it is *shared vocabulary*: one row per booking in
`fact_booking`, every label resolved through a dimension, so an ad-hoc SQL
question and the Python report cannot quietly disagree about what "Direct" or
"Online TA" means.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

# source column -> (dimension table, surrogate-key column, attribute column)
DIMENSIONS = {
    "hotel": ("dim_hotel", "hotel_key", "hotel_name"),
    "country": ("dim_country", "country_key", "country_code"),
    "market_segment": ("dim_market_segment", "market_segment_key",
                       "market_segment"),
    "distribution_channel": ("dim_distribution_channel",
                             "distribution_channel_key",
                             "distribution_channel"),
    "customer_type": ("dim_customer_type", "customer_type_key",
                      "customer_type"),
    "deposit_type": ("dim_deposit_type", "deposit_type_key", "deposit_type"),
    "meal": ("dim_meal", "meal_key", "meal_plan"),
}

# Degenerate attributes: low-cardinality labels kept on the fact table because
# a dimension for each would be a join with nothing on the other side.
FACT_ATTRIBUTES = [
    "lead_time", "lead_time_bucket", "stay_type", "party_type", "season",
    "arrival_dow", "arrival_year_month", "reserved_room_type",
    "assigned_room_type", "reservation_status", "booking_changes",
    "days_in_waiting_list", "previous_cancellations",
    "previous_bookings_not_canceled", "agent_id", "company_id",
]

FACT_MEASURES = [
    "adults", "children", "babies", "total_guests",
    "stays_in_weekend_nights", "stays_in_week_nights", "total_nights",
    "adr", "adr_per_guest", "booking_value", "realized_revenue",
    "lost_revenue", "required_car_parking_spaces", "total_of_special_requests",
]

FACT_FLAGS = [
    "is_canceled", "is_no_show", "is_loyal_guest", "has_prior_cancellation",
    "room_changed", "is_family", "requires_parking", "is_weekend_arrival",
    "is_zero_night", "is_duplicate_row", "has_agent", "has_company",
]


def _date_key(series: pd.Series) -> pd.Series:
    """YYYYMMDD integer key. Readable in raw SQL, and sorts chronologically."""
    return series.dt.strftime("%Y%m%d").astype("Int64")


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    """A dense calendar spanning every date the fact table can reference.

    Dense, not just the dates that appear: a gap-free calendar is what lets a
    time series show a day with zero bookings as a zero instead of omitting it.
    """
    dates = pd.concat([df["arrival_date"], df["reservation_status_date"]])
    dates = dates.dropna()
    calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    dim = pd.DataFrame({"full_date": calendar})
    dim.insert(0, "date_key", _date_key(dim["full_date"]))
    dim["year"] = dim["full_date"].dt.year
    dim["quarter"] = dim["full_date"].dt.quarter
    dim["month_num"] = dim["full_date"].dt.month
    dim["month_name"] = dim["full_date"].dt.month_name()
    dim["year_month"] = dim["full_date"].dt.to_period("M").astype(str)
    dim["day_of_month"] = dim["full_date"].dt.day
    dim["day_name"] = dim["full_date"].dt.day_name()
    dim["week_of_year"] = dim["full_date"].dt.isocalendar().week.astype(int)
    dim["is_weekend"] = (dim["full_date"].dt.dayofweek >= 5).astype(int)
    dim["season"] = dim["month_num"].map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
         5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
         10: "Autumn", 11: "Autumn"}
    )
    dim["full_date"] = dim["full_date"].dt.strftime("%Y-%m-%d")
    return dim


def build_dimension(df: pd.DataFrame, source_col: str, key_col: str,
                    attr_col: str) -> pd.DataFrame:
    """One surrogate key per distinct value, assigned in sorted order.

    Sorted rather than first-seen so the keys are reproducible across runs.
    """
    values = sorted(df[source_col].dropna().unique())
    return pd.DataFrame({
        key_col: range(1, len(values) + 1),
        attr_col: values,
    })


def build_star(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the staged frame into dimension tables plus one fact table."""
    log.info("Building star schema from %s staged rows", f"{len(df):,}")
    tables: dict[str, pd.DataFrame] = {"dim_date": build_dim_date(df)}

    fact = pd.DataFrame({"booking_id": df["booking_id"]})
    fact["arrival_date_key"] = _date_key(df["arrival_date"])
    fact["status_date_key"] = _date_key(df["reservation_status_date"])

    for source_col, (table, key_col, attr_col) in DIMENSIONS.items():
        dim = build_dimension(df, source_col, key_col, attr_col)
        tables[table] = dim
        lookup = dict(zip(dim[attr_col], dim[key_col]))
        fact[key_col] = df[source_col].map(lookup).astype("Int64")
        log.info("  %-28s %4d member(s)", table, len(dim))

    for col in FACT_ATTRIBUTES + FACT_MEASURES:
        fact[col] = df[col]
    for col in FACT_FLAGS:
        fact[col] = df[col].astype(int)  # SQLite has no boolean type

    tables["fact_booking"] = fact
    log.info("  %-28s %s row(s) x %d column(s)", "fact_booking",
             f"{len(fact):,}", fact.shape[1])
    return tables


DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_fact_arrival ON fact_booking(arrival_date_key)",
    "CREATE INDEX IF NOT EXISTS ix_fact_hotel ON fact_booking(hotel_key)",
    "CREATE INDEX IF NOT EXISTS ix_fact_country ON fact_booking(country_key)",
    "CREATE INDEX IF NOT EXISTS ix_fact_segment ON fact_booking(market_segment_key)",
    "CREATE INDEX IF NOT EXISTS ix_fact_canceled ON fact_booking(is_canceled)",
]

# One denormalised view so ad-hoc SQL does not have to re-type seven joins.
DDL_VIEWS = [
    """
    CREATE VIEW IF NOT EXISTS v_bookings AS
    SELECT f.booking_id,
           d.full_date        AS arrival_date,
           d.year             AS arrival_year,
           d.month_name       AS arrival_month,
           d.year_month       AS arrival_year_month,
           d.day_name         AS arrival_day_name,
           d.season,
           h.hotel_name,
           c.country_code,
           ms.market_segment,
           dc.distribution_channel,
           ct.customer_type,
           dt.deposit_type,
           m.meal_plan,
           f.lead_time, f.lead_time_bucket, f.stay_type, f.party_type,
           f.reserved_room_type, f.assigned_room_type, f.room_changed,
           f.total_guests, f.total_nights, f.adr, f.booking_value,
           f.realized_revenue, f.lost_revenue,
           f.total_of_special_requests, f.required_car_parking_spaces,
           f.is_canceled, f.is_no_show, f.is_loyal_guest, f.is_duplicate_row
    FROM fact_booking f
    JOIN dim_date d               ON f.arrival_date_key = d.date_key
    JOIN dim_hotel h              ON f.hotel_key = h.hotel_key
    JOIN dim_country c            ON f.country_key = c.country_key
    JOIN dim_market_segment ms    ON f.market_segment_key = ms.market_segment_key
    JOIN dim_distribution_channel dc
                                  ON f.distribution_channel_key = dc.distribution_channel_key
    JOIN dim_customer_type ct     ON f.customer_type_key = ct.customer_type_key
    JOIN dim_deposit_type dt      ON f.deposit_type_key = dt.deposit_type_key
    JOIN dim_meal m               ON f.meal_key = m.meal_key
    """,
    """
    CREATE VIEW IF NOT EXISTS v_monthly_performance AS
    SELECT arrival_year_month,
           hotel_name,
           COUNT(*)                                   AS bookings,
           SUM(is_canceled)                           AS cancellations,
           ROUND(AVG(is_canceled) * 100, 2)           AS cancellation_rate_pct,
           ROUND(AVG(adr), 2)                         AS avg_adr,
           ROUND(SUM(realized_revenue), 2)            AS realized_revenue,
           ROUND(SUM(lost_revenue), 2)                AS lost_revenue
    FROM v_bookings
    GROUP BY arrival_year_month, hotel_name
    """,
]


def load(tables: dict[str, pd.DataFrame],
         cfg: Config = DEFAULT_CONFIG) -> Path:
    """Write every table to SQLite, then add indexes and views.

    The database is rebuilt from scratch each run: the pipeline is a pure
    function of the raw snapshot, so a partial old state is never worth keeping.
    """
    db_path = cfg.paths.database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    # `closing` as well as the transaction context: sqlite3's own context
    # manager commits but does NOT close, and a leaked handle blocks the next
    # run from replacing the file on Windows.
    with closing(sqlite3.connect(db_path)) as conn:
        for name, table in tables.items():
            table.to_sql(name, conn, index=False, if_exists="replace")
        for statement in DDL_INDEXES + DDL_VIEWS:
            conn.execute(statement)
        conn.commit()
        conn.execute("ANALYZE")

    size_mb = db_path.stat().st_size / 1e6
    log.info("Warehouse written -> %s (%.1f MB, %d tables + %d views)",
             db_path.name, size_mb, len(tables), len(DDL_VIEWS))
    return db_path


def query(sql: str, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Run a read-only query against the warehouse."""
    with closing(sqlite3.connect(cfg.paths.database)) as conn:
        return pd.read_sql_query(sql, conn)
