"""Stage 2 - Transform.

Cleans the raw frame and derives the analytical features the warehouse and the
report are built on. Every rule here is a judgement call about real-world
messiness, so each one says *why* it exists and, where a row is changed rather
than dropped, the original value is preserved alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_TO_NUM = {name: i + 1 for i, name in enumerate(MONTHS)}

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}

MEAL_PLANS = {"BB": "Bed & Breakfast", "HB": "Half Board", "FB": "Full Board",
              "SC": "Self Catering", "Undefined": "Self Catering"}

UNKNOWN = "Unknown"


def _log_drop(before: int, after: int, reason: str) -> None:
    removed = before - after
    log.info("  dropped %6s rows (%.2f%%) - %s",
             f"{removed:,}", 100 * removed / max(before, 1), reason)


def clean(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Apply the cleaning policy. Returns a new frame; the input is untouched."""
    df = df.copy()
    start_rows = len(df)
    log.info("Cleaning %s raw rows", f"{start_rows:,}")

    # --- 1. Whitespace ----------------------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # --- 2. Missing values, each with its own meaning ---------------------
    # `children` is null for 4 rows; every other party-size field says 0 there,
    # so 0 is the only reading consistent with the row.
    df["children"] = df["children"].fillna(0).astype("int64")

    # A missing country is genuinely unknown, not zero. Keep the rows - they
    # are valid bookings - but label them so they never pass as a real country.
    df["country"] = df["country"].fillna(UNKNOWN).replace("NULL", UNKNOWN)

    # `agent`/`company` are IDs, not quantities: null means "no intermediary",
    # which is information. Encode that as a flag and keep the ID nullable.
    df["has_agent"] = df["agent"].notna()
    df["has_company"] = df["company"].notna()
    df["agent_id"] = df["agent"].astype("Int64")
    df["company_id"] = df["company"].astype("Int64")
    df = df.drop(columns=["agent", "company"])

    # --- 3. Category domains ----------------------------------------------
    # "Undefined" in `meal` is the source spelling of self-catering (no meal
    # package), which is what SC already means - collapse the two.
    df["meal_code"] = df["meal"].replace("Undefined", "SC")
    df["meal"] = df["meal_code"].map(MEAL_PLANS).fillna(UNKNOWN)
    for col in ("market_segment", "distribution_channel"):
        df[col] = df[col].replace("Undefined", UNKNOWN)

    # --- 4. Dates ----------------------------------------------------------
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_TO_NUM)
    df["arrival_date"] = pd.to_datetime(
        dict(year=df["arrival_date_year"],
             month=df["arrival_month_num"],
             day=df["arrival_date_day_of_month"]),
        errors="coerce",
    )
    df["reservation_status_date"] = pd.to_datetime(
        df["reservation_status_date"], errors="coerce"
    )
    bad_dates = int(df["arrival_date"].isna().sum())
    if bad_dates:
        before = len(df)
        df = df[df["arrival_date"].notna()]
        _log_drop(before, len(df), f"{bad_dates} unparseable arrival date(s)")

    # --- 5. Party size and stay length -------------------------------------
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["total_nights"] = (df["stays_in_weekend_nights"]
                          + df["stays_in_week_nights"])

    if cfg.drop_zero_guest_bookings:
        before = len(df)
        df = df[df["total_guests"] > 0]
        _log_drop(before, len(df),
                  "zero occupants (data-entry artifacts, not stays)")

    # Zero-night rows are bookings that never became a stay. They are kept -
    # they are real demand signal - but flagged, because any per-night rate
    # computed from them divides by zero.
    df["is_zero_night"] = df["total_nights"] == 0

    # --- 6. Rate outliers ---------------------------------------------------
    # One row lists ADR 5400 and one is negative; both are entry errors that
    # would dominate any mean. Winsorise for analysis, keep the original.
    df["adr_raw"] = df["adr"]
    df["adr"] = df["adr"].clip(lower=cfg.adr_floor, upper=cfg.adr_ceiling)
    adjusted = int((df["adr"] != df["adr_raw"]).sum())
    if adjusted:
        log.info("  winsorised %d ADR value(s) into [%g, %g]",
                 adjusted, cfg.adr_floor, cfg.adr_ceiling)

    # --- 7. Exact duplicates ------------------------------------------------
    # ~27% of rows are field-for-field identical. Two guests really can make the
    # same booking on the same day, so deleting them would erase real demand;
    # flagging lets the analyst test the assumption either way.
    key = [c for c in df.columns
           if not c.startswith(("adr_raw", "is_", "has_"))]
    df["is_duplicate_row"] = df.duplicated(subset=key, keep="first")
    log.info("  flagged %s exact duplicate row(s) (kept; set "
             "drop_exact_duplicates=True to exclude)",
             f"{int(df['is_duplicate_row'].sum()):,}")
    if cfg.drop_exact_duplicates:
        before = len(df)
        df = df[~df["is_duplicate_row"]]
        _log_drop(before, len(df), "exact duplicates (per config)")

    df = df.reset_index(drop=True)
    log.info("Cleaning complete: %s -> %s rows (%.1f%% retained)",
             f"{start_rows:,}", f"{len(df):,}", 100 * len(df) / start_rows)
    return df


def engineer_features(df: pd.DataFrame,
                      cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Derive the analytical columns. Pure: no rows added or removed."""
    df = df.copy()
    log.info("Engineering features")

    # Stable surrogate key. Row order is deterministic from the source file, so
    # the same input always yields the same IDs.
    df.insert(0, "booking_id", np.arange(1, len(df) + 1, dtype="int64"))

    # --- Calendar -----------------------------------------------------------
    df["arrival_year_month"] = df["arrival_date"].dt.to_period("M").astype(str)
    df["arrival_dow"] = df["arrival_date"].dt.day_name()
    df["arrival_quarter"] = df["arrival_date"].dt.quarter
    df["season"] = df["arrival_month_num"].map(SEASONS)
    df["is_weekend_arrival"] = df["arrival_date"].dt.dayofweek >= 5
    df["departure_date"] = df["arrival_date"] + pd.to_timedelta(
        df["total_nights"], unit="D"
    )

    # --- Booking behaviour --------------------------------------------------
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"], bins=list(cfg.lead_time_bins),
        labels=list(cfg.lead_time_labels), right=True, include_lowest=True,
    ).astype(str)

    df["stay_type"] = np.select(
        [(df["stays_in_week_nights"] > 0) & (df["stays_in_weekend_nights"] > 0),
         df["stays_in_week_nights"] > 0,
         df["stays_in_weekend_nights"] > 0],
        ["Mixed", "Weekday only", "Weekend only"],
        default="No overnight stay",
    )

    df["is_family"] = (df["children"] + df["babies"]) > 0
    df["party_type"] = np.select(
        [df["is_family"], df["total_guests"] == 1, df["total_guests"] == 2],
        ["Family", "Solo", "Couple"], default="Group",
    )

    # An assigned room different from the one reserved is an upgrade, a
    # downgrade, or an operational scramble - all worth isolating.
    df["room_changed"] = df["reserved_room_type"] != df["assigned_room_type"]
    df["is_loyal_guest"] = df["is_repeated_guest"] == 1
    df["has_prior_cancellation"] = df["previous_cancellations"] > 0
    df["requires_parking"] = df["required_car_parking_spaces"] > 0

    # --- Money --------------------------------------------------------------
    # ADR is the average nightly rate, so the value of a booking is rate x
    # nights. A cancelled booking earns nothing but represents demand that was
    # there and then lost - the two are tracked separately and never summed.
    df["booking_value"] = (df["adr"] * df["total_nights"]).round(2)
    canceled = df["is_canceled"] == 1
    df["realized_revenue"] = df["booking_value"].where(~canceled, 0.0)
    df["lost_revenue"] = df["booking_value"].where(canceled, 0.0)
    df["adr_per_guest"] = (
        df["adr"] / df["total_guests"].replace(0, np.nan)
    ).round(2)

    df["is_canceled"] = df["is_canceled"].astype(bool)
    df["is_no_show"] = df["reservation_status"] == "No-Show"

    log.info("Feature engineering complete: %d columns", df.shape[1])
    return df


def transform(df: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Full stage 2: clean, then derive features."""
    return engineer_features(clean(df, cfg), cfg)
