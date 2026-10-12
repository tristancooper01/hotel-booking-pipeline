"""What the model is allowed to know, and when it is allowed to know it.

This module exists so that the leakage rules are *data*, enforced in one place,
rather than a thing someone has to remember while writing a training script.

Features are sorted into three tiers by **when the value becomes known**:

    SAFE        known the moment the booking is created
    PRE_ARRIVAL known before arrival but after booking - valid only for a model
                that re-scores an existing booking, never for one that scores at
                booking time
    LEAKED      known only at or after the outcome. Never usable, at all.

The LEAKED tier is the important one. `room_changed` correlates ferociously with
cancellation (about 5% versus 45%) and is completely worthless: rooms are
assigned at check-in, so the feature really means "the guest turned up". A model
trained on it scores beautifully and predicts nothing. Several columns here are
that same trap in different clothing.
"""

from __future__ import annotations

TARGET = "is_canceled"

# --- Tier 1: known when the booking is made -------------------------------

SAFE_NUMERIC = [
    "lead_time",
    "arrival_month_num",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "adr",
    "booking_value",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "total_of_special_requests",
    "is_repeated_guest",
]

SAFE_CATEGORICAL = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "season",
    "arrival_dow",
    "party_type",
    "stay_type",
    "has_agent",
    "has_company",
]

# --- Tier 2: accumulates between booking and arrival ----------------------
# Legitimate for a model that re-scores a live booking each night; wrong for one
# that scores at the moment of booking, when both are zero by definition.

PRE_ARRIVAL_NUMERIC = [
    "booking_changes",
    "days_in_waiting_list",
]

# --- Tier 3: never ---------------------------------------------------------

LEAKED = {
    "is_canceled": "the target itself",
    "reservation_status": "literally states the outcome (Canceled/Check-Out/No-Show)",
    "reservation_status_date": "the date the outcome was recorded",
    "is_no_show": "a form of the outcome",
    "assigned_room_type": "assigned at check-in, so it implies arrival",
    "room_changed": "derived from the assigned room, so it implies arrival",
    "realized_revenue": "computed from the target",
    "lost_revenue": "computed from the target",
    "departure_date": "only meaningful if the stay happened",
    # Found empirically, not assumed: ALL 7,409 bookings that requested a
    # parking space completed their stay. Exactly zero cancellations is not
    # behaviour, it is structure - the field is evidently populated when a
    # space is allocated at check-in, so like `room_changed` it really means
    # "the guest arrived". Left in the feature set it earns a coefficient
    # near -14 and pins any booking with parking to ~0% risk.
    "required_car_parking_spaces": (
        "perfect separation - 0 cancellations in 7,409 rows; allocated at "
        "check-in, so it implies arrival"
    ),
}

# Excluded for reasons other than leakage: identifiers, pipeline bookkeeping,
# and columns that cannot generalise to a future period.
EXCLUDED = {
    "booking_id": "surrogate key, carries no signal",
    "is_duplicate_row": "pipeline bookkeeping, not a property of the booking",
    "is_zero_night": "fully determined by total_nights",
    "adr_raw": "pre-winsorisation duplicate of adr",
    "arrival_date": "raw date; the model uses month/week/weekday instead",
    "arrival_year_month": "ties the model to specific calendar months",
    "arrival_date_year": "cannot generalise - the scoring year is never in train",
    "arrival_quarter": "redundant with arrival_month_num",
    "arrival_date_day_of_month": "noise at this sample size",
    "arrival_date_month": "redundant with arrival_month_num",
    "agent_id": "high-cardinality identifier; has_agent carries the signal",
    "company_id": "high-cardinality identifier; has_company carries the signal",
    "meal_code": "redundant with meal",
    "lead_time_bucket": "a coarsening of lead_time, which is already included",
    "adr_per_guest": "ratio of two features already present",
    "is_family": "redundant with party_type",
    "is_loyal_guest": "redundant with is_repeated_guest",
    "has_prior_cancellation": "a coarsening of previous_cancellations",
    "requires_parking": "a coarsening of required_car_parking_spaces",
    "is_weekend_arrival": "redundant with arrival_dow",
}


def feature_columns(include_pre_arrival: bool = False
                    ) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names for the chosen tier."""
    numeric = list(SAFE_NUMERIC)
    if include_pre_arrival:
        numeric += PRE_ARRIVAL_NUMERIC
    return numeric, list(SAFE_CATEGORICAL)


def assert_no_leakage(columns) -> None:
    """Fail loudly if any blocked column reached the feature matrix."""
    found = [c for c in columns if c in LEAKED]
    if found:
        detail = "; ".join(f"{c} ({LEAKED[c]})" for c in found)
        raise ValueError(
            f"Leaked column(s) in the feature set: {detail}. "
            f"These are known only at or after the outcome - a model trained "
            f"on them scores well and predicts nothing."
        )


def audit(staged_columns) -> dict[str, list[str]]:
    """Classify every staged column, so nothing is silently unaccounted for."""
    numeric, categorical = feature_columns(include_pre_arrival=True)
    used = set(numeric) | set(categorical)
    report = {
        "used_as_feature": sorted(c for c in staged_columns if c in used),
        "blocked_as_leakage": sorted(c for c in staged_columns if c in LEAKED),
        "excluded_by_design": sorted(c for c in staged_columns if c in EXCLUDED),
    }
    accounted = used | set(LEAKED) | set(EXCLUDED) | {TARGET}
    report["unclassified"] = sorted(c for c in staged_columns
                                    if c not in accounted)
    return report
