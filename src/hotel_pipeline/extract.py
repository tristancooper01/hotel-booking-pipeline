"""Stage 1 - Extract.

Pulls the source CSV to ``data/raw/`` exactly once and snapshots it to Parquet.
The raw layer is immutable: nothing downstream ever writes back into it, so a
bad transform can always be replayed from the original bytes.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

# The 32 columns the rest of the pipeline is written against.
EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "assigned_room_type", "booking_changes",
    "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "reservation_status",
    "reservation_status_date",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(cfg: Config = DEFAULT_CONFIG, force: bool = False) -> Path:
    """Fetch the source CSV unless a local copy already exists."""
    target = cfg.paths.raw_csv
    if target.exists() and not force:
        log.info("Raw CSV already present (%s), skipping download.", target.name)
        return target

    cfg.paths.raw.mkdir(parents=True, exist_ok=True)
    log.info("Downloading source data from %s", cfg.source_url)
    tmp = target.with_suffix(".csv.part")
    try:
        request = urllib.request.Request(
            cfg.source_url, headers={"User-Agent": "hotel-pipeline/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, \
                tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the source data: {exc}.\n"
            f"Fix: download {cfg.source_url} by hand and save it as {target}."
        ) from exc

    tmp.replace(target)  # atomic: a half-written file never looks complete
    log.info("Downloaded %.1f MB -> %s", target.stat().st_size / 1e6, target)
    return target


def extract(cfg: Config = DEFAULT_CONFIG, force: bool = False) -> pd.DataFrame:
    """Load the raw CSV, verify its shape, and snapshot it to Parquet."""
    csv_path = download(cfg, force=force)

    # Read everything as-is; typing is the staging layer's job, and coercing
    # here would silently destroy the sentinel values we need to see.
    df = pd.read_csv(csv_path, dtype={"children": "float64"}, low_memory=False)
    log.info("Read %s rows x %s columns", f"{len(df):,}", df.shape[1])

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Source is missing {len(missing)} expected column(s): {missing}. "
            "The upstream schema changed - update EXPECTED_COLUMNS and the "
            "transform rules together."
        )
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        log.warning("Source has %d unexpected column(s), ignored: %s",
                    len(extra), extra)

    df = df[EXPECTED_COLUMNS]
    log.info("Source checksum (sha256): %s", _sha256(csv_path)[:16])

    df.to_parquet(cfg.paths.raw_parquet, index=False)
    log.info("Raw snapshot written -> %s", cfg.paths.raw_parquet.name)
    return df
