"""Central configuration.

Every path and tunable the pipeline uses lives here so that a run can be
redirected (tests, a scratch directory, a different source file) by building a
new ``Config`` rather than by editing module internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The Hotel Booking Demand dataset (Antonio, Almeida & Nunes, 2019), mirrored by
# the TidyTuesday project. 119,390 rows x 32 columns of real bookings from a
# city hotel and a resort hotel in Portugal, July 2015 - August 2017.
SOURCE_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw: Path = field(init=False)
    staging: Path = field(init=False)
    warehouse: Path = field(init=False)
    reports: Path = field(init=False)
    figures: Path = field(init=False)
    logs: Path = field(init=False)
    sql: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", self.root / "data" / "raw")
        object.__setattr__(self, "staging", self.root / "data" / "staging")
        object.__setattr__(self, "warehouse", self.root / "data" / "warehouse")
        object.__setattr__(self, "reports", self.root / "reports")
        object.__setattr__(self, "figures", self.root / "reports" / "figures")
        object.__setattr__(self, "logs", self.root / "logs")
        object.__setattr__(self, "sql", self.root / "sql")

    def ensure(self) -> None:
        for p in (self.raw, self.staging, self.warehouse,
                  self.reports, self.figures, self.logs):
            p.mkdir(parents=True, exist_ok=True)

    # --- Concrete artifacts produced by each stage -----------------------
    @property
    def raw_csv(self) -> Path:
        return self.raw / "hotel_bookings.csv"

    @property
    def raw_parquet(self) -> Path:
        return self.raw / "hotel_bookings.parquet"

    @property
    def staged_parquet(self) -> Path:
        return self.staging / "bookings_clean.parquet"

    @property
    def database(self) -> Path:
        return self.warehouse / "hotels.db"

    @property
    def quality_report(self) -> Path:
        return self.reports / "data_quality.md"

    @property
    def quality_json(self) -> Path:
        return self.reports / "data_quality.json"

    @property
    def analysis_report(self) -> Path:
        return self.reports / "analysis_report.md"

    @property
    def metrics_json(self) -> Path:
        return self.reports / "metrics.json"

    @property
    def model(self) -> Path:
        return self.warehouse / "cancellation_model.joblib"

    @property
    def model_metrics(self) -> Path:
        return self.reports / "model_metrics.json"

    @property
    def web_model(self) -> Path:
        """The logistic model as JSON, for the browser scoring app."""
        return self.reports / "web_model.json"

    @property
    def risk_app(self) -> Path:
        return self.reports / "risk_app.html"


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    source_url: str = SOURCE_URL

    # --- Cleaning policy -------------------------------------------------
    # Bookings with zero occupants are data-entry artifacts, not stays.
    drop_zero_guest_bookings: bool = True
    # ~32k rows are field-for-field identical. Two guests can genuinely make the
    # same booking on the same day, so these are FLAGGED, not deleted; set this
    # True to exclude them and see how much each headline number moves.
    drop_exact_duplicates: bool = False
    # A handful of ADR values are absurd (one at 5400, a few negative). They are
    # winsorised into this band and the original is kept in `adr_raw`.
    adr_floor: float = 0.0
    adr_ceiling: float = 1000.0

    # --- Feature policy --------------------------------------------------
    lead_time_bins: tuple[int, ...] = (0, 7, 30, 90, 180, 365, 10_000)
    lead_time_labels: tuple[str, ...] = (
        "0-7 days", "8-30 days", "31-90 days",
        "91-180 days", "181-365 days", "365+ days",
    )

    # --- Reporting policy ------------------------------------------------
    top_n_countries: int = 10
    figure_dpi: int = 160
    random_seed: int = 42


DEFAULT_CONFIG = Config()
