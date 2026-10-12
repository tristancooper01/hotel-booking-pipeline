"""Data quality gates.

A tiny expectations engine. Each check states what it expects, records what it
actually observed, and carries a severity:

    ERROR - the data is unusable; the run stops.
    WARN  - known, tolerated messiness; recorded and carried forward.
    INFO  - a measurement worth trending, never a failure.

Checks run against the raw frame (is the source still what we think it is?) and
again against the staged frame (did the transform actually fix things?).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .logging_setup import get_logger

log = get_logger(__name__)

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


@dataclass
class CheckResult:
    name: str
    level: str
    passed: bool
    observed: str
    expected: str

    @property
    def status(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.level == ERROR else self.level


@dataclass
class Suite:
    """A named collection of expectations evaluated against one DataFrame."""

    label: str
    results: list[CheckResult] = field(default_factory=list)

    # --- primitives ------------------------------------------------------
    def check(self, name: str, passed: bool, observed: str, expected: str,
              level: str = ERROR) -> None:
        self.results.append(
            CheckResult(name, level, bool(passed), str(observed), str(expected))
        )

    # --- expectations ----------------------------------------------------
    def expect_columns(self, df: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        self.check("schema: required columns present", not missing,
                   f"{len(missing)} missing: {missing[:5]}",
                   f"all {len(columns)} present")

    def expect_row_count_between(self, df: pd.DataFrame, low: int,
                                 high: int) -> None:
        self.check("volume: row count in range", low <= len(df) <= high,
                   f"{len(df):,} rows", f"between {low:,} and {high:,}")

    def expect_unique(self, df: pd.DataFrame, column: str) -> None:
        dupes = int(df[column].duplicated().sum())
        self.check(f"uniqueness: {column}", dupes == 0,
                   f"{dupes:,} duplicate values", "0 duplicates")

    def expect_no_nulls(self, df: pd.DataFrame, columns: Iterable[str],
                        level: str = ERROR) -> None:
        for col in columns:
            n = int(df[col].isna().sum())
            self.check(f"completeness: {col}", n == 0,
                       f"{n:,} nulls", "0 nulls", level=level)

    def expect_null_rate_below(self, df: pd.DataFrame, column: str,
                               max_rate: float, level: str = WARN) -> None:
        rate = float(df[column].isna().mean())
        self.check(f"completeness: {column} null rate", rate <= max_rate,
                   f"{rate:.2%}", f"<= {max_rate:.0%}", level=level)

    def expect_values_in_set(self, df: pd.DataFrame, column: str,
                             allowed: Iterable, level: str = ERROR) -> None:
        allowed = set(allowed)
        unexpected = sorted(set(df[column].dropna().unique()) - allowed)
        self.check(f"validity: {column} domain", not unexpected,
                   f"unexpected: {unexpected[:5]}" if unexpected else "all known",
                   f"one of {sorted(allowed)[:6]}...", level=level)

    def expect_between(self, df: pd.DataFrame, column: str, low: float,
                       high: float, level: str = ERROR) -> None:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        outside = int(((series < low) | (series > high)).sum())
        rng = f"[{series.min():g}, {series.max():g}]" if len(series) else "empty"
        self.check(f"range: {column}", outside == 0,
                   f"{outside:,} outside, actual {rng}",
                   f"within [{low:g}, {high:g}]", level=level)

    def expect_outlier_rate_below(self, df: pd.DataFrame, column: str,
                                  low: float, high: float, max_rate: float,
                                  level: str = WARN) -> None:
        """Tolerate a known handful of outliers, but notice a new infestation.

        A plain range check on a column with two known-bad rows would fail on
        every single run, which trains everyone to ignore it. This one passes
        while the problem stays the size it has always been.
        """
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        outside = int(((series < low) | (series > high)).sum())
        rate = outside / max(len(series), 1)
        self.check(f"range: {column} outlier rate", rate <= max_rate,
                   f"{outside:,} outside [{low:g}, {high:g}] ({rate:.4%})",
                   f"<= {max_rate:.2%} of rows", level=level)

    def expect(self, name: str, df: pd.DataFrame,
               predicate: Callable[[pd.DataFrame], tuple[bool, str]],
               expected: str, level: str = ERROR) -> None:
        """Escape hatch for a rule that needs more than one column."""
        passed, observed = predicate(df)
        self.check(name, passed, observed, expected, level=level)

    # --- outcomes --------------------------------------------------------
    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.level == ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.level == WARN]

    def log_summary(self) -> None:
        passed = sum(r.passed for r in self.results)
        log.info("%s: %d/%d checks passed, %d warning(s), %d failure(s)",
                 self.label, passed, len(self.results),
                 len(self.warnings), len(self.failures))
        for r in self.warnings:
            log.warning("  WARN  %s -> %s (expected %s)",
                        r.name, r.observed, r.expected)
        for r in self.failures:
            log.error("  FAIL  %s -> %s (expected %s)",
                      r.name, r.observed, r.expected)

    def raise_if_failed(self) -> None:
        if self.failures:
            names = "\n".join(f"  - {r.name}: got {r.observed}, expected "
                              f"{r.expected}" for r in self.failures)
            raise ValueError(
                f"{self.label} failed {len(self.failures)} ERROR-level "
                f"check(s):\n{names}"
            )


def write_report(suites: Sequence[Suite], md_path: Path,
                 json_path: Path) -> None:
    """Persist all suites as a human-readable table and a machine-readable log."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report", "",
             f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]
    total = sum(len(s.results) for s in suites)
    fails = sum(len(s.failures) for s in suites)
    warns = sum(len(s.warnings) for s in suites)
    lines += [f"**{total - fails - warns}/{total} checks passed** "
              f"({warns} warning(s), {fails} failure(s))", ""]

    for suite in suites:
        lines += [f"## {suite.label}", "",
                  "| Status | Check | Observed | Expected |",
                  "|---|---|---|---|"]
        for r in suite.results:
            lines.append(
                f"| {r.status} | {r.name} | {r.observed} | {r.expected} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(
        {s.label: [asdict(r) for r in s.results] for s in suites}, indent=2
    ), encoding="utf-8")
    log.info("Data quality report -> %s", md_path.name)
