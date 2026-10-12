#!/usr/bin/env python
"""Score a booking's cancellation risk.

    python predict.py --lead-time 300 --country PRT --market-segment Groups
    python predict.py --json '{"lead_time": 5, "country": "GBR"}'
    python predict.py --batch bookings.csv --out scored.csv
    python predict.py --explain --lead-time 250 --deposit-type "Non Refund"
    python predict.py --fields

Any field you leave out falls back to the dataset's modal booking, which is
itself high-risk - run with --explain to see exactly which fields were
defaulted and what the model actually received.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hotel_pipeline.config import DEFAULT_CONFIG  # noqa: E402
from hotel_pipeline.model import (BOOKING_DEFAULTS, CancellationModel,  # noqa: E402
                                  prepare_booking)

# CLI flag -> booking field. Kept explicit so --help lists real options
# rather than a generic key=value bag.
FLAGS = {
    "--hotel": ("hotel", str, "City Hotel | Resort Hotel"),
    "--lead-time": ("lead_time", int, "days between booking and arrival"),
    "--arrival-month": ("arrival_month", int, "1-12"),
    "--arrival-dow": ("arrival_dow", str, "Monday ... Sunday"),
    "--week-number": ("arrival_date_week_number", int, "ISO week of arrival"),
    "--weekend-nights": ("weekend_nights", int, "Saturday/Sunday nights"),
    "--week-nights": ("week_nights", int, "Monday-Friday nights"),
    "--adults": ("adults", int, "number of adults"),
    "--children": ("children", int, "number of children"),
    "--babies": ("babies", int, "number of babies"),
    "--adr": ("adr", float, "average daily rate in EUR"),
    "--meal": ("meal", str, "Bed & Breakfast | Half Board | Full Board | Self Catering"),
    "--country": ("country", str, "ISO-3 code, e.g. PRT, GBR, FRA"),
    "--market-segment": ("market_segment", str,
                         "Online TA | Offline TA/TO | Groups | Direct | Corporate | Aviation | Complementary"),
    "--distribution-channel": ("distribution_channel", str,
                               "TA/TO | Direct | Corporate | GDS"),
    "--room-type": ("reserved_room_type", str, "reserved room type, A-P"),
    "--deposit-type": ("deposit_type", str,
                       "No Deposit | Non Refund | Refundable"),
    "--customer-type": ("customer_type", str,
                        "Transient | Transient-Party | Contract | Group"),
    "--special-requests": ("special_requests", int, "count of special requests"),
    "--repeat-guest": ("is_repeated_guest", int, "1 if a returning guest"),
    "--previous-cancellations": ("previous_cancellations", int,
                                 "this guest's prior cancellations"),
    "--previous-stays": ("previous_bookings_not_canceled", int,
                         "this guest's prior completed stays"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the chance a hotel booking is cancelled.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    for flag, (field, caster, help_text) in FLAGS.items():
        parser.add_argument(flag, dest=field, type=caster,
                            help=f"{help_text} (default: {BOOKING_DEFAULTS[field]})")
    parser.add_argument("--json", metavar="STR",
                        help="booking as a JSON object, instead of flags")
    parser.add_argument("--batch", metavar="CSV",
                        help="score a CSV of bookings, one row each")
    parser.add_argument("--out", metavar="CSV",
                        help="write --batch results here (default: stdout)")
    parser.add_argument("--explain", action="store_true",
                        help="show which fields were defaulted and the full "
                             "feature row the model received")
    parser.add_argument("--fields", action="store_true",
                        help="list every booking field with its default, then exit")
    return parser


def _print_fields() -> int:
    width = max(len(f) for f in BOOKING_DEFAULTS)
    flag_for = {field: flag for flag, (field, _, _) in FLAGS.items()}
    print(f"{'field'.ljust(width)}  {'default':<16} flag")
    print(f"{'-' * width}  {'-' * 16} {'-' * 24}")
    for field, default in BOOKING_DEFAULTS.items():
        print(f"{field.ljust(width)}  {str(default):<16} "
              f"{flag_for.get(field, '(--json only)')}")
    return 0


def _score_batch(model: CancellationModel, source: Path,
                 out: Path | None) -> int:
    import pandas as pd

    rows = pd.read_csv(source)
    prepared = pd.concat(
        [prepare_booking(**{k: v for k, v in row.items()
                            if k in BOOKING_DEFAULTS and pd.notna(v)})
         for row in rows.to_dict(orient="records")],
        ignore_index=True,
    )
    rows["cancel_probability"] = model.predict_proba(prepared).round(4)
    rows["risk_band"] = pd.cut(
        rows["cancel_probability"], bins=[-0.01, 0.20, 0.40, 0.60, 1.01],
        labels=["Low", "Moderate", "Elevated", "High"],
    ).astype(str)
    if out:
        rows.to_csv(out, index=False)
        print(f"Scored {len(rows):,} booking(s) -> {out}")
    else:
        print(rows.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fields:
        return _print_fields()

    try:
        model = CancellationModel.load(cfg=DEFAULT_CONFIG)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if args.batch:
        return _score_batch(model, Path(args.batch),
                            Path(args.out) if args.out else None)

    booking = {field: getattr(args, field) for _, (field, _, _) in
               FLAGS.items() if getattr(args, field, None) is not None}
    if args.json:
        try:
            booking.update(json.loads(args.json))
        except json.JSONDecodeError as exc:
            print(f"--json is not valid JSON: {exc}", file=sys.stderr)
            return 1

    try:
        result = model.predict_one(**booking)
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    bar_width = 44
    filled = round(result["probability"] * bar_width)
    print()
    print(f"  Cancellation risk   {result['percent']:.1f}%   "
          f"[{result['risk_band']}]")
    print(f"  {'#' * filled}{'.' * (bar_width - filled)}")
    print(f"  {result['lift_vs_base']}x the {result['base_rate'] * 100:.1f}% "
          f"base rate for this period")
    print()

    if args.explain:
        print(f"  Supplied ({len(result['fields_supplied'])}): "
              f"{', '.join(result['fields_supplied']) or 'nothing'}")
        print(f"  Defaulted ({len(result['fields_defaulted'])}): "
              f"{', '.join(result['fields_defaulted'])}")
        print()
        print("  Defaults are the dataset's modal booking, which is itself")
        print("  high-risk (Portuguese, Online TA, 90-day lead time).")
        print()
        print("  Feature row the model received:")
        for key, value in result["inputs_used"].items():
            print(f"    {key:<34} {value}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
