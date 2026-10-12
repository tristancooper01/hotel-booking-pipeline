#!/usr/bin/env python
"""Command-line entry point for the hotel booking pipeline.

    python run_pipeline.py                 # full run
    python run_pipeline.py --force-download
    python run_pipeline.py --strict        # warnings become failures
    python run_pipeline.py --stage report  # re-render from the existing warehouse
    python run_pipeline.py --with-model    # also train the cancellation model
    python run_pipeline.py --stage model   # train only, from the existing staging
    python run_pipeline.py --drop-duplicates
    python run_pipeline.py --query "SELECT * FROM v_monthly_performance LIMIT 5"
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hotel_pipeline.config import DEFAULT_CONFIG  # noqa: E402
from hotel_pipeline.logging_setup import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hotel booking analytics warehouse and report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--force-download", action="store_true",
                        help="re-fetch the source CSV even if it is cached")
    parser.add_argument("--strict", action="store_true",
                        help="treat data quality warnings as failures")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="exclude exact duplicate rows instead of flagging "
                             "them (see README: the duplicates question)")
    parser.add_argument("--with-model", action="store_true",
                        help="also train the cancellation model (adds several "
                             "minutes; see predict.py to score bookings)")
    parser.add_argument("--stage", choices=["all", "extract", "transform",
                                            "load", "analyze", "model",
                                            "report"],
                        default="all",
                        help="run one stage against existing artifacts "
                             "(default: all)")
    parser.add_argument("--query", metavar="SQL",
                        help="run one SQL query against the warehouse and exit")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG
    if args.drop_duplicates:
        cfg = dataclasses.replace(cfg, drop_exact_duplicates=True)

    # Ad-hoc query mode: no pipeline run, just read the warehouse.
    if args.query:
        import pandas as pd
        from hotel_pipeline.warehouse import query
        if not cfg.paths.database.exists():
            print(f"No warehouse at {cfg.paths.database}. Run the pipeline "
                  f"first.", file=sys.stderr)
            return 1
        pd.set_option("display.width", 200, "display.max_columns", 50)
        print(query(args.query, cfg).to_string(index=False))
        return 0

    try:
        if args.stage == "all":
            from hotel_pipeline.pipeline import run
            result = run(cfg, force_download=args.force_download,
                         strict=args.strict, with_model=args.with_model)
            print()
            print(f"Done in {result.duration_seconds:.1f}s "
                  f"({result.rows_staged:,} rows, {result.warnings} warning(s))")
            print(f"  Report        : {result.analysis_report}")
            print(f"  Data quality  : {result.quality_report}")
            print(f"  Warehouse     : {result.database}")
            return 0

        return _run_single_stage(args.stage, cfg, args.force_download)

    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1


def _run_single_stage(stage: str, cfg, force_download: bool) -> int:
    """Run one stage, reading whatever the previous stage left on disk."""
    import pandas as pd
    setup_logging(cfg.paths.logs)
    cfg.paths.ensure()

    from hotel_pipeline import (analyze as analyze_stage, extract as
                                extract_stage, report as report_stage,
                                transform as transform_stage,
                                warehouse as warehouse_stage)

    def require(path: Path, produced_by: str):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - run `--stage {produced_by}` first."
            )
        return path

    if stage == "extract":
        extract_stage.extract(cfg, force=force_download)
    elif stage == "transform":
        raw = pd.read_parquet(require(cfg.paths.raw_parquet, "extract"))
        transform_stage.transform(raw, cfg).to_parquet(
            cfg.paths.staged_parquet, index=False
        )
    elif stage == "load":
        staged = pd.read_parquet(require(cfg.paths.staged_parquet, "transform"))
        warehouse_stage.load(warehouse_stage.build_star(staged), cfg)
    elif stage == "analyze":
        require(cfg.paths.database, "load")
        analyze_stage.analyze(cfg)
    elif stage == "model":
        require(cfg.paths.staged_parquet, "transform")
        require(cfg.paths.database, "load")
        from hotel_pipeline import model as model_stage
        trained, _ = model_stage.train(cfg)
        model_stage.score_warehouse(trained, cfg)
    elif stage == "report":
        require(cfg.paths.database, "load")
        results, insights = analyze_stage.analyze(cfg)
        report_stage.report(results, insights, cfg)

    print(f"Stage '{stage}' complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
