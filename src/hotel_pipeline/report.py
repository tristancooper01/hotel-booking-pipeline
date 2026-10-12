"""Stage 5 - Report.

Turns the analysis tables into charts and two renderings of the same report:

    reports/analysis_report.html   the designed, self-contained page
    reports/analysis_report.md     the plain-text version, for diffs and terminals

Every figure is paired with its underlying table. That is not padding: it is the
table view that makes the chart readable for anyone the colour encoding fails -
screen readers, print, colour-vision deficiency - and it lets a reader check a
number instead of estimating it off an axis.

Charts are rendered twice, once per theme, and inlined as SVG data URIs so the
HTML file is a single self-contained artifact with no asset directory to carry
around.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import viz
from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


# --- Charts ---------------------------------------------------------------

def chart_market_origin(df: pd.DataFrame, path) -> None:
    """Realised vs lost per origin -> part-to-whole, two categorical slots."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    sub = df[df["origin"] != "Unknown origin"].sort_values("realized_revenue")
    y = np.arange(len(sub))
    ax.barh(y, sub["realized_revenue"], color=viz.cat(0), height=0.55,
            label="Realised revenue", edgecolor=viz.surface(), linewidth=2)
    ax.barh(y, sub["lost_revenue"], left=sub["realized_revenue"],
            color=viz.cat(1), height=0.55, label="Lost to cancellation",
            edgecolor=viz.surface(), linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["origin"])
    for i, row in enumerate(sub.itertuples(index=False)):
        ax.annotate(f"{row.cancellation_rate_pct:.0f}% cancelled",
                    (row.realized_revenue + row.lost_revenue, i),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=viz.ink_secondary())
    viz.finish(ax, "Domestic demand loses more than it earns",
               "Booked value by guest origin: collected versus cancelled",
               xlabel="Booking value", grid_axis="x")
    viz.money_axis(ax, axis="x")
    ax.set_xlim(0, (sub["realized_revenue"] + sub["lost_revenue"]).max() * 1.3)
    ax.legend(loc="lower right")
    viz.save(fig, path)


def chart_cancellation_by_lead_time(df: pd.DataFrame, path) -> None:
    """Magnitude on an ordered scale -> sequential ramp, not categorical hues."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = viz.sequential_steps(len(df))
    bars = ax.bar(df["lead_time_bucket"], df["cancellation_rate_pct"],
                  color=colors, width=0.68)
    viz.label_bars(ax, bars, df["cancellation_rate_pct"])
    viz.finish(ax, "Cancellation rate climbs with booking lead time",
               "Share of bookings cancelled, by how far ahead they were made",
               ylabel="Cancellation rate")
    viz.pct_axis(ax)
    ax.set_ylim(0, max(df["cancellation_rate_pct"]) * 1.18)
    viz.save(fig, path)


def chart_monthly_demand(df: pd.DataFrame, path) -> None:
    """Part-to-whole over time -> stacked bar, two categorical slots."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(df))
    # 2px surface-coloured edge keeps the two segments from fusing into one mark
    ax.bar(x, df["stays"], color=viz.cat(0), width=0.72,
           label="Stayed", edgecolor=viz.surface(), linewidth=2)
    ax.bar(x, df["cancellations"], bottom=df["stays"],
           color=viz.cat(1), width=0.72, label="Cancelled",
           edgecolor=viz.surface(), linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["arrival_year_month"], rotation=60, ha="right",
                       fontsize=8)
    viz.finish(ax, "Monthly demand, split by what actually happened",
               "Bookings by arrival month; the orange band never became revenue",
               ylabel="Bookings")
    viz.count_axis(ax)
    ax.legend(loc="upper left", ncols=2)
    viz.save(fig, path)


def chart_adr_seasonality(df: pd.DataFrame, path) -> None:
    """Two series that ARE the subject -> categorical, plus direct labels."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, hotel in enumerate(["City Hotel", "Resort Hotel"]):
        sub = df[df["hotel_name"] == hotel].sort_values("month_num")
        ax.plot(sub["month_num"], sub["avg_daily_rate"], color=viz.cat(i),
                marker="o", label=hotel, zorder=3 - i)
        last = sub.iloc[-1]
        ax.annotate(hotel, (last["month_num"], last["avg_daily_rate"]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.ink_secondary())
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([m[:3] for m in MONTH_ORDER])
    ax.set_xlim(0.6, 13.8)
    viz.finish(ax, "The resort is a seasonal business; the city hotel is not",
               "Average daily rate by arrival month, all years pooled",
               ylabel="Average daily rate")
    viz.money_axis(ax)
    ax.legend(loc="upper left")
    viz.save(fig, path)


def chart_cancellation_by_segment(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sub = df.sort_values("cancellation_rate_pct")
    colors = viz.sequential_steps(len(sub))
    bars = ax.barh(sub["market_segment"], sub["cancellation_rate_pct"],
                   color=colors, height=0.66)
    viz.label_bars(ax, bars, sub["cancellation_rate_pct"], horizontal=True)
    viz.finish(ax, "Cancellation risk by acquisition channel",
               "Groups and online travel agents behave very differently",
               xlabel="Cancellation rate", grid_axis="x")
    viz.pct_axis(ax, axis="x")
    ax.set_xlim(0, max(sub["cancellation_rate_pct"]) * 1.15)
    viz.save(fig, path)


def chart_revenue_at_risk(df: pd.DataFrame, path) -> None:
    """Realised vs lost, per channel -> part-to-whole, two categorical slots."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sub = df.sort_values("realized_revenue")
    y = np.arange(len(sub))
    ax.barh(y, sub["realized_revenue"], color=viz.cat(0), height=0.66,
            label="Realised revenue", edgecolor=viz.surface(), linewidth=2)
    ax.barh(y, sub["lost_revenue"], left=sub["realized_revenue"],
            color=viz.cat(1), height=0.66, label="Lost to cancellation",
            edgecolor=viz.surface(), linewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["market_segment"])
    viz.finish(ax, "Where the money lands, and where it evaporates",
               "Booking value by channel: collected versus cancelled",
               xlabel="Booking value", grid_axis="x")
    viz.money_axis(ax, axis="x")
    ax.legend(loc="lower right")
    viz.save(fig, path)


def chart_top_countries(df: pd.DataFrame, path, top_n: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sub = df.head(top_n).sort_values("bookings")
    colors = viz.sequential_steps(len(sub))
    bars = ax.barh(sub["country_code"], sub["bookings"], color=colors,
                   height=0.66)
    viz.label_bars(ax, bars, sub["bookings"], fmt="{:,.0f}", horizontal=True)
    viz.finish(ax, f"Top {top_n} source markets by volume",
               "Portugal alone is more than a third of all bookings",
               xlabel="Bookings", grid_axis="x")
    viz.count_axis(ax, axis="x")
    ax.set_xlim(0, max(sub["bookings"]) * 1.14)
    viz.save(fig, path)


def chart_length_of_stay(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.3))
    sub = df.sort_values("sort_key").head(12)
    colors = viz.sequential_steps(len(sub))[::-1]  # dark = common, not long
    bars = ax.bar(sub["stay_length"], sub["bookings"], color=colors, width=0.68)
    viz.label_bars(ax, bars, sub["bookings"], fmt="{:,.0f}")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([s.replace(" nights", "").replace(" night", "")
                        for s in sub["stay_length"]])
    viz.finish(ax, "Most stays are short: one to four nights",
               "Bookings by length of stay (nights booked)",
               xlabel="Nights", ylabel="Bookings")
    viz.count_axis(ax)
    ax.set_ylim(0, max(sub["bookings"]) * 1.15)
    viz.save(fig, path)


def chart_special_requests(df: pd.DataFrame, path) -> None:
    """One bar carries the point -> emphasis, not a full categorical set."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    colors = viz.emphasis_colors(df["special_requests"], "0")
    bars = ax.bar(df["special_requests"], df["cancellation_rate_pct"],
                  color=colors, width=0.62)
    viz.label_bars(ax, bars, df["cancellation_rate_pct"])
    viz.finish(ax, "Guests who ask for something rarely cancel",
               "Cancellation rate by number of special requests on the booking",
               xlabel="Special requests", ylabel="Cancellation rate")
    viz.pct_axis(ax)
    ax.set_ylim(0, max(df["cancellation_rate_pct"]) * 1.18)
    viz.save(fig, path)


def chart_deposit_type(df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sub = df.sort_values("cancellation_rate_pct")
    colors = viz.emphasis_colors(sub["deposit_type"], "Non Refund")
    bars = ax.bar(sub["deposit_type"], sub["cancellation_rate_pct"],
                  color=colors, width=0.56)
    viz.label_bars(ax, bars, sub["cancellation_rate_pct"])
    viz.finish(ax, "Non-refundable bookings cancel the most, not the least",
               "Cancellation rate by deposit policy - the opposite of the "
               "intuition",
               xlabel="Deposit policy", ylabel="Cancellation rate")
    viz.pct_axis(ax)
    ax.set_ylim(0, max(sub["cancellation_rate_pct"]) * 1.18)
    viz.save(fig, path)


def chart_demand_heatmap(df: pd.DataFrame, path) -> None:
    """A grid of magnitudes -> heatmap on one sequential hue."""
    pivot = (df.pivot_table(index="arrival_day_name", columns="arrival_month",
                            values="bookings", aggfunc="sum")
             .reindex(index=DAY_ORDER, columns=MONTH_ORDER))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    mesh = ax.imshow(pivot.values, cmap=viz.sequential_cmap(), aspect="auto")
    ax.set_xticks(range(12), [m[:3] for m in MONTH_ORDER])
    ax.set_yticks(range(7), [d[:3] for d in DAY_ORDER])

    # Label every cell: at 84 cells the grid doubles as the table view.
    peak = np.nanmax(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value / 1000:.1f}k", ha="center", va="center",
                    fontsize=7.5, color=viz.on_fill(value, peak))

    # Title states what this grid actually shows, read off the grid itself, so
    # it cannot drift out of step with the data on a later run.
    by_day = pivot.sum(axis=1)
    by_month = pivot.sum(axis=0)
    viz.finish(ax,
               f"{by_day.idxmax()} is the busiest arrival day, "
               f"{by_day.idxmin()} the quietest",
               f"Bookings by arrival weekday and month, all years pooled; "
               f"volume peaks in {by_month.idxmax()}")
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.015)
    cbar.set_label("Bookings", color=viz.ink_secondary(), fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, color=viz.grid())
    viz.save(fig, path)


def chart_calibration(metrics: dict, path) -> None:
    """Predicted probability against observed frequency, versus the diagonal.

    The only chart here with a *reference* line rather than a scale: the
    diagonal is perfect calibration, and the gap to it is the error that
    matters when probabilities are summed into expected arrivals.
    """
    curve = metrics["calibration_curve"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], color=viz.ink_secondary(), linewidth=1.2,
            linestyle=(0, (4, 4)), label="Perfect calibration", zorder=1)
    ax.plot(curve["predicted"], curve["observed"], color=viz.cat(0),
            marker="o", label="This model", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    viz.finish(ax, "The model under-predicts across most of the range",
               f"Observed cancellation rate against predicted probability, "
               f"held-out 2017 arrivals (n={metrics['n']:,})",
               xlabel="Predicted probability", ylabel="Observed rate")
    ax.grid(axis="x")
    viz.fraction_axis(ax, axis="x")   # these are probabilities, not 0-100
    viz.fraction_axis(ax, axis="y")
    ax.legend(loc="upper left")
    viz.save(fig, path)


def chart_risk_deciles(metrics: dict, path) -> None:
    deciles = pd.DataFrame(metrics["deciles"])
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    colors = viz.sequential_steps(len(deciles))
    bars = ax.bar(deciles["decile"].astype(str),
                  deciles["actual_rate"] * 100, color=colors, width=0.68)
    viz.label_bars(ax, bars, deciles["actual_rate"] * 100)
    base = metrics["base_rate"] * 100
    ax.axhline(base, color=viz.ink_secondary(), linewidth=1.2,
               linestyle=(0, (4, 4)), zorder=1)
    ax.annotate(f"overall {base:.1f}%", (len(deciles) - 0.4, base),
                xytext=(0, 6), textcoords="offset points", ha="right",
                fontsize=8.5, color=viz.ink_secondary())
    viz.finish(ax, "Risk ranking separates the book cleanly",
               "Actual cancellation rate by predicted-risk decile, "
               "held-out 2017 arrivals",
               xlabel="Risk decile (10 = highest predicted risk)",
               ylabel="Actual cancellation rate")
    viz.pct_axis(ax)
    ax.set_ylim(0, 105)
    viz.save(fig, path)


def chart_feature_importance(importance: list[dict], path,
                             top_n: int = 14) -> None:
    if not importance:
        raise ValueError("no permutation importance recorded - retrain the "
                         "model to populate it")
    frame = pd.DataFrame(importance).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = viz.sequential_steps(len(frame))
    ax.barh(frame["feature"], frame["importance"], color=colors, height=0.68,
            xerr=frame["std"],
            error_kw=dict(ecolor=viz.ink_muted(), elinewidth=1, capsize=2))
    # Read the headline off the ranking so it cannot contradict the bars.
    top_two = [r["feature"].replace("_", " ") for r in importance[:2]]
    viz.finish(ax, f"{top_two[0].capitalize()} and {top_two[1]} carry the model",
               "Permutation importance: Brier degradation when the column is "
               "shuffled (bars show +/- 1 sd over 3 repeats)",
               xlabel="Increase in Brier score when shuffled", grid_axis="x")
    viz.save(fig, path)


MODEL_CHARTS = [
    ("calibration", chart_calibration, "model_calibration"),
    ("deciles", chart_risk_deciles, "model_deciles"),
    ("importance", chart_feature_importance, "model_importance"),
]


CHARTS = [
    ("market_origin", chart_market_origin, "market_origin"),
    ("cancellation_by_lead_time", chart_cancellation_by_lead_time,
     "cancellation_by_lead_time"),
    ("cancellation_by_deposit", chart_deposit_type, "deposit_type"),
    ("cancellation_by_segment", chart_cancellation_by_segment,
     "cancellation_by_segment"),
    ("cancellation_by_segment", chart_revenue_at_risk, "revenue_at_risk"),
    ("monthly_trend", chart_monthly_demand, "monthly_demand"),
    ("adr_by_month_hotel", chart_adr_seasonality, "adr_seasonality"),
    ("weekday_demand", chart_demand_heatmap, "demand_heatmap"),
    ("top_countries", chart_top_countries, "top_countries"),
    ("length_of_stay", chart_length_of_stay, "length_of_stay"),
    ("special_requests_vs_cancellation", chart_special_requests,
     "special_requests"),
]


def build_figures(results: dict[str, pd.DataFrame],
                  cfg: Config = DEFAULT_CONFIG) -> dict[str, Path]:
    """Write PNG figures for the Markdown report."""
    viz.apply_style(viz.LIGHT)
    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    for source, fn, name in CHARTS:
        if source not in results:
            log.warning("Skipping %s: analysis %r missing", name, source)
            continue
        path = cfg.paths.figures / f"{name}.png"
        try:
            fn(results[source], path)
        except Exception as exc:  # a broken chart must not kill the report
            log.error("Chart %s failed: %s", name, exc)
            continue
        made[f"{name}.png"] = path
    log.info("  %d PNG figure(s) written", len(made))
    return made


def load_model_metrics(cfg: Config = DEFAULT_CONFIG) -> dict | None:
    """Read the trained model's evaluation, if the model stage has been run."""
    path = cfg.paths.model_metrics
    if not path.exists():
        return None
    import json
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        log.warning("Could not read %s: %s", path.name, exc)
        return None
    selected = blob.get("models", {}).get(blob.get("selected", ""))
    if not selected:
        return None
    return {"selected": selected, "all": blob["models"],
            "importance": blob.get("models", {}).get("importance", []),
            "meta": blob.get("metadata", {})}


def _model_chart_inputs(model: dict) -> dict[str, object]:
    return {"calibration": model["selected"], "deciles": model["selected"],
            "importance": model["importance"]}


def build_inline_figures(results: dict[str, pd.DataFrame],
                         model: dict | None = None
                         ) -> dict[str, dict[str, str]]:
    """Render every chart once per theme as an inline SVG data URI."""
    inline: dict[str, dict[str, str]] = {}
    previous = plt.rcParams["savefig.format"]
    plt.rcParams["savefig.format"] = "svg"  # file-like targets read this
    model_inputs = _model_chart_inputs(model) if model else {}
    jobs = [(results.get(src), fn, name) for src, fn, name in CHARTS]
    jobs += [(model_inputs.get(src), fn, name) for src, fn, name
             in MODEL_CHARTS if model]
    try:
        for theme in (viz.LIGHT, viz.DARK):
            viz.apply_style(theme)
            for data, fn, name in jobs:
                empty = (data is None
                         or (hasattr(data, "empty") and data.empty)
                         or (isinstance(data, (list, dict)) and not data))
                if empty:
                    continue
                buf = io.BytesIO()
                try:
                    fn(data, buf)
                except Exception as exc:
                    log.error("Chart %s (%s) failed: %s", name, theme.name, exc)
                    continue
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
                inline.setdefault(name, {})[theme.name] = (
                    f"data:image/svg+xml;base64,{encoded}"
                )
    finally:
        plt.rcParams["savefig.format"] = previous
        viz.apply_style(viz.LIGHT)
    total_kb = sum(len(v) for d in inline.values() for v in d.values()) / 1024
    log.info("  %d inline chart(s), both themes, %.0f KB", len(inline), total_kb)
    return inline


# --- Shared content -------------------------------------------------------

def _money(value: float) -> str:
    return f"EUR {value / 1e6:.2f}M" if abs(value) >= 1e6 else f"EUR {value:,.0f}"


def _eur(value: float) -> str:
    """Typographic euro, for the HTML rendering only."""
    return f"€{value / 1e6:.2f}M" if abs(value) >= 1e6 else f"€{value:,.0f}"


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "")
        elif pd.api.types.is_integer_dtype(sub[col]):
            sub[col] = sub[col].map(lambda v: f"{v:,}")
    return sub


def model_sections(model: dict | None) -> list[dict]:
    """The three sections describing the trained model, or none if untrained."""
    if not model:
        return []
    s = model["selected"]
    meta = model.get("meta", {})
    everything = model.get("all", {})
    baseline = everything.get("baseline", {})
    importance = model.get("importance") or []
    top = ", ".join(f["feature"] for f in importance[:3])
    ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth",
                "seventh", "eighth", "ninth", "tenth"]
    names = [f["feature"] for f in importance]
    lead_rank = (ORDINALS[names.index("lead_time")]
                 if "lead_time" in names[:10] else "well down the list")
    deciles = pd.DataFrame(s["deciles"]) if s.get("deciles") else pd.DataFrame()
    top_decile = deciles.iloc[-1] if len(deciles) else None
    bottom_decile = deciles.iloc[0] if len(deciles) else None

    gap = ""
    if top_decile is not None and bottom_decile is not None:
        gap = (f" In the held-out period the highest-risk tenth of bookings "
               f"cancelled {top_decile['actual_rate'] * 100:.0f}% of the time "
               f"and the lowest-risk tenth "
               f"{bottom_decile['actual_rate'] * 100:.1f}% - the ranking "
               f"separates the book cleanly enough to act on.")

    sections = [
        dict(id="model-ranking", title="Predicting cancellation: risk ranking",
             prose=(f"A model trained on arrivals before "
                    f"{meta.get('test_from', '2017-01-01')} and tested on "
                    f"everything after, so it is only ever scored on bookings "
                    f"it could not have seen. It reaches "
                    f"{s['roc_auc']:.3f} ROC-AUC and {s['pr_auc']:.3f} PR-AUC "
                    f"against a {baseline.get('pr_auc', 0):.3f} no-skill "
                    f"baseline.{gap}"),
             chart="model_deciles", table=None,
             alt="Actual cancellation rate by predicted risk decile"),
        dict(id="model-calibration", title="How much to trust the probability",
             prose=(f"Ranking is the easy half. To use these numbers for "
                    f"overbooking you have to sum them into expected arrivals, "
                    f"which requires the probability to be literally true. It "
                    f"is not, quite: the model averages "
                    f"{s['mean_predicted'] * 100:.1f}% against an actual "
                    f"{s['base_rate'] * 100:.1f}%, under-predicting by about "
                    f"{s['calibration_mae'] * 100:.1f} points across the "
                    f"range. That gap is distribution shift - cancellation "
                    f"rose between the training period and the test period - "
                    f"and it is the reason the model reports a Brier score of "
                    f"{s['brier']:.4f} rather than only an AUC. Correct for it "
                    f"before trusting an absolute forecast; the ranking is "
                    f"unaffected."),
             chart="model_calibration", table=None,
             alt="Calibration curve: predicted probability versus observed rate"),
    ]
    if not importance:
        return sections
    return sections + [
        dict(id="model-features", title="What the model leans on",
             prose=(f"Permutation importance on held-out data: how much the "
                    f"Brier score degrades when a single column is shuffled. "
                    f"{top.replace('_', ' ')} dominate - and note that lead "
                    f"time, the strongest signal in the univariate analysis "
                    f"above, ranks only {lead_rank} here. That is not a "
                    f"contradiction: much of what lead time tells you on its "
                    f"own is already carried by channel and origin once those "
                    f"are in the model. Deposit type leading the ranking does "
                    f"deserve suspicion, given the 99% cancellation rate "
                    f"flagged earlier may be a recording artifact - but "
                    f"retraining without it costs only 0.008 ROC-AUC and "
                    f"0.004 Brier, so no conclusion here rests on it. Note "
                    f"too what is absent: room reassignment correlates far "
                    f"more strongly with cancellation than anything shown, "
                    f"and is excluded by design, because rooms are assigned "
                    f"at check-in and the feature really means the guest "
                    f"arrived."),
             chart="model_importance", table=None,
             alt="Permutation feature importance"),
    ]


def build_sections(results: dict[str, pd.DataFrame], insights: dict
                   ) -> list[dict]:
    """The report's narrative, ordered by how much each finding is worth.

    Each entry carries its own prose, chart and table, so the HTML and Markdown
    renderings are two views of one structure rather than two documents.
    """
    i = insights
    deposits = i["deposit_rates"]

    origin = results.get("market_origin")
    if origin is not None and len(origin):
        dom = origin[origin["origin"] == "Portugal (domestic)"]
        intl = origin[origin["origin"] == "International"]
        if len(dom) and len(intl):
            d, n = dom.iloc[0], intl.iloc[0]
            origin_prose = (
                f"The headline rate is two populations averaged into a number "
                f"that describes neither. Domestic Portuguese bookings cancel "
                f"at {d['cancellation_rate_pct']:.0f}% and lose "
                f"{_eur(d['lost_revenue'])} against {_eur(d['realized_revenue'])} "
                f"earned - {d['lost_per_eur_earned']:.2f} euros lost for every "
                f"euro collected. International bookings cancel at "
                f"{n['cancellation_rate_pct']:.0f}% and are "
                f"{n['lost_per_eur_earned']:.2f}. Almost every other pattern in "
                f"this report is partly this one in disguise."
            )
        else:
            origin_prose = "Domestic and international demand behave differently."
    else:
        origin_prose = "Domestic and international demand behave differently."

    monthly = results.get("monthly_trend")
    if monthly is not None and len(monthly):
        monthly_prose = (
            f"Monthly arrivals split into stays and cancellations. The cancelled "
            f"band is present in every single month of the series - it ranges "
            f"from {monthly['cancellation_rate_pct'].min():.0f}% to "
            f"{monthly['cancellation_rate_pct'].max():.0f}% of arrivals and "
            f"never approaches zero. A loss that reliable is a loss you can "
            f"forecast, and therefore partly recover through disciplined "
            f"overbooking."
        )
    else:
        monthly_prose = "Monthly arrivals split into stays and cancellations."

    segment = results.get("cancellation_by_segment")
    channel_prose = (
        f"Volume and risk are not correlated. {i['riskiest_segment']} "
        f"contributes {i['riskiest_segment_share']:.0f}% of bookings at a "
        f"{i['riskiest_segment_rate']:.1f}% cancellation rate; "
        f"{i['safest_segment']} converts far more reliably at "
        f"{i['safest_segment_rate']:.1f}%. Channel-level targets beat a "
        f"blended one."
    )
    if segment is not None and len(segment):
        per_booking = (segment["realized_revenue"] / segment["bookings"])
        best = segment.loc[per_booking.idxmax()]
        worst = segment.loc[per_booking.idxmin()]
        material = segment[segment["bookings"] >= 1000]
        if len(material):
            pb = material["realized_revenue"] / material["bookings"]
            best, worst = material.loc[pb.idxmax()], material.loc[pb.idxmin()]
            channel_prose += (
                f" In money rather than counts: a {best['market_segment']} "
                f"booking realises "
                f"{_eur(best['realized_revenue'] / best['bookings'])} on "
                f"average against "
                f"{_eur(worst['realized_revenue'] / worst['bookings'])} for "
                f"{worst['market_segment']} - and commission costs, which "
                f"would widen that gap further, are not in this dataset."
            )

    return [
        dict(id="origin", title="Domestic versus international demand",
             prose=origin_prose, chart="market_origin", table="market_origin",
             alt="Booked value by guest origin, realised versus lost"),
        dict(id="lead-time", title="Cancellations and lead time",
             prose=(f"The single strongest predictor in the dataset, and it "
                    f"rises monotonically with the booking window - from "
                    f"{i['lead_time_low_rate']:.1f}% in the last week before "
                    f"arrival to {i['lead_time_high_rate']:.1f}% more than a "
                    f"year out. The money, though, sits in the middle: the one-"
                    f"to-twelve-month window carries the bulk of the loss, "
                    f"while the extreme bucket is a high rate on very little "
                    f"volume. Any overbooking model that ignores lead time is "
                    f"leaving the best available feature on the table."),
             chart="cancellation_by_lead_time",
             table="cancellation_by_lead_time",
             alt="Cancellation rate by lead time bucket"),
        dict(id="deposit", title="Deposit policy",
             prose=(f"The counter-intuitive result, and the one to treat with "
                    f"most caution. Non-refundable bookings cancel at "
                    f"{deposits.get('Non Refund', 0):.1f}% against "
                    f"{deposits.get('No Deposit', 0):.1f}% for no deposit at "
                    f"all. A rate that close to total is not plausible as "
                    f"ordinary guest behaviour - investigate it as a data "
                    f"question before drawing any policy conclusion from it."),
             chart="deposit_type", table="cancellation_by_deposit",
             alt="Cancellation rate by deposit policy"),
        dict(id="channels", title="Acquisition channels",
             prose=channel_prose, chart="cancellation_by_segment",
             table="cancellation_by_segment",
             alt="Cancellation rate by market segment"),
        dict(id="revenue", title="Where revenue is won and lost",
             prose=("The same channel split in money rather than counts. This "
                    "is the view that decides where commission and marketing "
                    "spend should go: a channel with high volume and high "
                    "cancellation can contribute less net revenue than a "
                    "smaller, more reliable one."),
             chart="revenue_at_risk", table=None,
             alt="Realised versus lost revenue by market segment"),
        dict(id="over-time", title="Demand over time",
             prose=monthly_prose, chart="monthly_demand", table="monthly_trend",
             alt="Monthly bookings split by outcome"),
        dict(id="seasonality", title="Rate seasonality",
             prose=(f"The two properties are different businesses. The resort "
                    f"swings sharply - peaking in "
                    f"{i.get('peak_adr_month', 'August')} and bottoming in "
                    f"{i.get('trough_adr_month', 'January')} - while the city "
                    f"hotel holds a far flatter curve driven by business "
                    f"demand. Averaging them together describes neither, and "
                    f"the city hotel's flat rate against variable demand may "
                    f"mean it is underpricing its peak."),
             chart="adr_seasonality", table="performance_by_hotel",
             alt="Average daily rate by month and property"),
        dict(id="arrivals", title="Arrival patterns",
             prose=("Arrival weekday by month. The operational value is in the "
                    "corners: staffing and check-in capacity should follow "
                    "this grid, not a flat weekly average."),
             chart="demand_heatmap", table=None,
             alt="Bookings by arrival weekday and month"),
        dict(id="markets", title="Source markets",
             prose=(f"{i['top_country']} alone supplies "
                    f"{i['top_country_share']:.0f}% of bookings and cancels at "
                    f"{i['top_country_rate']:.1f}%, well above the "
                    f"{i['cancellation_rate_pct']:.1f}% average. The largest "
                    f"market is also the least reliable one - a concentration "
                    f"risk worth naming explicitly."),
             chart="top_countries", table="top_countries",
             alt="Top source markets by booking volume"),
        dict(id="stay-length", title="Length of stay",
             prose=("Short stays dominate: the distribution is heavily "
                    "weighted toward one to four nights, with a thin tail. "
                    "Minimum-stay rules and length-of-stay pricing should be "
                    "calibrated against this shape rather than against the "
                    "mean."),
             chart="length_of_stay", table="length_of_stay",
             alt="Bookings by length of stay"),
        dict(id="engagement", title="Guest engagement",
             prose=(f"Special requests act as a commitment signal: "
                    f"{i['requests_zero_rate']:.1f}% cancellation with none, "
                    f"falling to {i['requests_many_rate']:.1f}% with three or "
                    f"more. Note the shape - almost the entire effect lands "
                    f"between zero requests and one, so read it as a binary "
                    f"engagement flag rather than a dose response. It is free, "
                    f"available at booking time, and trivially usable as a "
                    f"model feature."),
             chart="special_requests",
             table="special_requests_vs_cancellation",
             alt="Cancellation rate by number of special requests"),
    ]


CAVEATS = [
    ("Cancellation rate counts bookings, not room nights.",
     "A cancelled two-week stay counts the same as a cancelled one-night stay. "
     "The revenue figures are the ones that weight by length."),
    ("Lost revenue is not a cash loss.",
     "It is booked value that was cancelled (rate x nights) - demand that "
     "existed and did not convert. Some of those rooms were resold, and this "
     "dataset cannot say which. Treat it as the size of the opportunity."),
    ("Parking requests are leakage too, and that one is easy to miss.",
     "Not one of the 7,409 bookings that requested a car parking space was "
     "ever cancelled - zero, not nearly zero. Perfect separation like that is "
     "structure, not behaviour: the field is evidently filled in when a space "
     "is allocated on arrival. It is excluded from the model for the same "
     "reason as room reassignment, and it is worth assuming other columns in "
     "any booking export hide the same problem."),
    ("Room reassignment is leakage, not a predictor.",
     "Bookings where the room changed cancel at around 5% versus 45%, which "
     "looks like the strongest signal here and is worthless: rooms are "
     "assigned at check-in, so it really means the guest arrived. In a model "
     "it produces excellent accuracy that predicts nothing."),
    ("Everything here is correlational.",
     "Nothing in this data establishes that adding a deposit causes "
     "cancellation, or that soliciting a request prevents one. Those need an "
     "experiment."),
    ("About 27% of source rows are exact duplicates.",
     "They are retained by default, since identical bookings on the same day "
     "are plausible in a real system. Re-run with --drop-duplicates to see "
     "every figure here under the opposite assumption."),
    ("No cost, commission or capacity data.",
     "Channel profitability is therefore directional only, and occupancy and "
     "RevPAR - the industry's primary metric - cannot be computed at all."),
]


# --- HTML -----------------------------------------------------------------

def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    sub = _format_table(df.head(max_rows))
    head = "".join(f"<th scope=\"col\">{_esc(c).replace('_', ' ')}</th>"
                   for c in sub.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>"
        for row in sub.itertuples(index=False)
    )
    note = (f"<p class=\"table-note\">Showing {max_rows} of {len(df)} rows; "
            f"the full table is in <code>reports/tables/</code>.</p>"
            if len(df) > max_rows else "")
    return (f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>{note}")


def _figure(name: str, alt: str, inline: dict[str, dict[str, str]]) -> str:
    sources = inline.get(name)
    if not sources:
        return f"<p class=\"missing\">Figure <code>{_esc(name)}</code> unavailable.</p>"
    light = sources.get("light", "")
    dark = sources.get("dark", light)
    return (
        f"<figure class=\"chart\">"
        f"<img class=\"only-light\" src=\"{light}\" alt=\"{_esc(alt)}\">"
        f"<img class=\"only-dark\" src=\"{dark}\" alt=\"\" aria-hidden=\"true\">"
        f"</figure>"
    )


STYLE = """
:root {
  color-scheme: light;
  --ground: #fbfaf8;
  --surface: #ffffff;
  --surface-sunk: #f4f2ee;
  --ink: #16181d;
  --ink-2: #4e5360;
  --ink-3: #82868f;
  --rule: #e4e1da;
  --rule-strong: #d2cec4;
  --accent: #2a78d6;
  --accent-soft: #e8f0fc;
  --loss: #c4521f;
  --loss-soft: #fbeade;
  --shadow: 0 1px 2px rgba(20, 22, 28, .05), 0 8px 24px -12px rgba(20, 22, 28, .14);
  --display: "Newsreader", Georgia, "Times New Roman", serif;
  --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #141518;
    --surface: #1c1e22;
    --surface-sunk: #222429;
    --ink: #f2f1ee;
    --ink-2: #b4b3ac;
    --ink-3: #85847e;
    --rule: #2e3036;
    --rule-strong: #3d4047;
    --accent: #3987e5;
    --accent-soft: #16243a;
    --loss: #e6855c;
    --loss-soft: #2e2019;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -12px rgba(0, 0, 0, .6);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #141518;
  --surface: #1c1e22;
  --surface-sunk: #222429;
  --ink: #f2f1ee;
  --ink-2: #b4b3ac;
  --ink-3: #85847e;
  --rule: #2e3036;
  --rule-strong: #3d4047;
  --accent: #3987e5;
  --accent-soft: #16243a;
  --loss: #e6855c;
  --loss-soft: #2e2019;
  --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px -12px rgba(0, 0, 0, .6);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--body);
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
.only-dark { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) .only-light { display: none; }
  :root:not([data-theme="light"]) .only-dark { display: block; }
}
:root[data-theme="dark"] .only-light { display: none; }
:root[data-theme="dark"] .only-dark { display: block; }

.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 24px; padding-block: 0 72px; }

/* --- Masthead --- */
.masthead { padding-block: 56px 36px; border-bottom: 1px solid var(--rule-strong); }
.eyebrow {
  font-family: var(--mono); font-size: 11.5px; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 18px;
}
h1 {
  font-family: var(--display); font-weight: 500; font-size: clamp(34px, 5.6vw, 54px);
  line-height: 1.08; letter-spacing: -.02em; margin: 0 0 20px; text-wrap: balance;
}
.standfirst {
  font-size: clamp(17px, 2vw, 19px); color: var(--ink-2); max-width: 62ch;
  margin: 0; line-height: 1.6;
}
.provenance {
  font-family: var(--mono); font-size: 12px; color: var(--ink-3);
  margin: 24px 0 0; display: flex; flex-wrap: wrap; gap: 8px 20px;
}

/* --- KPI tiles --- */
.kpis {
  list-style: none; margin: 32px 0 0; padding: 0;
  display: grid; gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); border-radius: 10px; overflow: hidden;
  grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
}
.kpi { background: var(--surface); padding: 20px 22px 18px; }
.kpi dt {
  font-family: var(--mono); font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--ink-3); margin: 0 0 10px;
}
.kpi dd {
  margin: 0; font-family: var(--display); font-weight: 500;
  font-size: 30px; line-height: 1; letter-spacing: -.015em;
  font-variant-numeric: tabular-nums;
}
.kpi .note { display: block; font-family: var(--body); font-size: 12.5px;
  color: var(--ink-3); margin-top: 8px; letter-spacing: 0; }
.kpi.is-loss dd { color: var(--loss); }

/* --- Summary --- */
.summary { margin-block: 56px 0; }
.summary h2 { margin-top: 0; }
.verdict {
  font-family: var(--display); font-size: clamp(20px, 2.6vw, 25px);
  line-height: 1.42; font-weight: 500; margin: 0 0 32px; max-width: 34ch;
  border-left: 3px solid var(--accent); padding-left: 22px;
}
.verdict b { font-weight: 600; color: var(--accent); }
.findings { list-style: none; counter-reset: f; margin: 0; padding: 0;
  display: grid; gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; }
.findings li {
  counter-increment: f; background: var(--surface);
  padding: 22px 24px 22px 62px; position: relative; max-width: none;
}
.findings li::before {
  content: counter(f); position: absolute; left: 24px; top: 23px;
  font-family: var(--mono); font-size: 12px; font-weight: 600;
  color: var(--accent); background: var(--accent-soft);
  width: 22px; height: 22px; border-radius: 5px;
  display: grid; place-items: center;
}
.findings strong { display: block; font-size: 16.5px; margin-bottom: 4px; }
.findings p { margin: 0; color: var(--ink-2); font-size: 15px; }

/* --- Layout with rail --- */
.layout { display: grid; gap: 48px; margin-top: 64px; }
@media (min-width: 1000px) {
  .layout { grid-template-columns: 186px minmax(0, 1fr); gap: 56px; align-items: start; }
  .rail { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px); }
}
.rail nav { border-left: 1px solid var(--rule); }
.rail ol { list-style: none; margin: 0; padding: 0;
  counter-reset: r; font-size: 13.5px; }
.rail a {
  display: block; padding: 7px 0 7px 16px; margin-left: -1px;
  border-left: 2px solid transparent; color: var(--ink-3);
  text-decoration: none; line-height: 1.35;
}
.rail a:hover { color: var(--ink); border-left-color: var(--rule-strong); }
.rail-title {
  font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
  text-transform: uppercase; color: var(--ink-3);
  padding: 0 0 12px 16px; margin: 0;
}

/* --- Sections --- */
section { scroll-margin-top: 24px; }
section + section { margin-top: 64px; }
h2 {
  font-family: var(--display); font-weight: 500;
  font-size: clamp(24px, 3.2vw, 31px); line-height: 1.2;
  letter-spacing: -.015em; margin: 0 0 16px; text-wrap: balance;
}
section > p { max-width: 68ch; color: var(--ink-2); margin: 0 0 26px; }

.chart {
  margin: 0; padding: 18px; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 10px; box-shadow: var(--shadow);
}
.chart img { display: block; width: 100%; height: auto; }

details { margin-top: 14px; }
summary {
  cursor: pointer; font-family: var(--mono); font-size: 12px;
  letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3);
  padding: 8px 0; list-style: none; display: inline-flex;
  align-items: center; gap: 8px;
}
summary::-webkit-details-marker { display: none; }
summary::before { content: "+"; font-size: 14px; line-height: 1; }
details[open] summary::before { content: "\\2212"; }
summary:hover { color: var(--accent); }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 3px; }

.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 9px 14px; text-align: right; white-space: nowrap;
  border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
thead th {
  position: sticky; top: 0; background: var(--surface-sunk);
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  letter-spacing: .04em; text-transform: uppercase; color: var(--ink-3);
}
tbody tr:last-child td { border-bottom: 0; }
tbody td { font-family: var(--mono); font-variant-numeric: tabular-nums;
  color: var(--ink-2); background: var(--surface); }
tbody td:first-child { color: var(--ink); }
.table-note { font-size: 12.5px; color: var(--ink-3); margin: 10px 0 0; }
.missing { color: var(--ink-3); font-style: italic; }

/* --- Caveats --- */
.caveats { margin-top: 72px; padding-top: 40px; border-top: 1px solid var(--rule-strong); }
.caveats ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 20px; }
.caveats li { max-width: 72ch; padding-left: 18px; border-left: 2px solid var(--loss); }
.caveats strong { display: block; margin-bottom: 3px; }
.caveats p { margin: 0; color: var(--ink-2); font-size: 15px; }

footer {
  margin-top: 56px; padding-top: 24px; border-top: 1px solid var(--rule);
  font-family: var(--mono); font-size: 12px; color: var(--ink-3);
  display: flex; flex-wrap: wrap; gap: 6px 24px;
}
code { font-family: var(--mono); font-size: .92em;
  background: var(--surface-sunk); padding: 1px 5px; border-radius: 4px; }
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


def build_html(results: dict[str, pd.DataFrame], insights: dict,
               inline: dict[str, dict[str, str]],
               model: dict | None = None,
               cfg: Config = DEFAULT_CONFIG) -> Path:
    """Assemble the designed, self-contained HTML report."""
    i = insights
    sections = build_sections(results, insights) + model_sections(model)
    hotels = i["hotel_rates"]

    kpis = [
        ("Bookings", f"{i['total_bookings']:,}",
         f"{i['period_start']} to {i['period_end']}", False),
        ("Cancellation rate", f"{i['cancellation_rate_pct']:.1f}%",
         "of all bookings made", True),
        ("Realised revenue", _eur(i["realized_revenue"]),
         "booked value collected", False),
        ("Lost to cancellation", _eur(i["lost_revenue"]),
         f"{i['revenue_at_risk_pct']:.0f}% of booked demand", True),
        ("Average daily rate", f"€{i['avg_daily_rate']:.2f}",
         "across both properties", False),
        ("Average lead time", f"{i['avg_lead_time_days']:.0f} days",
         "between booking and arrival", False),
    ]
    kpi_html = "".join(
        f"<div class=\"kpi{' is-loss' if loss else ''}\">"
        f"<dt>{_esc(label)}</dt>"
        f"<dd>{_esc(value)}<span class=\"note\">{_esc(note)}</span></dd>"
        f"</div>"
        for label, value, note, loss in kpis
    )

    findings = [
        ("Lead time is the dominant signal.",
         f"Bookings made {i['lead_time_low']} ahead cancel at "
         f"{i['lead_time_low_rate']:.1f}%; those made {i['lead_time_high']} "
         f"ahead cancel at {i['lead_time_high_rate']:.1f}%. Distant bookings "
         f"are options, not commitments."),
        ("Channel quality varies enormously.",
         f"{i['riskiest_segment']} bookings ({i['riskiest_segment_share']:.0f}% "
         f"of volume) cancel at {i['riskiest_segment_rate']:.1f}%, while "
         f"{i['safest_segment']} cancels at {i['safest_segment_rate']:.1f}%. "
         f"Blended channel reporting hides this completely."),
        ("Non-refundable deposits mark risk, they do not prevent it.",
         f"Bookings with a non-refundable deposit cancel at "
         f"{i['deposit_rates'].get('Non Refund', 0):.1f}% against "
         f"{i['deposit_rates'].get('No Deposit', 0):.1f}% with no deposit. "
         f"Treat this as a data question before a policy one."),
        ("Engagement predicts arrival.",
         f"Bookings with no special request cancel at "
         f"{i['requests_zero_rate']:.1f}%; those with three or more at "
         f"{i['requests_many_rate']:.1f}%. A guest who asks for a cot has "
         f"already decided to come."),
    ]
    findings_html = "".join(
        f"<li><strong>{_esc(title)}</strong><p>{_esc(body)}</p></li>"
        for title, body in findings
    )

    rail_html = "".join(
        f"<li><a href=\"#{s['id']}\">{_esc(s['title'])}</a></li>"
        for s in sections
    )

    body_parts = []
    for s in sections:
        table_html = ""
        if s["table"] and s["table"] in results:
            table_html = (
                f"<details><summary>Show the data</summary>"
                f"{_html_table(results[s['table']])}</details>"
            )
        body_parts.append(
            f"<section id=\"{s['id']}\">"
            f"<h2>{_esc(s['title'])}</h2>"
            f"<p>{_esc(s['prose'])}</p>"
            f"{_figure(s['chart'], s['alt'], inline)}"
            f"{table_html}"
            f"</section>"
        )

    caveats_html = "".join(
        f"<li><strong>{_esc(t)}</strong><p>{_esc(b)}</p></li>"
        for t, b in CAVEATS
    )

    doc = f"""<title>Hotel Cancellation Economics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap">
<style>{STYLE}</style>
<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">Revenue analysis &middot; {_esc(i['period_start'][:4])}&ndash;{_esc(i['period_end'][:4])}</p>
    <h1>What {i['cancellation_rate_pct']:.0f}% cancellation costs two Portuguese hotels</h1>
    <p class="standfirst">{i['total_bookings']:,} bookings across a city hotel and a
      resort hotel, {_esc(i['period_start'])} to {_esc(i['period_end'])}, drawn from
      {i['countries']} source markets. Cancellation is not a side effect here &mdash;
      it is the central fact of the business.</p>
    <p class="provenance">
      <span>City Hotel {hotels.get('City Hotel', 0):.1f}% cancelled</span>
      <span>Resort Hotel {hotels.get('Resort Hotel', 0):.1f}% cancelled</span>
      <span>Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}</span>
    </p>
  </header>

  <dl class="kpis">{kpi_html}</dl>

  <div class="summary">
    <p class="verdict">Of every euro of booked value,
      <b>{i['revenue_at_risk_pct']:.0f} cents never converts</b> &mdash;
      {_eur(i['lost_revenue'])} against {_eur(i['realized_revenue'])} collected.</p>
    <ol class="findings">{findings_html}</ol>
  </div>

  <div class="layout">
    <aside class="rail">
      <nav aria-label="Sections">
        <p class="rail-title">Contents</p>
        <ol>{rail_html}</ol>
      </nav>
    </aside>
    <main>
      {''.join(body_parts)}
      <div class="caveats">
        <h2>How to read these numbers</h2>
        <ul>{caveats_html}</ul>
      </div>
      <footer>
        <span>Hotel Booking Demand &mdash; Antonio, Almeida &amp; Nunes (2019)</span>
        <span>Generated by the hotel booking pipeline</span>
      </footer>
    </main>
  </div>
</div>
"""
    path = cfg.paths.reports / "analysis_report.html"
    path.write_text(doc, encoding="utf-8")
    log.info("HTML report -> %s (%.0f KB)", path.name,
             path.stat().st_size / 1024)
    return path


# --- Markdown -------------------------------------------------------------

def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    sub = _format_table(df.head(max_rows))
    header = "| " + " | ".join(str(c) for c in sub.columns) + " |"
    rule = "|" + "|".join("---" for _ in sub.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in sub.itertuples(index=False)]
    note = (f"\n_Showing {max_rows} of {len(df)} rows; full table in "
            f"`reports/tables/`._\n" if len(df) > max_rows else "")
    return "\n".join([header, rule, *rows]) + "\n" + note


def build_markdown(results: dict[str, pd.DataFrame], insights: dict,
                   figures: dict[str, Path], model: dict | None = None,
                   cfg: Config = DEFAULT_CONFIG) -> Path:
    """The plain-text rendering of the same report structure."""
    i = insights
    hotels = i["hotel_rates"]
    lines: list[str] = []
    add = lines.append

    add("# Hotel Booking Analysis")
    add("")
    add(f"Arrivals from **{i['period_start']}** to **{i['period_end']}** - "
        f"{i['total_bookings']:,} bookings across two Portuguese properties "
        f"and {i['countries']} source markets.")
    add("")
    add(f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by the hotel booking "
        f"pipeline. A designed version of this report is at "
        f"`analysis_report.html`._")
    add("")
    add("## Executive summary")
    add("")
    add(f"**{i['cancellation_rate_pct']:.1f}% of all bookings are cancelled** - "
        f"{_money(i['lost_revenue'])} of booked value that never converts, "
        f"against {_money(i['realized_revenue'])} realised. "
        f"{i['revenue_at_risk_pct']:.0f}% of booked demand evaporates before "
        f"check-in.")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add(f"| Total bookings | {i['total_bookings']:,} |")
    add(f"| Cancellation rate | {i['cancellation_rate_pct']:.2f}% |")
    add(f"| Realised revenue | {_money(i['realized_revenue'])} |")
    add(f"| Revenue lost to cancellation | {_money(i['lost_revenue'])} |")
    add(f"| Booked demand at risk | {i['revenue_at_risk_pct']:.1f}% |")
    add(f"| Average daily rate | EUR {i['avg_daily_rate']:.2f} |")
    add(f"| Average lead time | {i['avg_lead_time_days']:.0f} days |")
    for name, rate in hotels.items():
        add(f"| {name} cancellation rate | {rate:.2f}% |")
    add("")

    for s in build_sections(results, insights) + model_sections(model):
        add(f"## {s['title']}")
        add("")
        add(s["prose"].replace("€", "EUR "))
        add("")
        figure_name = f"{s['chart']}.png"
        add(f"![{s['alt']}](figures/{figure_name})" if figure_name in figures
            else f"_Figure `{figure_name}` unavailable._")
        add("")
        if s["table"] and s["table"] in results:
            add(_md_table(results[s["table"]]))
            add("")

    add("## Supporting detail")
    add("")
    for key, title in [("performance_by_hotel", "Property comparison"),
                       ("repeat_guest_behaviour", "Repeat versus first-time guests"),
                       ("party_type_profile", "Party composition"),
                       ("room_change_impact", "Room reassignment")]:
        if key in results:
            add(f"### {title}")
            add("")
            add(_md_table(results[key]))
            add("")

    add("## How to read these numbers")
    add("")
    for title, body in CAVEATS:
        add(f"- **{title}** {body}")
    add("")
    add("See `data_quality.md` for the full check results behind this report.")
    add("")

    path = cfg.paths.analysis_report
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown report -> %s", path.name)
    return path


def report(results: dict[str, pd.DataFrame], insights: dict,
           cfg: Config = DEFAULT_CONFIG) -> Path:
    """Full stage 5: figures, then both renderings of the report."""
    log.info("Building report")
    model = load_model_metrics(cfg)
    if model:
        log.info("  including model evaluation (%s)",
                 model["selected"]["label"])
    figures = build_figures(results, cfg)
    inline = build_inline_figures(results, model)
    build_markdown(results, insights, figures, model, cfg)
    return build_html(results, insights, inline, model, cfg)
