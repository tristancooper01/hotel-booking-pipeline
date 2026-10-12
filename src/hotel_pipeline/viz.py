"""Chart styling: one palette, one set of rules, applied everywhere.

Colour is assigned by the *job* it does, not by series order:

    sequential  - magnitude on an ordered scale (one hue, low to high)
    categorical - telling distinct series apart (fixed slot order, never cycled)
    emphasis    - one series is the point, the rest are context (hue + grey)

The categorical order is validated for colour-vision deficiency on adjacent
pairs; it is a fixed sequence, so slot 1 is always blue and a chart with three
series uses slots 1-3 rather than any three hues.

Both themes are *selected*, not inverted. The dark steps are the same eight hues
re-stepped for a dark surface, and the sequential ramp flips direction so that
"more" always means "further from the background" in either theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless: the pipeline writes files, never a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    deemphasis: str
    categorical: tuple[str, ...]
    # Ordered low -> high. On light that runs pale to deep; on dark it runs
    # deep to pale, so the high end always stands off the background.
    sequential: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#16181d",
    ink_secondary="#5a5f6b",
    ink_muted="#8d919b",
    grid="#e7e5e0",
    deemphasis="#cbc9c3",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    # Starts at step 250 - the lightest step that still clears 2:1 on white.
    sequential=("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
)

DARK = Theme(
    name="dark",
    surface="#1c1e22",
    ink="#f2f1ee",
    ink_secondary="#a8a79f",
    ink_muted="#7c7b75",
    grid="#33353b",
    deemphasis="#4a4c52",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"),
    # Stops at step 600 - the darkest step that still clears 2:1 on the dark
    # surface - and runs outward to the palest.
    sequential=("#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
                "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#cde2fb"),
)

_current: Theme = LIGHT


# --- Accessors: chart code reads the active theme, never a literal ---------

def theme() -> Theme:
    return _current


def cat(index: int) -> str:
    """Categorical slot by position. Fixed order, never cycled."""
    return _current.categorical[index]


def surface() -> str:
    return _current.surface


def ink() -> str:
    return _current.ink


def ink_secondary() -> str:
    return _current.ink_secondary


def ink_muted() -> str:
    return _current.ink_muted


def grid() -> str:
    return _current.grid


def accent() -> str:
    return _current.categorical[0]


def sequential_steps(n: int) -> list[str]:
    """`n` evenly spaced steps from the active ramp, lowest value first."""
    ramp = _current.sequential
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    span = len(ramp) - 1
    return [ramp[round(i * span / (n - 1))] for i in range(n)]


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"seq_{_current.name}", list(_current.sequential)
    )


def emphasis_colors(labels: Sequence[str], highlight: str) -> list[str]:
    """Accent for the one series that carries the point, grey for context."""
    return [accent() if str(label) == highlight else _current.deemphasis
            for label in labels]


def on_fill(value: float, peak: float) -> str:
    """Readable ink for a label sitting on a filled sequential cell.

    The ramps run in opposite directions per theme - high values are the
    *darkest* fill on light and the *palest* on dark - so the contrasting ink
    flips with them.
    """
    high = value > peak * 0.55
    if _current.name == "dark":
        return "#16181d" if high else _current.ink
    return "#ffffff" if high else _current.ink


# --- Style ----------------------------------------------------------------

def apply_style(active: Theme = LIGHT) -> None:
    """Set the active theme and matplotlib defaults to match it."""
    global _current
    _current = active
    plt.rcParams.update({
        "figure.facecolor": active.surface,
        "axes.facecolor": active.surface,
        "savefig.facecolor": active.surface,
        "savefig.edgecolor": active.surface,
        "axes.edgecolor": active.grid,
        "axes.labelcolor": active.ink_secondary,
        "axes.titlecolor": active.ink,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        # Enough headroom for the title plus a subtitle line beneath it.
        "axes.titlepad": 30,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind the marks, never over
        "grid.color": active.grid,
        "grid.linewidth": 0.8,
        "xtick.color": active.ink_secondary,
        "ytick.color": active.ink_secondary,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": active.ink,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": active.ink_secondary,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.constrained_layout.use": True,
    })


def finish(ax, title: str, subtitle: str | None = None,
           xlabel: str = "", ylabel: str = "",
           grid_axis: str = "y") -> None:
    """Apply the shared chart anatomy: title, spines, grid direction."""
    ax.set_title(title)
    if subtitle:
        # Sits in the titlepad gap, between the title and the plot area.
        ax.text(0, 1.0, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=_current.ink_secondary, va="bottom", linespacing=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_current.grid)
    ax.grid(axis=grid_axis)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


def pct_axis(ax, axis: str = "y") -> None:
    """For values already expressed as 0-100."""
    fmt = FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fraction_axis(ax, axis: str = "y") -> None:
    """For values expressed as 0-1, such as probabilities."""
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def money_axis(ax, axis: str = "y") -> None:
    def fmt(v, _):
        if abs(v) >= 1e6:
            return f"EUR {v / 1e6:.1f}M"
        if abs(v) >= 1e3:
            return f"EUR {v / 1e3:.0f}k"
        return f"EUR {v:.0f}"
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(
        FuncFormatter(fmt)
    )


def count_axis(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def label_bars(ax, bars, values, fmt: str = "{:.1f}%",
               horizontal: bool = False, color: str | None = None) -> None:
    """Direct-label every bar. Labels sit outside the mark, in text ink."""
    color = color or _current.ink_secondary
    for bar, value in zip(bars, values):
        if horizontal:
            ax.annotate(fmt.format(value),
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=color)
        else:
            ax.annotate(fmt.format(value),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        va="bottom", ha="center", fontsize=8.5, color=color)


def save(fig, path, dpi: int = 160) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
