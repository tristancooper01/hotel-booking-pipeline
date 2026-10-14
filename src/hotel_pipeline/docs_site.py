"""Assembles the GitHub Pages site from the generated HTML.

The report and the scoring app are both self-contained single files, so
publishing them is mostly a copy. The one thing added here is a nav bar
linking the two, injected at build time rather than baked into the templates:
cross-links are a property of *the site*, not of either page, and the files in
``reports/`` stay standalone so they can be opened or shared on their own.

Run via ``python run_pipeline.py --stage docs``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

ANCHOR = '<div class="wrap">'

NAV_STYLE = """
<style>
.site-nav {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 20px;
  padding: 14px 0 0; margin: 0 0 -8px;
  font-family: var(--mono, ui-monospace, monospace); font-size: 12px;
}
.site-nav a { color: var(--ink-3, #888); text-decoration: none;
  border-bottom: 1px solid transparent; padding-bottom: 1px; }
.site-nav a:hover { color: var(--accent, #2a78d6); border-bottom-color: currentColor; }
.site-nav a[aria-current="page"] { color: var(--ink, #111); font-weight: 600; }
.site-nav .spacer { flex: 1; }
</style>
"""


def _nav(current: str) -> str:
    def link(href: str, label: str, key: str) -> str:
        mark = ' aria-current="page"' if key == current else ""
        return f'<a href="{href}"{mark}>{label}</a>'

    return (
        f'{NAV_STYLE}<nav class="site-nav" aria-label="Site">'
        f'{link("index.html", "Risk scorer", "app")}'
        f'{link("report.html", "Full analysis", "report")}'
        f'<span class="spacer"></span>'
        f'<a href="https://github.com/tristancooper01/hotel-booking-pipeline">'
        f'Source on GitHub</a>'
        f'</nav>'
    )


def _publish(source: Path, target: Path, current: str) -> int:
    html = source.read_text(encoding="utf-8")
    if ANCHOR in html:
        html = html.replace(ANCHOR, ANCHOR + _nav(current), 1)
    else:
        # Never silently publish a page without its nav - that would leave a
        # visitor on a dead end with no way back to the other page.
        log.warning("%s has no %s anchor; published without nav",
                    source.name, ANCHOR)
    target.write_text(html, encoding="utf-8")
    return target.stat().st_size


def build_docs(cfg: Config = DEFAULT_CONFIG) -> Path:
    """Copy the report and app into docs/ for GitHub Pages."""
    docs = cfg.paths.docs
    docs.mkdir(parents=True, exist_ok=True)

    # GitHub Pages runs Jekyll by default, which skips files and folders
    # beginning with an underscore and can rewrite what it serves. This opts
    # out and publishes the directory verbatim.
    (docs / ".nojekyll").write_text("", encoding="utf-8")

    published = []
    pairs = [
        (cfg.paths.risk_app, docs / "index.html", "app", "scoring app"),
        (cfg.paths.reports / "analysis_report.html", docs / "report.html",
         "report", "analysis report"),
    ]
    for source, target, current, label in pairs:
        if not source.exists():
            log.warning("Skipping %s: %s not built yet", label, source.name)
            continue
        size = _publish(source, target, current)
        published.append(target)
        log.info("  %-18s -> docs/%-12s %6.0f KB", label, target.name,
                 size / 1024)

    if not published:
        raise FileNotFoundError(
            "Nothing to publish. Run `--stage model` (builds the app) and "
            "`--stage report` first."
        )

    # The figures are already in the repo for the README; copy them so the
    # site is self-contained if a page ever references them relatively.
    figures = docs / "figures"
    if cfg.paths.figures.exists():
        figures.mkdir(exist_ok=True)
        for png in cfg.paths.figures.glob("*.png"):
            shutil.copy2(png, figures / png.name)

    log.info("Pages site -> %s (%d page(s))", docs.name, len(published))
    return docs
