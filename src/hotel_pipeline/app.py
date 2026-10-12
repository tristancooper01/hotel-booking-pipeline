"""Builds the browser scoring app.

The page is a template plus the exported model, injected at build time. Keeping
them separate means the app can never ship a stale model: it is rebuilt from
`web_model.json` every time the model stage runs, and that file is itself
verified against scikit-learn before it is written.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config, DEFAULT_CONFIG
from .logging_setup import get_logger

log = get_logger(__name__)

TEMPLATE = Path(__file__).parent / "templates" / "risk_app.html"
PLACEHOLDER = "__MODEL_JSON__"


def build_app(cfg: Config = DEFAULT_CONFIG) -> Path:
    """Render the template with the exported model inlined."""
    if not cfg.paths.web_model.exists():
        raise FileNotFoundError(
            f"No exported model at {cfg.paths.web_model}. Run "
            f"`python run_pipeline.py --stage model` first."
        )
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"{TEMPLATE.name} has no {PLACEHOLDER} placeholder.")

    model_json = cfg.paths.web_model.read_text(encoding="utf-8")
    # The payload sits inside a <script> block, so the one sequence that could
    # break out of it has to be neutralised. JSON escapes survive JSON.parse.
    safe = model_json.replace("</", r"<\/")

    page = template.replace(PLACEHOLDER, safe)
    path = cfg.paths.risk_app
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    log.info("Scoring app -> %s (%.0f KB)", path.name, path.stat().st_size / 1024)
    return path
