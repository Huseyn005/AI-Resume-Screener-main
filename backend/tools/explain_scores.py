from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app import main  # noqa: E402
from app.config import FINETUNED_DIR, get_settings  # noqa: E402
from app.embed import load_model, model_name  # noqa: E402
from app.extract import extract_pdf  # noqa: E402
from app.scoring import load_calibrator  # noqa: E402
from app.skills import load_skills  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain resume scores for a job description."
    )
    parser.add_argument("--jd-file", required=True, type=Path)
    parser.add_argument("--pdfs", required=True, help="PDF path glob")
    parser.add_argument(
        "--no-calibrator",
        action="store_true",
        help="Use the linear semantic fallback instead of the calibrator.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    jd = args.jd_file.read_text(encoding="utf-8").strip()
    paths = sorted(glob.glob(args.pdfs, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No PDFs matched glob: {args.pdfs}")

    settings = get_settings()
    model = load_model(settings)
    selected_model = model_name(settings)
    calibrator = None
    if (
        not args.no_calibrator
        and selected_model == str(FINETUNED_DIR)
    ):
        calibrator = load_calibrator(settings)

    state = main.app.state
    marker = object()
    previous: dict[str, Any] = {}
    for name, value in (
        ("model", model),
        ("lexicon", load_skills()),
        ("calibrator", calibrator),
    ):
        previous[name] = getattr(state, name, marker)
        setattr(state, name, value)

    try:
        for filename in paths:
            path = Path(filename)
            extracted = extract_pdf(
                path.read_bytes(),
                path.name,
                settings.min_text_chars,
                settings.hidden_font_size,
            )
            if extracted.status == "ok":
                _result, detail = await main._score_text(
                    path.name,
                    extracted.text,
                    extracted.flags,
                    extracted.candidate_name,
                    jd,
                    settings,
                    include_debug=True,
                )
            else:
                stub = main._stub(
                    path.name,
                    extracted.flags,
                    extracted.status,
                    extracted.candidate_name,
                )
                _result, detail = main._with_explanation(
                    stub, True, settings, jd, extracted.text,
                    "PDF could not be scored.",
                )
            print(json.dumps(detail, indent=2, ensure_ascii=False))
    finally:
        for name, value in previous.items():
            if value is marker:
                delattr(state, name)
            else:
                setattr(state, name, value)


def main_cli() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main_cli()
