from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app import main
from app.config import FINETUNED_DIR, get_settings
from app.embed import load_model
from app.scoring import load_calibrator
from app.skills import load_skills

EVAL_DIR = Path(__file__).resolve().parent
SYNTHETIC_JD = (
    "Process Automation Engineer: design, program, and commission "
    "industrial automation using PLC, SCADA, DCS, HMI, Modbus, PID "
    "control, and MATLAB. Experience in instrumentation, process control, "
    "troubleshooting, safety systems, and manufacturing is required."
)


def ndcg_at_10(scores: list[float], relevance: list[int]) -> float:
    order = np.argsort(np.asarray(scores))[::-1]
    discounts = np.log2(np.arange(2, min(10, len(order)) + 2))
    ranked_gain = 2 ** np.asarray(relevance)[order[:10]] - 1
    ideal_gain = 2 ** np.sort(np.asarray(relevance))[::-1][:10] - 1
    dcg = float(np.sum(ranked_gain / discounts[:len(ranked_gain)]))
    idcg = float(np.sum(ideal_gain / discounts[:len(ideal_gain)]))
    return dcg / idcg if idcg else 0.0


@pytest.fixture(scope="module")
def scored_resumes() -> list[dict[str, Any]]:
    resumes = json.loads(
        (EVAL_DIR / "synthetic_resumes.json").read_text(encoding="utf-8")
    )
    settings = get_settings()
    previous_state = {
        key: getattr(main.app.state, key, None)
        for key in ("settings", "model", "model_name", "calibrator", "lexicon")
    }
    main.app.state.settings = settings
    main.app.state.model_name = settings.model_path()
    main.app.state.model = load_model(settings)
    main.app.state.calibrator = (
        load_calibrator(settings)
        if main.app.state.model_name == str(FINETUNED_DIR) else None
    )
    main.app.state.lexicon = load_skills()

    async def score_all() -> list[dict[str, Any]]:
        results = []
        for resume in resumes:
            result = await main._score_text(
                f"{resume['role']}.pdf",
                resume["text"],
                [],
                resume["name"],
                SYNTHETIC_JD,
                settings,
            )
            results.append({
                **resume,
                "score": result.match_percentage,
                "result": result,
            })
        return results

    try:
        return asyncio.run(score_all())
    finally:
        for key, value in previous_state.items():
            setattr(main.app.state, key, value)


def test_relevance_ranking_and_ndcg(scored_resumes: list[dict[str, Any]]) -> None:
    scores = [resume["score"] for resume in scored_resumes]
    relevance = [resume["relevance"] for resume in scored_resumes]
    metric = ndcg_at_10(scores, relevance)
    assert metric >= 0.95

    relevant_scores = [
        resume["score"] for resume in scored_resumes
        if resume["relevance"] > 0
    ]
    irrelevant_scores = [
        resume["score"] for resume in scored_resumes
        if resume["relevance"] == 0
    ]
    assert min(relevant_scores) > max(irrelevant_scores)


def test_candidate_name_does_not_change_score() -> None:
    resume = json.loads(
        (EVAL_DIR / "synthetic_resumes.json").read_text(encoding="utf-8")
    )[0]
    settings = get_settings()
    state = main.app.state
    previous = {
        key: getattr(state, key, None)
        for key in ("settings", "model", "model_name", "calibrator", "lexicon")
    }
    state.settings = settings
    state.model_name = settings.model_path()
    state.model = load_model(settings)
    state.calibrator = (
        load_calibrator(settings)
        if state.model_name == str(FINETUNED_DIR) else None
    )
    state.lexicon = load_skills()

    async def score(text: str, name: str) -> int:
        result = await main._score_text(
            "synthetic.pdf",
            text,
            [],
            name,
            SYNTHETIC_JD,
            settings,
        )
        return result.match_percentage

    renamed_text = resume["text"].replace(
        resume["name"], "Morgan Reed", 1
    )

    async def score_pair() -> tuple[int, int]:
        original = await score(resume["text"], resume["name"])
        renamed = await score(renamed_text, "Morgan Reed")
        return original, renamed

    try:
        scores = asyncio.run(score_pair())
    finally:
        for key, value in previous.items():
            setattr(state, key, value)
    assert abs(scores[0] - scores[1]) <= 1
