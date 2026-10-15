from pathlib import Path

import pytest

from app.skills import extract_lexical, load_skills, technical_skills
from tools.evaluate_synthetic import run_evaluation


@pytest.fixture(scope="module")
def report():
    return run_evaluation()


def test_synthetic_profiles_rank_by_evidence(report):
    assert len(report["jds"]) == 12
    for jd_id, result in report["jds"].items():
        scores = result["after_scores"]
        assert len(scores) == 4
        assert len(set(scores)) == len(scores), jd_id
        assert scores[0] > scores[3], jd_id
    assert report["metrics"]["after_pairwise_accuracy"] >= 0.85
    assert report["metrics"]["after_ndcg_at_10"] >= 0.85
    assert report["jds"]["accountant"]["false_positives_removed"] == [
        "budgeting", "end",
    ]


def test_cook_teacher_and_hr_lexicons_are_domain_specific():
    lexicon = load_skills()
    cases = {
        "line_cook": {"food safety", "knife skills", "haccp"},
        "history_teacher": {
            "lesson planning", "classroom management", "curriculum design",
        },
        "hr_specialist": {
            "recruitment", "onboarding", "employee relations",
        },
    }
    for jd_id, expected in cases.items():
        jd = (
            Path(__file__).parent / "eval" / "jds" / f"{jd_id}.txt"
        ).read_text(encoding="utf-8")
        skills = extract_lexical(jd, lexicon)
        hard = technical_skills(skills, lexicon)
        assert expected.intersection(hard), jd_id
        assert not any(
            "software" in lexicon.skill_domains.get(skill, ())
            for skill in hard
        ), jd_id


def test_hr_cook_scenario_is_ordered_without_ties(report):
    candidates = report["cook_scenario"]
    scores = [row["score"] for row in candidates]
    assert len(candidates) == 10
    assert len(set(scores)) == 10
    assert report["metrics"]["cook_pairwise_accuracy"] >= 0.85
    assert report["metrics"]["cook_ndcg_at_10"] >= 0.85
