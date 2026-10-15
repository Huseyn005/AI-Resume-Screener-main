from __future__ import annotations
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from app.config import Settings


@dataclass
class ScoreParts:
    semantic: float
    calibrated_semantic: float
    skill_coverage: float
    section_fit: float
    final: float
    semantic_display: int
    skills_display: int
    section_display: int
    match_percentage: int
    semantic_weight: float = 0.0
    skill_weight: float = 0.0
    section_weight: float = 0.0
    limited_skill_signal: bool = False
    hard_skill_penalty: float = 1.0


def topk_mean(resume_emb: np.ndarray, jd_emb: np.ndarray,
              k: int = 3) -> float:
    if resume_emb.size == 0 or jd_emb.size == 0:
        return 0.0
    sim = resume_emb @ jd_emb.T
    per_chunk = sim.max(axis=1)
    k = min(k, per_chunk.shape[0])
    top = np.sort(per_chunk)[-k:]
    return float(np.clip(top.mean(), 0.0, 1.0))


def calibrate(raw: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((raw - low) / (high - low), 0.0, 1.0))


def load_calibrator(settings: Settings) -> IsotonicRegression | None:
    path = settings.calibrator_path()
    if not path.is_file():
        return None
    return joblib.load(path)


def semantic_probability(
        raw: float,
        settings: Settings,
        calibrator: IsotonicRegression | None,
) -> float:
    if calibrator is None:
        return calibrate(raw, settings.semantic_low, settings.semantic_high)
    predicted = calibrator.predict(np.asarray([[raw]], dtype=np.float64))
    return float(np.clip(predicted[0], 0.0, 1.0))


def skill_coverage(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(np.clip(matched / total, 0.0, 1.0))


def evidence_weighted_coverage(
        jd_weights: dict[str, float],
        evidence_weights: dict[str, float],
) -> float:
    total = sum(jd_weights.values())
    if total <= 0.0:
        return 0.0
    covered = sum(
        weight * evidence_weights.get(skill, 0.0)
        for skill, weight in jd_weights.items()
    )
    return float(np.clip(covered / total, 0.0, 1.0))


def section_fit_score(jd_vec: np.ndarray,
                      section_vec: np.ndarray) -> float:
    if jd_vec.size == 0 or section_vec.size == 0:
        return 0.0
    return float(np.clip(np.dot(jd_vec, section_vec), 0.0, 1.0))


def combine(semantic: float, skills: float, section: float,
            settings: Settings,
            calibrator: IsotonicRegression | None = None) -> ScoreParts:
    return _combine(
        semantic,
        skills,
        section,
        settings,
        settings.semantic_weight,
        settings.skill_weight,
        settings.section_weight,
        calibrator,
    )


def combine_with_signal(
        semantic: float,
        skills: float,
        section: float,
        settings: Settings,
        hard_skill_count: int,
        soft_skill_count: int,
        hard_matched_count: int,
        calibrator: IsotonicRegression | None = None,
) -> ScoreParts:
    limited_signal = (
        hard_skill_count < settings.min_hard_skills
        and hard_skill_count + 0.5 * soft_skill_count
        < settings.min_hard_skills
    )
    semantic_weight = settings.semantic_weight
    skill_weight = settings.skill_weight
    section_weight = settings.section_weight
    if limited_signal:
        remaining = semantic_weight + section_weight
        if remaining > 0:
            semantic_weight /= remaining
            section_weight /= remaining
        else:
            semantic_weight = 1.0
            section_weight = 0.0
        skill_weight = 0.0
    penalty = (
        0.6
        if (
            hard_skill_count >= settings.min_hard_skills
            and hard_matched_count == 0
        )
        else 1.0
    )
    return _combine(
        semantic,
        skills,
        section,
        settings,
        semantic_weight,
        skill_weight,
        section_weight,
        calibrator,
        limited_signal,
        penalty,
    )


def _combine(
        semantic: float,
        skills: float,
        section: float,
        settings: Settings,
        semantic_weight: float,
        skill_weight: float,
        section_weight: float,
        calibrator: IsotonicRegression | None = None,
        limited_signal: bool = False,
        penalty: float = 1.0,
) -> ScoreParts:
    linear_semantic = calibrate(
        semantic, settings.semantic_low, settings.semantic_high
    )
    calibrated = semantic_probability(semantic, settings, calibrator)
    weighted = (
        semantic_weight * linear_semantic
        + skill_weight * skills
        + section_weight * section
    )
    final = float(np.clip(weighted * penalty, 0.0, 1.0))
    sem_i = int(round(linear_semantic * 100))
    sk_i = int(round(skills * 100))
    sec_i = int(round(section * 100))
    return ScoreParts(
        semantic=linear_semantic,
        calibrated_semantic=calibrated,
        skill_coverage=skills,
        section_fit=section,
        final=final,
        semantic_display=sem_i,
        skills_display=sk_i,
        section_display=sec_i,
        match_percentage=int(round(final * 100)),
        semantic_weight=semantic_weight,
        skill_weight=skill_weight,
        section_weight=section_weight,
        limited_skill_signal=limited_signal,
        hard_skill_penalty=penalty,
    )
