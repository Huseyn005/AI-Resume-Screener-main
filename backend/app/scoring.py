from __future__ import annotations
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from app.config import Settings


@dataclass
class ScoreParts:
    semantic: float
    skill_coverage: float
    section_fit: float
    final: float
    semantic_display: int
    skills_display: int
    section_display: int
    match_percentage: int


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


def section_fit_score(jd_vec: np.ndarray,
                      section_vec: np.ndarray) -> float:
    if jd_vec.size == 0 or section_vec.size == 0:
        return 0.0
    return float(np.clip(np.dot(jd_vec, section_vec), 0.0, 1.0))


def combine(semantic: float, skills: float, section: float,
            settings: Settings,
            calibrator: IsotonicRegression | None = None) -> ScoreParts:
    calibrated = semantic_probability(semantic, settings, calibrator)
    semantic_support = 0.5 + 0.5 * float(np.sqrt(skills))
    final = (
        settings.semantic_weight * calibrated * semantic_support
        + settings.skill_weight * skills
        + settings.section_weight * section
    )
    final = float(np.clip(final, 0.0, 1.0))
    sem_i = int(round(calibrated * 100))
    sk_i = int(round(skills * 100))
    sec_i = int(round(section * 100))
    return ScoreParts(
        semantic=calibrated,
        skill_coverage=skills,
        section_fit=section,
        final=final,
        semantic_display=sem_i,
        skills_display=sk_i,
        section_display=sec_i,
        match_percentage=int(round(final * 100)),
    )
