from __future__ import annotations
import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from app import config, scoring
from app.config import Settings
from app.scoring import combine, load_calibrator
from ml.train import round_robin


def test_score_weights_must_sum_to_one() -> None:
    with pytest.raises(
            ValueError,
            match="semantic_weight, skill_weight, and section_weight "
                  "must sum to 1.0",
    ):
        Settings(
            semantic_weight=0.5,
            skill_weight=0.3,
            section_weight=0.1,
        )


def test_missing_model_files_use_pretrained_and_linear_fallback(
        tmp_path,
        monkeypatch,
) -> None:
    monkeypatch.setattr(config, "FINETUNED_DIR", tmp_path / "finetuned")
    monkeypatch.setattr(
        config, "CALIBRATOR_PATH", tmp_path / "calibrator.pkl"
    )
    settings = Settings(embed_model="pretrained-model-id")
    assert settings.model_path() == "pretrained-model-id"
    assert load_calibrator(settings) is None
    parts = combine(0.4, 0.0, 0.0, settings)
    assert parts.semantic == scoring.calibrate(0.4, 0.2, 0.75)


def test_calibrated_semantic_score_is_bounded_and_monotonic() -> None:
    settings = Settings()
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0, 0, 1, 1]),
    )
    raw_scores = np.linspace(0.0, 0.6, 31)
    parts = [
        combine(float(score), 0.0, 0.0, settings, calibrator)
        for score in raw_scores
    ]
    calibrated = [part.semantic for part in parts]
    display = [part.semantic_display for part in parts]
    assert all(0.0 <= score <= 1.0 for score in calibrated)
    assert all(0 <= score <= 100 for score in display)
    assert calibrated == sorted(calibrated)
    assert display == sorted(display)


def test_semantic_score_is_gated_when_required_skills_are_missing() -> None:
    settings = Settings()

    missing = combine(0.9, 0.0, 0.5, settings)
    partial = combine(0.9, 0.5, 0.5, settings)

    assert missing.match_percentage <= 45
    assert partial.match_percentage > missing.match_percentage


def test_round_robin_batches_have_distinct_categories() -> None:
    rows = [
        {"category": category, "jd": str(index)}
        for category in ("a", "b", "c", "d")
        for index in range(4)
    ]
    ordered = round_robin(rows)
    for start in range(0, len(ordered), 3):
        batch = ordered[start:start + 3]
        categories = [row["category"] for row in batch]
        assert len(categories) == len(set(categories))
