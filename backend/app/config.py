from functools import lru_cache
from math import isclose
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_DIR = BACKEND_ROOT / "models" / "finetuned"
CALIBRATOR_PATH = BACKEND_ROOT / "models" / "calibrator.pkl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allowed_origins: str = (
        "pushcv.vercel.app, statcv.vercel.app, pullcv.vercel.app, taphire.vercel.app, dragcv.vercel.app,http://localhost:5173,http://localhost:8080"
    )
    semantic_weight: float = 0.5
    skill_weight: float = 0.40
    section_weight: float = 0.10
    experience_skill_weight: float = 1.0
    project_skill_weight: float = 1.0
    skills_skill_weight: float = 0.6
    summary_skill_weight: float = 0.3
    education_skill_weight: float = 0.3
    other_skill_weight: float = 0.3
    debug_endpoints: bool = False
    semantic_low: float = 0.20
    semantic_high: float = 0.75
    skill_cosine_threshold: float = 0.72
    min_hard_skills: int = 4
    chunk_tokens: int = 200
    chunk_overlap: int = 40
    max_pdf_bytes: int = 10 * 1024 * 1024
    min_text_chars: int = 200
    hidden_font_size: float = 2.0

    @model_validator(mode="after")
    def validate_score_weights(self) -> "Settings":
        if self.min_hard_skills < 1:
            raise ValueError("min_hard_skills must be at least 1")
        total = (
            self.semantic_weight
            + self.skill_weight
            + self.section_weight
        )
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "semantic_weight, skill_weight, and section_weight "
                f"must sum to 1.0 (got {total:.6g})"
            )
        evidence_weights = (
            self.experience_skill_weight,
            self.project_skill_weight,
            self.skills_skill_weight,
            self.summary_skill_weight,
            self.education_skill_weight,
            self.other_skill_weight,
        )
        if any(not 0.0 <= weight <= 1.0 for weight in evidence_weights):
            raise ValueError("skill evidence weights must be between 0 and 1")
        return self

    def origin_list(self) -> list[str]:
        return [p.strip() for p in self.allowed_origins.split(",")
                if p.strip()]

    def model_path(self) -> str:
        if FINETUNED_DIR.is_dir() and any(FINETUNED_DIR.iterdir()):
            return str(FINETUNED_DIR)
        return self.embed_model

    def calibrator_path(self) -> Path:
        return CALIBRATOR_PATH


@lru_cache
def get_settings() -> Settings:
    return Settings()
