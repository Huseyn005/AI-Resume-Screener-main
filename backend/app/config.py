from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
FINETUNED_DIR = BACKEND_ROOT / "models" / "finetuned"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allowed_origins: str = (
        "http://localhost:5173,http://localhost:8080"
    )
    semantic_weight: float = 0.45
    skill_weight: float = 0.40
    section_weight: float = 0.15
    semantic_low: float = 0.20
    semantic_high: float = 0.75
    skill_cosine_threshold: float = 0.72
    chunk_tokens: int = 200
    chunk_overlap: int = 40
    max_pdf_bytes: int = 10 * 1024 * 1024
    min_text_chars: int = 200
    hidden_font_size: float = 2.0

    def origin_list(self) -> list[str]:
        return [p.strip() for p in self.allowed_origins.split(",")
                if p.strip()]

    def model_path(self) -> str:
        if FINETUNED_DIR.is_dir() and any(FINETUNED_DIR.iterdir()):
            return str(FINETUNED_DIR)
        return self.embed_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
