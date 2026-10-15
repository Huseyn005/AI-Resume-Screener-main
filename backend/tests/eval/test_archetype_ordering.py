from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from app import main
from app.config import Settings
from app.skills import load_skills

EVAL_DIR = Path(__file__).resolve().parent
ARCHETYPE_JD = (
    "Process Automation Engineer. Required skills: PLC, SCADA, HMI, DCS, "
    "Modbus, PID, MATLAB, instrumentation, process control, "
    "troubleshooting, safety systems, and manufacturing. Embedded, ESP32, "
    "and IoT are optional."
)


class TokenModel:
    max_seq_length = 256

    def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), 2048), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9+#]+", text.casefold()):
                index = int.from_bytes(
                    hashlib.sha256(token.encode()).digest()[:4],
                    "big",
                ) % vectors.shape[1]
                vectors[row, index] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vectors[row])
                if norm:
                    vectors[row] /= norm
        return vectors


def test_synthetic_archetype_ordering(
        monkeypatch,
) -> None:
    resumes: list[dict[str, Any]] = json.loads(
        (EVAL_DIR / "archetype_resumes.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(main.app.state, "model", TokenModel(), raising=False)
    monkeypatch.setattr(
        main.app.state, "lexicon", load_skills(), raising=False
    )
    monkeypatch.setattr(
        main.app.state, "calibrator", None, raising=False
    )
    settings = Settings()

    async def score_all() -> dict[str, int]:
        scores = {}
        for resume in resumes:
            result = await main._score_text(
                f"{resume['archetype']}.pdf",
                resume["text"],
                [],
                resume["name"],
                ARCHETYPE_JD,
                settings,
            )
            scores[resume["archetype"]] = result.match_percentage
        return scores

    scores = asyncio.run(score_all())
    assert scores["practitioner"] > scores["coursework-only"]
    assert scores["intern"] > scores["coursework-only"]
    assert scores["intern"] > scores["embedded-student"]
