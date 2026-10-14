from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from app.preprocess import all_chunks, chunk_text, split_sections
from app.scoring import topk_mean

ML_ROOT = Path(__file__).resolve().parent
DATA_DIR = ML_ROOT / "data"
BACKEND_ROOT = ML_ROOT.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def resume_chunks(text: str) -> list[str]:
    sections = split_sections(text)
    return [
        chunk
        for _section, chunk in all_chunks(sections, size=200, overlap=40)
    ]


def jd_chunks(text: str) -> list[str]:
    return chunk_text(text, size=200, overlap=40)


def encode_chunks(
        model: Any,
        records: list[dict[str, Any]],
        batch_size: int = 64,
) -> tuple[list[list[str]], list[np.ndarray]]:
    chunks_by_resume = [resume_chunks(row["text"]) for row in records]
    flat_chunks = [chunk for chunks in chunks_by_resume for chunk in chunks]
    flat_embeddings = model.encode(
        flat_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    vectors = np.asarray(flat_embeddings, dtype=np.float32)
    output: list[np.ndarray] = []
    offset = 0
    for chunks in chunks_by_resume:
        output.append(vectors[offset:offset + len(chunks)])
        offset += len(chunks)
    return chunks_by_resume, output


def semantic_scores(
        model: Any,
        jd: str,
        resume_embeddings: list[np.ndarray],
) -> list[float]:
    jd_parts = jd_chunks(jd)
    if not jd_parts:
        return [0.0] * len(resume_embeddings)
    jd_embeddings = np.asarray(
        model.encode(
            jd_parts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    return [
        topk_mean(embeddings, jd_embeddings, k=3)
        for embeddings in resume_embeddings
    ]
