from __future__ import annotations
import asyncio
from typing import Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.config import Settings


def model_name(settings: Settings) -> str:
    return settings.model_path()


def load_model(settings: Settings) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name(settings), device=device)


def embedding_dim(model: Any) -> int:
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        return int(getter())
    vec = encode_sync(model, ["dimension"])
    return int(vec.shape[1])


def encode_sync(model: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        dim = embedding_dim(model)
        return np.zeros((0, dim), dtype=np.float32)
    out = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(out, dtype=np.float32)


async def encode_texts(model: Any, texts: list[str]) -> np.ndarray:
    return await asyncio.to_thread(encode_sync, model, texts)


def mean_pool(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    pooled = vectors.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm > 0:
        pooled = pooled / norm
    return pooled.astype(np.float32)
