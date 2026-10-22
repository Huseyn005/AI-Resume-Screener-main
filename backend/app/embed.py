from __future__ import annotations
import asyncio
from typing import Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.config import Settings


def model_name(settings: Settings) -> str:
    return settings.model_path()


def load_model(settings: Settings, model_path: str | None = None) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = model_path or model_name(settings)
    try:
        return SentenceTransformer(name, device=device)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Embedding model '{name}' was not found in the local cache. "
            "The model must be baked into the image at build time; "
            "EMBED_MODEL must not be overridden on the host."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not access embedding model '{name}' because of an OS "
            f"error ({type(exc).__name__}): {exc}"
        ) from exc


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
    texts = _truncate_texts(model, texts)
    out = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(out, dtype=np.float32)


def _truncate_texts(model: Any, texts: list[str]) -> list[str]:
    tokenizer = getattr(model, "tokenizer", None)
    max_seq_length = getattr(model, "max_seq_length", None)
    if (
        tokenizer is None
        or not callable(tokenizer)
        or max_seq_length is None
    ):
        return texts

    limit = int(max_seq_length)
    if limit <= 0:
        raise ValueError("model.max_seq_length must be a positive integer")
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise ValueError("Tokenizer cannot decode tokenized inputs")
    encode = getattr(tokenizer, "encode", None)
    can_recount = callable(encode)
    special_token_count = getattr(
        tokenizer, "num_special_tokens_to_add", None
    )
    minimum_tokens = (
        int(special_token_count(pair=False))
        if callable(special_token_count) else 0
    )
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=limit,
        padding=False,
    )
    input_ids = encoded["input_ids"]
    truncated: list[str] = []
    for token_ids in input_ids:
        end = min(len(token_ids), limit)
        text = decode(
            token_ids[:end],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        while (
            can_recount
            and len(encode(
                text, add_special_tokens=True, verbose=False
            )) > limit
            and end > minimum_tokens
        ):
            end -= 1
            text = decode(
                token_ids[:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        if (
            can_recount
            and len(encode(
                text, add_special_tokens=True, verbose=False
            )) > limit
        ):
            raise ValueError(
                "Decoded input still exceeds model.max_seq_length"
            )
        truncated.append(text)
    return truncated


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
