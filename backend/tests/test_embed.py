from __future__ import annotations

import numpy as np

from app.embed import encode_sync
from app.skills import noun_phrases


class CharacterTokenizer:
    def __call__(
            self,
            texts: list[str],
            add_special_tokens: bool,
            truncation: bool,
            max_length: int,
            padding: bool,
    ) -> dict[str, list[list[int]]]:
        sequences = [[1, *(ord(char) for char in text), 2] for text in texts]
        if truncation:
            sequences = [sequence[:max_length] for sequence in sequences]
        return {"input_ids": sequences}

    def decode(
            self,
            token_ids: list[int],
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
    ) -> str:
        values = token_ids[1:-1] if skip_special_tokens else token_ids
        return "".join(chr(token_id) for token_id in values)


class TokenLimitModel:
    max_seq_length = 64
    tokenizer = CharacterTokenizer()

    def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        for text in texts:
            token_count = len(self.tokenizer(
                [text],
                add_special_tokens=True,
                truncation=False,
                max_length=self.max_seq_length,
                padding=False,
            )["input_ids"][0])
            assert token_count <= self.max_seq_length
        return np.ones((len(texts), 3), dtype=np.float32)


def test_encode_sync_truncates_long_and_ragged_inputs_by_tokens() -> None:
    texts = ["a very short input", "x" * 2000, "another input"]

    embeddings = encode_sync(TokenLimitModel(), texts)

    assert embeddings.shape == (3, 3)


def test_noun_phrase_generation_is_bounded() -> None:
    text = " ".join(f"term{index}" for index in range(1000))

    phrases = noun_phrases(text)

    assert len(phrases) <= 256
    assert all(len(phrase) <= 128 for phrase in phrases)
    assert noun_phrases("x" * 10000) == []
