from __future__ import annotations

import numpy as np
import pytest

from app import embed
from app.config import Settings
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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (FileNotFoundError("missing"), "must be baked into the image"),
        (PermissionError("denied"), "PermissionError"),
    ],
)
def test_model_load_distinguishes_missing_model_from_os_errors(
        monkeypatch: pytest.MonkeyPatch,
        error: OSError,
        expected: str,
) -> None:
    monkeypatch.setattr(embed.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        embed, "model_name", lambda _settings: "local-model"
    )

    def fail_load(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(embed, "SentenceTransformer", fail_load)
    with pytest.raises(RuntimeError, match=expected):
        embed.load_model(Settings(_env_file=None))


def test_model_load_accepts_explicit_training_model_path(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = []
    monkeypatch.setattr(embed.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        embed,
        "SentenceTransformer",
        lambda name, device: loaded.append((name, device)) or object(),
    )

    embed.load_model(
        Settings(_env_file=None, embed_model="base-model"),
        model_path="local-finetuned-output",
    )

    assert loaded == [("local-finetuned-output", "cpu")]
