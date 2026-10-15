from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re

import numpy as np
import pymupdf

from app.config import Settings
from tools import explain_scores


class FakeTokenizer:
    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            verbose: bool = True,
    ) -> list[int]:
        words = re.findall(r"[A-Za-z]+|[0-9]+|[^\w\s]", text)
        tokens = [1] * len(words)
        return [2, *tokens, 3] if add_special_tokens else tokens


class FakeModel:
    max_seq_length = 256
    tokenizer = FakeTokenizer()

    def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(16, dtype=np.float32)
            index = int.from_bytes(
                hashlib.sha256(text.casefold().encode()).digest()[:4],
                "big",
            ) % len(vector)
            vector[index] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def generated_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    text = (
        "JANE DOE\nAutomation Engineer\nSkills\nPython, PLC, SCADA\n"
        "Experience\n"
        + "Designed automation systems using Python, PLC, and SCADA. " * 8
    )
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 760), text, fontsize=10)
    data = document.tobytes()
    document.close()
    return data


def test_cli_explains_generated_pdf(
        tmp_path,
        monkeypatch,
        capsys,
) -> None:
    pdf_path = tmp_path / "generated.pdf"
    pdf_path.write_bytes(generated_pdf())
    jd_path = tmp_path / "job-description.txt"
    jd_path.write_text(
        "Automation Engineer requiring Python, PLC, SCADA, HMI, "
        "and industrial systems experience.",
        encoding="utf-8",
    )
    settings = Settings(min_text_chars=1)
    monkeypatch.setattr(explain_scores, "get_settings", lambda: settings)
    monkeypatch.setattr(
        explain_scores, "load_model", lambda _settings: FakeModel()
    )
    args = argparse.Namespace(
        jd_file=jd_path,
        pdfs=str(tmp_path / "*.pdf"),
        no_calibrator=True,
    )

    asyncio.run(explain_scores.run(args))

    detail = json.loads(capsys.readouterr().out)
    assert detail["candidate_name"] == "JANE DOE"
    assert detail["role"] == "Automation Engineer"
    assert detail["sections"]["experience"] > 0
    assert detail["jd_skills"]
    assert detail["detected_domains"]
    assert detail["matched_skills"]["lexical"]
    assert all(
        "cosine" in skill for skill in detail["matched_skills"]["lexical"]
    )
    assert detail["top_chunks"]
    assert detail["raw_semantic"] is not None
    assert detail["calibrated_semantic"] is not None
    assert detail["formula"]["match_percentage"] >= 0
    assert all(len(chunk["text"]) <= 123 for chunk in detail["top_chunks"])


def test_cli_writes_utf8_json_file(
        tmp_path,
        monkeypatch,
        capsys,
) -> None:
    pdf_path = tmp_path / "generated.pdf"
    pdf_path.write_bytes(generated_pdf())
    jd_path = tmp_path / "job-description.txt"
    jd_path.write_text(
        "Automation Engineer requiring Python, PLC, SCADA, HMI, "
        "and industrial systems experience.",
        encoding="utf-8",
    )
    output_path = tmp_path / "scores.json"
    settings = Settings(min_text_chars=1)
    monkeypatch.setattr(explain_scores, "get_settings", lambda: settings)
    monkeypatch.setattr(
        explain_scores, "load_model", lambda _settings: FakeModel()
    )
    args = argparse.Namespace(
        jd_file=jd_path,
        pdfs=str(tmp_path / "*.pdf"),
        no_calibrator=True,
        out=output_path,
    )

    asyncio.run(explain_scores.run(args))

    assert capsys.readouterr().out == ""
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["candidate_name"] == "JANE DOE"
