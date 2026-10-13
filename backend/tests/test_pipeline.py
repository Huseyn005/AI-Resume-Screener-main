from __future__ import annotations

import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.explain import build_reason
from app.extract import extract_pdf
from app.preprocess import (
    chunk_text,
    guess_role,
    years_experience,
)
from app.skills import extract_lexical, load_skills

JOB_DESCRIPTION = (
    "We need a Machine Learning Engineer with Python, PyTorch, NLP "
    "and FastAPI experience to build and deploy ranking models."
)


class FakeTokenizer:
    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
    ) -> list[int]:
        tokens = re.findall(r"[A-Za-z]+|[0-9]+|[^\w\s]", text)
        ids = [abs(hash(token.lower())) % 10000 for token in tokens]
        return [10001, *ids, 10002] if add_special_tokens else ids


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
            vector = np.zeros(2048, dtype=np.float32)
            index = int.from_bytes(
                hashlib.sha256(text.casefold().encode()).digest()[:4],
                "big",
            ) % len(vector)
            vector[index] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 760),
        text,
        fontsize=10,
    )
    data = document.tobytes()
    document.close()
    return data


def resume_text(skills: str, role: str = "Engineer") -> str:
    return (
        f"Jane Doe\n{role}\nSkills\n{skills}\nExperience\n"
        + "Built reliable software systems and delivered projects. " * 8
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "load_model", lambda _settings: FakeModel())
    with TestClient(main.app) as test_client:
        yield test_client


def test_pdf_text_extraction() -> None:
    result = extract_pdf(make_pdf("Jane Doe\nPython developer"), min_chars=1)
    assert result.status == "ok"
    assert "Python developer" in result.text


def test_chunks_stay_within_model_token_limit() -> None:
    model = FakeModel()
    chunks = chunk_text(
        " ".join(["FastAPI/vector-database"] * 40),
        size=model.max_seq_length,
        overlap=20,
        token_counter=lambda text: len(model.tokenizer.encode(text)),
    )
    assert chunks
    assert all(
        len(model.tokenizer.encode(chunk)) <= model.max_seq_length
        for chunk in chunks
    )


def test_stopwords_are_not_required_skills() -> None:
    lexicon = load_skills()
    skills = extract_lexical(
        "We need the engineer to build with Python.", lexicon
    )
    assert skills == ["python"]
    assert not {"to", "we", "the"}.intersection(skills)


def test_javascript_alias_matching() -> None:
    assert extract_lexical("Experience with js", load_skills()) == [
        "javascript"
    ]


def test_skill_catalog_excludes_known_false_entries() -> None:
    catalog = load_skills().canonical
    assert "ranking" not in catalog
    assert "playwright python" not in catalog
    assert "advanced python scheduler" not in catalog


def test_numeric_only_entries_are_not_skills() -> None:
    lexicon = load_skills()
    assert "01" not in lexicon.canonical
    assert "1176" not in lexicon.canonical
    assert extract_lexical("01", lexicon) == []


@pytest.mark.parametrize(
    "resume_skill",
    ["programmable logic controllers", "PLCs", "PLC-based"],
)
def test_plc_aliases_match_jd_skill(resume_skill: str) -> None:
    lexicon = load_skills()
    assert extract_lexical(resume_skill, lexicon) == ["plc"]
    assert extract_lexical("plc", lexicon) == ["plc"]


def test_single_letter_skills_require_context() -> None:
    lexicon = load_skills()
    assert "c" not in extract_lexical("Used c at work", lexicon)
    assert "r" not in extract_lexical("Used r at work", lexicon)
    assert extract_lexical("Skills\nC\nExperience", lexicon) == ["c"]
    assert extract_lexical("C programming", lexicon) == ["c"]
    assert extract_lexical("R language", lexicon) == ["r"]


def test_degree_line_is_not_extracted_as_role() -> None:
    assert guess_role(
        "Bachelor of Engineering\nEducation\nUniversity of Example"
    ) == ""


def test_company_name_is_not_part_of_role() -> None:
    assert guess_role(
        "Jane Doe\nSenior Software Engineer at Acme Inc.\nExperience"
    ) == "Senior Software Engineer"


def test_role_and_experience_come_from_work_history() -> None:
    resume = (
        "Jane Doe\nBachelor of Engineering\nExperience\n"
        "Senior Software Engineer\n2022 - 2024\n"
        "Junior Developer\n2019 - 2022"
    )
    assert guess_role(resume) == "Senior Software Engineer"
    assert years_experience(resume) == "5 years"


def test_experience_is_empty_without_date_ranges() -> None:
    assert years_experience(
        "Experience\nSoftware Engineer\nWorked for 5 years"
    ) == ""


def test_top_skills_follow_jd_appearance(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        data={"job_description": JOB_DESCRIPTION},
        files=[
            (
                "resumes",
                (
                    "ml_engineer.pdf",
                    make_pdf(
                        resume_text(
                            "Python, PyTorch, NLP, FastAPI, Docker, SQL",
                            "Machine Learning Engineer",
                        )
                    ),
                    "application/pdf",
                ),
            )
        ],
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["matched"] == ["python", "pytorch", "nlp", "fastapi"]
    assert result["top_skills"] == ["python", "pytorch", "nlp"]
    assert not {"to", "we", "the"}.intersection(
        result["matched"] + result["missing_skills"]
    )


def test_matching_resume_scores_higher_and_results_are_sorted(
        client: TestClient,
) -> None:
    response = client.post(
        "/analyze",
        data={"job_description": JOB_DESCRIPTION},
        files=[
            (
                "resumes",
                (
                    "unrelated.pdf",
                    make_pdf(resume_text("React, CSS, TypeScript")),
                    "application/pdf",
                ),
            ),
            (
                "resumes",
                (
                    "matching.pdf",
                    make_pdf(
                        resume_text("Python, PyTorch, NLP, FastAPI")
                    ),
                    "application/pdf",
                ),
            ),
        ],
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["filename"] == "matching.pdf"
    assert results[0]["match_percentage"] > results[1]["match_percentage"]
    assert results[0]["match_percentage"] >= results[1]["match_percentage"]
    assert all(
        not {"to", "we", "the"}.intersection(
            result["matched"] + result["missing_skills"]
        )
        for result in results
    )
    assert set(results[0]) == {
        "filename",
        "candidate_name",
        "match_percentage",
        "top_skills",
        "missing_skills",
        "matched",
        "skills",
        "score_breakdown",
        "reason",
        "flags",
        "status",
        "role",
        "experience",
        "strengths",
        "gaps",
        "summary",
    }


def test_short_job_description_returns_422(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        data={"job_description": "Too short"},
        files=[
            (
                "resumes",
                (
                    "resume.pdf",
                    make_pdf(resume_text("Python")),
                    "application/pdf",
                ),
            )
        ],
    )
    assert response.status_code == 422


def test_text_upload_returns_415(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        data={"job_description": JOB_DESCRIPTION},
        files=[
            ("resumes", ("resume.txt", b"not a PDF", "text/plain"))
        ],
    )
    assert response.status_code == 415


def test_reason_has_two_sentences_and_omits_other_section() -> None:
    reason = build_reason(["python"], ["java", "sql"], "other")
    assert reason == (
        "This resume covers python. Key gaps include: java, sql."
    )
    assert reason.count(". ") == 1
