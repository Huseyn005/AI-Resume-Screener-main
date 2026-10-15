from __future__ import annotations

import hashlib
import re

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware

from app import main
from app.config import Settings
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
            verbose: bool = True,
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


def test_ambiguous_generic_terms_are_not_required_skills() -> None:
    lexicon = load_skills()

    assert extract_lexical("phoenix, lion, flow, equipment", lexicon) == []
    assert extract_lexical("Phoenix LiveView and flow control", lexicon) == [
        "phoenix liveview", "flow control",
    ]


def test_semantic_skill_hits_require_a_lexicon_phrase() -> None:
    from app.skills import semantic_skill_matches

    lexicon = load_skills()
    hits = semantic_skill_matches(
        ["plc"],
        ["phoenix", "equipment", "scada"],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        ),
        0.72,
        lexicon,
    )

    assert hits == [("plc", 1.0)]


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


def test_debug_explanation_is_opt_in_and_gated(
        client: TestClient,
) -> None:
    files = [
        (
            "resumes",
            (
                "ml_engineer.pdf",
                make_pdf(resume_text("Python, PyTorch, NLP, FastAPI")),
                "application/pdf",
            ),
        )
    ]
    data = {"job_description": JOB_DESCRIPTION}
    default = client.post("/analyze", data=data, files=files)
    assert default.status_code == 200
    assert "debug" not in default.json()

    denied = client.post(
        "/analyze?debug=true", data=data, files=files
    )
    assert denied.status_code == 403

    client.app.state.settings.debug_endpoints = True
    response = client.post(
        "/analyze?debug=true", data=data, files=files
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"results", "debug"}
    detail = payload["debug"][0]
    assert detail["candidate_name"] == payload["results"][0]["candidate_name"]
    assert {"python", "pytorch", "nlp", "fastapi"}.issubset(
        detail["jd_skills"]
    )
    assert detail["matched_skills"]["lexical"]
    assert detail["formula"]["match_percentage"] == (
        payload["results"][0]["match_percentage"]
    )
    assert detail["formula"]["expression"]
    formula = detail["formula"]
    assert formula["pre_clipped_score"] == pytest.approx(
        formula["weighted_semantic"]
        + formula["weighted_skills"]
        + formula["weighted_section_fit"]
    )
    assert formula["final_score"] == pytest.approx(
        min(1.0, max(0.0, formula["pre_clipped_score"]))
    )
    assert formula["match_percentage"] == round(
        formula["final_score"] * 100
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


def test_cors_uses_configured_origins(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        allowed_origins=(
            "https://pushcv.vercel.app,http://localhost:5173"
        )
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    app = main.create_app()

    cors = next(
        middleware for middleware in app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    assert cors.kwargs["allow_origins"] == [
        "https://pushcv.vercel.app", "http://localhost:5173",
    ]
