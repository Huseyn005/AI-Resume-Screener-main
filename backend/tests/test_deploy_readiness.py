from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.extract import extract_pdf

JD = (
    "We need a Python software engineer with FastAPI experience to build "
    "reliable APIs and deliver production projects."
)


class FakeModel:
    max_seq_length = 256
    tokenizer = None

    def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(128, dtype=np.float32)
            index = int.from_bytes(
                hashlib.sha256(text.casefold().encode()).digest()[:4],
                "big",
            ) % len(vector)
            vector[index] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def make_pdf(text: str, pages: int = 1) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 760),
            text,
            fontsize=10,
        )
    data = document.tobytes()
    document.close()
    return data


RESUME = (
    "Jane Doe\nExperience\n"
    + "Python FastAPI software development and project delivery. " * 8
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "load_model", lambda _settings: FakeModel())
    monkeypatch.setattr(main, "get_settings", lambda: Settings())
    with TestClient(main.app) as test_client:
        yield test_client


def post(client: TestClient, filename: str, content: bytes,
         content_type: str = "application/pdf"):
    return client.post(
        "/analyze",
        data={"job_description": JD},
        files=[("resumes", (filename, content, content_type))],
    )


def test_content_length_limit_returns_413(client: TestClient) -> None:
    client.app.state.settings.max_request_bytes = 100

    response = post(client, "resume.pdf", make_pdf(RESUME))

    assert response.status_code == 413
    assert response.json() == {"detail": "Request exceeds the maximum size"}


def test_per_pdf_byte_limit_returns_too_large_flag(client: TestClient) -> None:
    client.app.state.settings.max_pdf_bytes = 100

    response = post(client, "resume.pdf", make_pdf(RESUME))

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "error"
    assert result["flags"] == ["too_large"]


def test_rate_limit_returns_429_with_retry_after(client: TestClient) -> None:
    client.app.state.settings.rate_limit_per_minute = 1
    first = post(client, "first.pdf", make_pdf(RESUME))
    second = post(client, "second.pdf", make_pdf(RESUME))

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1
    assert second.json() == {"detail": "Analysis rate limit exceeded"}
    assert client.get("/health").status_code == 200


def test_full_analysis_queue_returns_503(
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.app.state.settings.max_concurrent_analyses = 1
    client.app.state.settings.queue_timeout_seconds = 0.05
    client.app.state.analysis_semaphore = asyncio.Semaphore(1)
    started_extracting = threading.Event()
    original_extract = main.extract_pdf

    def slow_extract(*args, **kwargs):
        started_extracting.set()
        time.sleep(0.5)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(main, "extract_pdf", slow_extract)
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = executor.submit(post, client, "first.pdf", make_pdf(RESUME))
        assert started_extracting.wait(timeout=1)
        response = post(client, "second.pdf", make_pdf(RESUME))

    assert running.result(timeout=5).status_code == 200

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json() == {
        "detail": "Analysis queue is full; retry shortly"
    }


def test_too_many_pages_returns_error_stub(client: TestClient) -> None:
    client.app.state.settings.max_pdf_pages = 1

    response = post(client, "resume.pdf", make_pdf(RESUME, pages=2))

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "error"
    assert result["flags"] == ["too_many_pages"]
    assert extract_pdf(make_pdf(RESUME, pages=2), max_pages=1).flags == [
        "too_many_pages"
    ]


def test_mixed_valid_and_unsupported_files_return_one_result_each(
        client: TestClient,
) -> None:
    response = client.post(
        "/analyze",
        data={"job_description": JD},
        files=[
            ("resumes", ("valid.pdf", make_pdf(RESUME), "application/pdf")),
            ("resumes", ("resume.txt", b"plain text", "text/plain")),
        ],
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    invalid = next(
        result for result in results if result["filename"] == "resume.txt"
    )
    assert invalid["status"] == "error"
    assert invalid["flags"] == ["unsupported_type"]


def test_cors_preflight_accepts_explicit_and_regex_origins(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        allowed_origins="https://allowed.example",
        allowed_origin_regex=r"^https://preview-[a-z0-9-]+\.vercel\.app$",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    app = main.create_app()
    client = TestClient(app)

    for origin in (
        "https://allowed.example",
        "https://preview-feature-123.vercel.app",
    ):
        response = client.options(
            "/analyze",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "access-control-allow-credentials" not in response.headers

    denied = client.options(
        "/analyze",
        headers={
            "Origin": "https://other.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert denied.status_code == 400


def test_health_stays_responsive_while_extraction_is_slow(
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    started_extracting = threading.Event()
    original_extract = main.extract_pdf

    def slow_extract(*args, **kwargs):
        started_extracting.set()
        time.sleep(1.2)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(main, "extract_pdf", slow_extract)
    with ThreadPoolExecutor(max_workers=1) as executor:
        analyze_future = executor.submit(
            post, client, "resume.pdf", make_pdf(RESUME)
        )
        assert started_extracting.wait(timeout=1)
        started = time.perf_counter()
        health = client.get("/health")
        elapsed = time.perf_counter() - started

    assert health.status_code == 200
    assert elapsed < 1
    assert analyze_future.result(timeout=5).status_code == 200
