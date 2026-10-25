from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymupdf
import pytest
from fastapi import Request
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
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_source"] == "base"


def test_debug_request_log_hashes_client_ip(
        client: TestClient,
        caplog: pytest.LogCaptureFixture,
) -> None:
    client.app.state.settings.log_level = "DEBUG"
    with caplog.at_level(logging.DEBUG, logger="screener"):
        response = client.post(
            "/analyze",
            headers={"X-Forwarded-For": "198.51.100.42"},
            data={"job_description": JD},
            files=[
                ("resumes", (
                    "resume.pdf", make_pdf(RESUME), "application/pdf"
                )),
            ],
        )

    assert response.status_code == 200
    assert "forwarded_for_entries=1" in caplog.text
    assert "client_hash=" in caplog.text
    assert "198.51.100.42" not in caplog.text


def _request(x_forwarded_for: str | None, host: str = "proxy") -> Request:
    headers = []
    if x_forwarded_for is not None:
        headers.append(
            (b"x-forwarded-for", x_forwarded_for.encode("ascii"))
        )
    return Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/analyze",
        "raw_path": b"/analyze",
        "query_string": b"",
        "headers": headers,
        "client": (host, 1234),
        "server": ("test", 80),
    })


def test_rate_limit_ignores_spoofed_leftmost_forwarded_value(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    settings.rate_limit_per_minute = 1
    settings.trusted_proxy_count = 1
    request_one = _request("spoofed-one, actual-client")
    request_two = _request("spoofed-two, actual-client")

    assert asyncio.run(
        main._rate_limit_retry_after(request_one, settings)
    ) is None
    assert asyncio.run(
        main._rate_limit_retry_after(request_two, settings)
    ) is not None
    assert set(client.app.state.rate_limit_hits) == {"actual-client"}


def test_rate_limit_uses_nth_forwarded_value_from_the_right(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    settings.trusted_proxy_count = 2

    assert asyncio.run(
        main._rate_limit_retry_after(
            _request("untrusted, selected-client, nearest-proxy"),
            settings,
        )
    ) is None
    assert set(client.app.state.rate_limit_hits) == {"selected-client"}


def test_rate_limit_missing_forwarded_header_uses_client_host(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    settings.trusted_proxy_count = 1

    assert asyncio.run(
        main._rate_limit_retry_after(_request(None, "direct-client"), settings)
    ) is None
    assert set(client.app.state.rate_limit_hits) == {"direct-client"}


def test_rate_limit_fewer_forwarded_entries_falls_back_to_client_host(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    settings.trusted_proxy_count = 3

    assert asyncio.run(
        main._rate_limit_retry_after(
            _request("one, two", "fallback-client"),
            settings,
        )
    ) is None
    assert set(client.app.state.rate_limit_hits) == {"fallback-client"}


def test_zero_trusted_proxies_ignores_forwarded_header(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    settings.trusted_proxy_count = 0

    assert asyncio.run(
        main._rate_limit_retry_after(
            _request("spoofed-client", "socket-client"),
            settings,
        )
    ) is None
    assert set(client.app.state.rate_limit_hits) == {"socket-client"}


def test_rate_limit_client_tracking_is_capped_and_evicts_oldest(
        client: TestClient,
) -> None:
    settings = client.app.state.settings
    now = time.monotonic()
    hits = client.app.state.rate_limit_hits
    hits.update({
        f"client-{index}": main.deque([now - (10_000 - index) / 1000])
        for index in range(main.MAX_TRACKED_CLIENTS)
    })

    asyncio.run(
        main._rate_limit_retry_after(_request(None, "new-client"), settings)
    )

    assert len(hits) == main.MAX_TRACKED_CLIENTS
    assert "client-0" not in hits
    assert "new-client" in hits


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


def test_default_allowed_origins_are_original(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ALLOWED_ORIGIN_REGEX", raising=False)
    settings = Settings(_env_file=None)
    assert settings.origin_list() == [
        "https://pushcv.vercel.app",
        "https://statcv.vercel.app",
        "https://pullcv.vercel.app",
        "https://taphire.vercel.app",
        "https://dragcv.vercel.app",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    assert settings.allowed_origin_regex == ""


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
