from __future__ import annotations

import argparse
import json
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pymupdf

ANALYZE_FIELDS = {
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


def make_pdf() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    text = (
        "Synthetic Candidate\nExperience\n"
        + "Python FastAPI software development and project delivery. " * 8
    )
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 760), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def request(url: str, data: bytes | None = None,
            headers: dict[str, str] | None = None) -> bytes:
    req = Request(url, data=data, headers=headers or {})
    with urlopen(req, timeout=180) as response:
        return response.read()


def run(url: str) -> None:
    base_url = url.rstrip("/")
    health = json.loads(request(f"{base_url}/health"))
    if health.get("status") != "ok":
        raise RuntimeError(f"Health check failed: {health!r}")

    boundary = uuid.uuid4().hex
    jd = (
        "We need a Python software engineer with FastAPI experience to "
        "build reliable APIs and deliver production projects."
    )
    parts = [
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="job_description"\r\n\r\n'
        f"{jd}\r\n",
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="resumes"; '
        'filename="smoke-test.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n",
    ]
    body = parts[0].encode() + parts[1].encode() + make_pdf()
    body += f"\r\n--{boundary}--\r\n".encode()
    started = time.perf_counter()
    result_body = request(
        f"{base_url}/analyze",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    latency = time.perf_counter() - started
    payload = json.loads(result_body)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"Unexpected analyze response: {payload!r}")
    result = results[0]
    if set(result) != ANALYZE_FIELDS or result.get("status") != "ok":
        raise RuntimeError(f"Unexpected analyze result: {result!r}")
    print(f"Smoke test passed; POST /analyze latency: {latency:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the backend API")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    try:
        run(args.url)
    except (
        HTTPError,
        URLError,
        TimeoutError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Smoke test failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
