from __future__ import annotations
import re
from collections.abc import Callable

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I)
URL_RE = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    re.I,
)
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[-.\s])?(?:\(?\d{3}\)?[-.\s])?\d{3}[-.\s]\d{4}\b"
)
ADDR_RE = re.compile(
    r"\b\d{1,5}\s+[\w.]+\s+"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|"
    r"lane|ln|drive|dr|court|ct|way|place|pl)\.?\b",
    re.I,
)
HEADINGS = {
    "experience": re.compile(
        r"^(?:work\s+)?experience|employment|"
        r"professional\s+experience|work\s+history$",
        re.I,
    ),
    "skills": re.compile(
        r"^(?:technical\s+)?skills|core\s+competenc(?:y|ies)|"
        r"technologies|tech\s+stack$",
        re.I,
    ),
    "education": re.compile(
        r"^education|academic|qualifications|degrees?$",
        re.I,
    ),
    "projects": re.compile(
        r"^(?:personal\s+|selected\s+|key\s+)?projects$",
        re.I,
    ),
}
ROLE_RE = re.compile(
    r"(designer|engineer|developer|manager|analyst|"
    r"scientist|lead|director|specialist|consultant|"
    r"architect|marketer|recruiter|accountant)",
    re.I,
)
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\b", re.I)


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[email]", text)
    text = URL_RE.sub("[url]", text)
    text = PHONE_RE.sub("[phone]", text)
    text = ADDR_RE.sub("[address]", text)
    return text


def split_sections(text: str) -> dict[str, str]:
    buckets: dict[str, list[str]] = {
        "experience": [],
        "skills": [],
        "education": [],
        "projects": [],
        "other": [],
    }
    current = "other"
    for raw in text.splitlines():
        key = _heading_key(raw)
        if key:
            current = key
            continue
        buckets[current].append(raw)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}


def _heading_key(line: str) -> str | None:
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > 48:
        return None
    for name, pat in HEADINGS.items():
        if pat.match(stripped):
            return name
    return None


def chunk_text(
        text: str,
        size: int = 200,
        overlap: int = 40,
        token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    size = max(1, size)
    overlap = min(max(0, overlap), size - 1)
    chunks: list[str] = []
    i = 0
    while i < len(tokens):
        end = min(i + size, len(tokens))
        if token_counter:
            while (
                end > i
                and token_counter(" ".join(tokens[i:end])) > size
            ):
                end -= 1
        if end == i:
            if not token_counter:
                end = i + 1
            else:
                chunks.extend(
                    _split_long_token(tokens[i], size, token_counter)
                )
                i += 1
                continue
        chunks.append(" ".join(tokens[i:end]))
        if end == len(tokens):
            break
        i = max(i + 1, end - overlap)
    return chunks


def _split_long_token(
        token: str,
        size: int,
        token_counter: Callable[[str], int],
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(token):
        low, high = start + 1, len(token)
        end = start
        while low <= high:
            mid = (low + high) // 2
            if token_counter(token[start:mid]) <= size:
                end = mid
                low = mid + 1
            else:
                high = mid - 1
        if end == start:
            raise ValueError("A single character exceeds the token limit")
        chunks.append(token[start:end])
        start = end
    return chunks


def all_chunks(
        sections: dict[str, str],
        size: int = 200,
        overlap: int = 40,
        token_counter: Callable[[str], int] | None = None,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    order = (
        "experience", "projects", "skills", "education", "other",
    )
    for name in order:
        body = sections.get(name) or ""
        if not body:
            continue
        for chunk in chunk_text(body, size, overlap, token_counter):
            out.append((name, chunk))
    if not out:
        joined = " ".join(sections.values())
        for chunk in chunk_text(joined, size, overlap, token_counter):
            out.append(("other", chunk))
    return out


def guess_role(text: str) -> str:
    for line in text.splitlines():
        t = line.strip()
        if t and len(t) < 60 and ROLE_RE.search(t):
            return re.sub(r"\s+", " ", t)
    return "Role not listed"


def years_experience(text: str) -> str:
    nums = [int(n) for n in YEARS_RE.findall(text)]
    nums = [n for n in nums if n < 45]
    if not nums:
        return "Not stated"
    return f"{max(nums)} years"
