from __future__ import annotations
from datetime import date
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
    r"\b(?:designer|engineer|developer|manager|analyst|"
    r"scientist|lead|director|specialist|consultant|"
    r"architect|marketer|recruiter|accountant|technician|"
    r"operator|administrator|executive|officer|teacher|nurse|"
    r"doctor|writer|editor|strategist|coordinator|trainer)\b",
    re.I,
)
DEGREE_RE = re.compile(
    r"\b(?:bachelor|master|bsc|msc|university|college|degree|"
    r"ph\.?d\.?|doctorate|diploma)\b",
    re.I,
)
COMPANY_RE = re.compile(
    r"\b(?:incorporated|inc\.?|llc|ltd\.?|limited|corp\.?|"
    r"corporation|gmbh|company|co\.)\b",
    re.I,
)
DATE_TOKEN = (
    r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+)?"
    r"(?:19|20)\d{2}|(?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN})\s*(?:-|–|—|to)\s*"
    rf"(?P<end>{DATE_TOKEN}|present|current|now)",
    re.I,
)
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
    lines = text.splitlines()
    in_experience = False
    seen_heading = False
    top_lines: list[str] = []
    experience_lines: list[str] = []
    for line in lines:
        key = _heading_key(line)
        if key:
            seen_heading = True
            in_experience = key == "experience"
            continue
        if in_experience:
            experience_lines.append(line)
        elif not seen_heading:
            top_lines.append(line)
    for line in top_lines[:12]:
        title = _role_candidate(line)
        if title:
            return title
    return _most_recent_role(experience_lines)


def years_experience(text: str) -> str:
    sections = split_sections(text)
    ranges = _date_ranges(sections.get("experience", ""))
    if not ranges:
        return ""
    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    months = sum(end - start for start, end in merged)
    if months < 12:
        return f"{max(months, 1)} months"
    years, remainder = divmod(months, 12)
    if not remainder:
        return f"{years} year" if years == 1 else f"{years} years"
    return f"{years} years {remainder} months"


def _role_candidate(line: str) -> str:
    parts = re.split(r"\s*[|•,]\s*|\s+at\s+", line.strip(), flags=re.I)
    for part in parts:
        title = re.sub(r"\s+", " ", part).strip(" ,:-")
        if (
            title
            and len(title) < 60
            and not DEGREE_RE.search(title)
            and not COMPANY_RE.search(title)
            and ROLE_RE.search(title)
        ):
            return title
    return ""


def _most_recent_role(lines: list[str]) -> str:
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        title = _role_candidate(line)
        if not title:
            continue
        nearby = range(max(0, index - 2), min(len(lines), index + 5))
        dates = [
            (abs(date_index - index), _date_range(match)[1])
            for date_index in nearby
            for match in DATE_RANGE_RE.finditer(lines[date_index])
        ]
        nearest = min((distance for distance, _ in dates), default=None)
        end = max(
            (date_end for distance, date_end in dates if distance == nearest),
            default=-1,
        )
        candidates.append((end, -index, title))
    if not candidates:
        return ""
    return max(candidates)[2]


def _date_ranges(text: str) -> list[tuple[int, int]]:
    return [
        _date_range(match)
        for match in DATE_RANGE_RE.finditer(text)
    ]


def _date_range(match: re.Match[str]) -> tuple[int, int]:
    start = _date_index(match.group("start"))
    end_text = match.group("end")
    if end_text.casefold() in {"present", "current", "now"}:
        today = date.today()
        end = today.year * 12 + today.month - 1
    else:
        end = _date_index(end_text)
    return start, max(start + 1, end)


def _date_index(value: str) -> int:
    value = value.strip()
    month_match = re.match(r"([A-Za-z]+)\s+(\d{4})", value)
    numeric_match = re.match(r"(\d{1,2})[/.-](\d{4})", value)
    if month_match:
        month_name, year_text = month_match.groups()
        month_names = (
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        )
        month = next(
            i for i, name in enumerate(month_names, 1)
            if month_name.casefold().startswith(name)
        )
        year = int(year_text)
    elif numeric_match:
        month, year = map(int, numeric_match.groups())
    else:
        year = int(value)
        month = 1
    return year * 12 + month - 1
