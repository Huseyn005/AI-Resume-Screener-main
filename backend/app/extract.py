from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import fitz

PDF_MAGIC = b"%PDF"
NAME_RE = re.compile(
    r"^[A-Za-zÀ-ž][A-Za-zÀ-ž'.-]+"
    r"(?:\s+[A-Za-zÀ-ž][A-Za-zÀ-ž'.-]+){1,3}$"
)


@dataclass
class Extracted:
    text: str
    flags: list[str] = field(default_factory=list)
    status: str = "ok"
    candidate_name: str = ""
    heading: str = ""


def name_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = re.sub(r"[_-]+", " ", stem)
    cleaned = re.sub(r"\b(resume|cv)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Unnamed candidate"


def _is_white(color: int) -> bool:
    r = (color >> 16) & 255
    g = (color >> 8) & 255
    b = color & 255
    return r >= 240 and g >= 240 and b >= 240


def _is_hidden(span: dict, size_limit: float) -> bool:
    size = float(span.get("size") or 0)
    color = int(span.get("color") or 0)
    return size < size_limit or _is_white(color)


def extract_pdf(
        data: bytes,
        filename: str = "resume.pdf",
        min_chars: int = 200,
        hidden_size: float = 2.0,
) -> Extracted:
    fallback = name_from_filename(filename)
    if not data.startswith(PDF_MAGIC):
        return Extracted(
            text="", flags=["invalid_pdf"], status="error",
            candidate_name=fallback, heading="",
        )
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return Extracted(
            text="", flags=["invalid_pdf"], status="error",
            candidate_name=fallback, heading="",
        )
    try:
        if doc.is_encrypted and not doc.is_pdf:
            return Extracted(
                text="", flags=["invalid_pdf"], status="error",
                candidate_name=fallback, heading="",
            )
        flags: list[str] = []
        rows: list[tuple[int, float, float, str, float]] = []
        hidden_found = False
        for page_i, page in enumerate(doc):
            payload = page.get_text("dict")
            for block in payload.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    bbox = line.get("bbox") or (0, 0, 0, 0)
                    visible: list[str] = []
                    sizes: list[float] = []
                    for span in line.get("spans", []):
                        piece = (span.get("text") or "")
                        if not piece.strip():
                            continue
                        if _is_hidden(span, hidden_size):
                            hidden_found = True
                            continue
                        visible.append(piece)
                        sizes.append(float(span.get("size") or 0))
                    if not visible:
                        continue
                    text = "".join(visible).strip()
                    if not text:
                        continue
                    size = max(sizes) if sizes else 0.0
                    rows.append(
                        (page_i, float(bbox[1]), float(bbox[0]),
                         text, size)
                    )
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        lines = [r[3] for r in rows]
        text = "\n".join(lines)
        heading = next((ln for ln in lines if ln.strip()), "")
        name = _name_from_heading(heading, rows, fallback)
        if hidden_found:
            flags.append("hidden_text")
        if len(text.strip()) < min_chars:
            flags.append("scanned_pdf")
            return Extracted(
                text=text, flags=flags, status="needs_ocr",
                candidate_name=name, heading=heading,
            )
        return Extracted(
            text=text, flags=flags, status="ok",
            candidate_name=name, heading=heading,
        )
    finally:
        doc.close()


def _name_from_heading(
        heading: str,
        rows: list[tuple[int, float, float, str, float]],
        fallback: str,
) -> str:
    first = heading.strip()
    if first and len(first) < 80 and NAME_RE.match(first):
        return first
    page0 = [r for r in rows if r[0] == 0]
    if page0:
        biggest = max(page0, key=lambda r: r[4])
        cand = biggest[3].strip()
        if cand and len(cand) < 80 and NAME_RE.match(cand):
            return cand
    if first and len(first) < 80 and not _looks_like_heading_label(first):
        return first
    return fallback


def _looks_like_heading_label(line: str) -> bool:
    t = line.strip().lower().rstrip(":")
    labels = {
        "experience", "work experience", "education", "skills",
        "projects", "summary", "objective", "profile",
    }
    return t in labels
