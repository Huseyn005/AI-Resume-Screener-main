from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pymupdf
from app.preprocess import strip_invisible_unicode

PDF_MAGIC = b"%PDF"
NAME_RE = re.compile(
    r"^[^\W\d_][\w'.’\-]*"
    r"(?:\s+[^\W\d_][\w'.’\-]*){1,3}$",
    re.UNICODE,
)
NAME_EXCLUDED_RE = re.compile(
    r"[@\d]|(?:https?://|www\.)|\b(?:"
    r"designer|engineer|developer|manager|analyst|scientist|lead|"
    r"director|specialist|consultant|architect|marketer|recruiter|"
    r"accountant|technician|operator|administrator|executive|officer|"
    r"teacher|nurse|doctor|writer|editor|strategist|coordinator|trainer"
    r")\b",
    re.I,
)
FILENAME_ROLE_RE = re.compile(
    r"\b(?:engineer|developer|manager|analyst|scientist|lead|director|"
    r"specialist|consultant|architect|marketer|recruiter|accountant|"
    r"technician|operator|administrator|executive|officer|teacher|nurse|"
    r"doctor|writer|editor|strategist|coordinator|trainer)\b",
    re.I,
)
FILENAME_TITLE_WORDS = {
    "automation", "java", "software", "senior", "junior", "principal",
    "staff", "associate", "professional", "process", "qa", "quality",
    "control", "instrument", "social", "media",
}


@dataclass(frozen=True)
class TextLine:
    page: int
    bbox: tuple[float, ...]
    text: str
    size: float
    block_bbox: tuple[float, ...]
    block_index: int
    page_width: float
    page_height: float


@dataclass
class Extracted:
    text: str
    flags: list[str] = field(default_factory=list)
    status: str = "ok"
    candidate_name: str = ""
    heading: str = ""


def name_from_filename(filename: str) -> str:
    stem = re.sub(r"[_-]+", " ", Path(filename).stem)
    tokens = [
        token for token in re.findall(r"[^\W\d_][\w'.’]*", stem, re.UNICODE)
        if not any(char.isdigit() for char in token)
        and token.casefold() not in {"resume", "cv"}
    ]
    site_prefixes = {
        "upstaff", "linkedin", "indeed", "qwikresume", "monster",
        "ziprecruiter",
    }
    if tokens and tokens[0].casefold() in site_prefixes:
        tokens = tokens[1:]

    role_positions = [
        index for index, token in enumerate(tokens)
        if FILENAME_ROLE_RE.fullmatch(token)
    ]
    if role_positions:
        role_index = role_positions[-1]
        suffix = tokens[role_index + 1:]
        candidates = [suffix, tokens[:role_index]]
    else:
        candidates = [tokens]

    for candidate in candidates:
        name_tokens = [
            token for token in candidate
            if token.casefold() not in FILENAME_TITLE_WORDS
            and not FILENAME_ROLE_RE.fullmatch(token)
        ]
        for start in range(max(0, len(name_tokens) - 4), len(name_tokens)):
            possible = " ".join(name_tokens[start:])
            if len(name_tokens[start:]) in range(2, 5) and NAME_RE.fullmatch(
                    possible
            ):
                return possible
    return "Unnamed candidate"


def _is_hidden(
        span: dict,
        size_limit: float,
        background: tuple[int, int, int] | None,
        page_rect: pymupdf.Rect,
) -> bool:
    size = float(span.get("size") or 0)
    color = int(span.get("color") or 0)
    bbox = pymupdf.Rect(span.get("bbox") or (0, 0, 0, 0))
    if size < size_limit or not page_rect.contains(bbox):
        return True

    if span.get("alpha") is not None:
        opacity = float(span["alpha"]) / 255
        if opacity <= 0.02:
            return True
    elif span.get("opacity") is not None:
        opacity = float(span["opacity"])
        if opacity > 1:
            opacity /= 255
        if opacity <= 0.02:
            return True

    if background is None:
        return False
    foreground = (
        (color >> 16) & 255,
        (color >> 8) & 255,
        color & 255,
    )
    return _contrast_ratio(foreground, background) < 1.5


def _contrast_ratio(
        foreground: tuple[int, int, int],
        background: tuple[int, int, int],
) -> float:
    def luminance(rgb: tuple[int, int, int]) -> float:
        channels = []
        for value in rgb:
            channel = value / 255
            channels.append(
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
            )
        return (
            0.2126 * channels[0]
            + 0.7152 * channels[1]
            + 0.0722 * channels[2]
        )

    first, second = luminance(foreground), luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _sample_background(
        pixmap: pymupdf.Pixmap,
        bbox: tuple[float, ...],
        zoom: float,
) -> tuple[int, int, int] | None:
    left = max(0, int(np.floor(bbox[0] * zoom)) - pixmap.x)
    top = max(0, int(np.floor(bbox[1] * zoom)) - pixmap.y)
    right = min(pixmap.width, int(np.ceil(bbox[2] * zoom)) - pixmap.x)
    bottom = min(pixmap.height, int(np.ceil(bbox[3] * zoom)) - pixmap.y)
    if left >= right or top >= bottom or pixmap.n < 3:
        return None

    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
    pixels = pixels.reshape(pixmap.height, pixmap.stride)
    pixels = pixels[:, :pixmap.width * pixmap.n]
    pixels = pixels.reshape(pixmap.height, pixmap.width, pixmap.n)
    region = pixels[top:bottom, left:right, :3].reshape(-1, 3)
    median = np.median(region, axis=0)
    return tuple(int(channel) for channel in median)


def extract_pdf(
        data: bytes,
        filename: str = "resume.pdf",
        min_chars: int = 200,
        hidden_size: float = 2.0,
        max_pages: int | None = None,
) -> Extracted:
    fallback = name_from_filename(filename)
    if not data.startswith(PDF_MAGIC):
        return Extracted(
            text="", flags=["invalid_pdf"], status="error",
            candidate_name=fallback, heading="",
        )
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
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
        if max_pages is not None and len(doc) > max_pages:
            return Extracted(
                text="", flags=["too_many_pages"], status="error",
                candidate_name=fallback, heading="",
            )
        flags: list[str] = []
        rows: list[TextLine] = []
        hidden_found = False
        for page_i, page in enumerate(doc):
            payload = page.get_text("dict")
            zoom = 1.25
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom),
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            for block_index, block in enumerate(payload.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                block_bbox = tuple(block.get("bbox") or page.rect)
                for line in block.get("lines", []):
                    bbox = line.get("bbox") or (0, 0, 0, 0)
                    visible: list[str] = []
                    sizes: list[float] = []
                    for span in line.get("spans", []):
                        piece = (span.get("text") or "")
                        if not piece.strip():
                            continue
                        span_bbox = tuple(span.get("bbox") or bbox)
                        background = _sample_background(
                            pixmap, span_bbox, zoom
                        )
                        if _is_hidden(
                                span, hidden_size, background, page.rect
                        ):
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
                    rows.append(TextLine(
                        page=page_i,
                        bbox=tuple(float(value) for value in bbox),
                        text=text,
                        size=size,
                        block_bbox=block_bbox,
                        block_index=block_index,
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                    ))
        rows = _column_aware_order(rows)
        lines = [row.text for row in rows]
        text = strip_invisible_unicode("\n".join(lines))
        heading = next((ln for ln in lines if ln.strip()), "")
        name = _name_from_heading(rows, fallback)
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
        rows: list[TextLine],
        fallback: str,
) -> str:
    page0 = [row for row in rows if row.page == 0]
    if page0:
        page_height = page0[0].page_height
        candidates = [
            row for row in page0
            if row.bbox[1] <= page_height * 0.4
            and len(row.text) < 80
            and not NAME_EXCLUDED_RE.search(row.text)
            and not _looks_like_heading_label(row.text)
            and NAME_RE.fullmatch(row.text.strip())
        ]
        candidates.sort(
            key=lambda row: (-row.size, row.bbox[1], row.bbox[0])
        )
        if candidates:
            return candidates[0].text.strip()
    return fallback


def _column_aware_order(rows: list[TextLine]) -> list[TextLine]:
    ordered: list[TextLine] = []
    pages = sorted({row.page for row in rows})
    for page_index in pages:
        page_rows = [row for row in rows if row.page == page_index]
        blocks: dict[int, list[TextLine]] = {}
        for row in page_rows:
            blocks.setdefault(row.block_index, []).append(row)

        header_ids: set[int] = set()
        for block_id, block_rows in blocks.items():
            bounds = block_rows[0].block_bbox
            page_width = block_rows[0].page_width
            page_height = block_rows[0].page_height
            if (
                bounds[1] <= page_height * 0.14
            ):
                header_ids.add(block_id)

        spanning_ids = {
            block_id for block_id, block_rows in blocks.items()
            if block_id not in header_ids
            and (
                block_rows[0].block_bbox[2]
                - block_rows[0].block_bbox[0]
                >= block_rows[0].page_width * 0.72
            )
        }
        column_blocks = [
            block_id for block_id in blocks
            if block_id not in header_ids | spanning_ids
        ]
        column_blocks.sort(
            key=lambda block_id: blocks[block_id][0].block_bbox[0]
        )
        columns: list[list[int]] = []
        for block_id in column_blocks:
            x_start = blocks[block_id][0].block_bbox[0]
            page_width = blocks[block_id][0].page_width
            if (
                not columns
                or x_start - _mean_block_x(columns[-1], blocks)
                > page_width * 0.12
            ):
                columns.append([block_id])
            else:
                columns[-1].append(block_id)

        def column_priority(column: list[int]) -> tuple[float, int]:
            block_rows = [
                row for block_id in column for row in blocks[block_id]
            ]
            width = max(
                row.block_bbox[2] - row.block_bbox[0]
                for row in block_rows
            )
            text_length = sum(len(row.text) for row in block_rows)
            return width, text_length

        main_column = (
            max(range(len(columns)), key=lambda i: column_priority(columns[i]))
            if columns else -1
        )
        column_order = (
            ([main_column] if main_column >= 0 else [])
            + [i for i in range(len(columns)) if i != main_column]
        )
        for block_id in sorted(
                header_ids,
                key=lambda bid: (
                    blocks[bid][0].bbox[1], blocks[bid][0].bbox[0]
                ),
        ):
            ordered.extend(
                sorted(
                    blocks[block_id],
                    key=lambda row: (row.bbox[1], row.bbox[0]),
                )
            )
        for column_index in column_order:
            column_rows = [
                row for block_id in columns[column_index]
                for row in blocks[block_id]
            ]
            if len(columns) == 1:
                column_rows.extend(
                    row for block_id in spanning_ids
                    for row in blocks[block_id]
                )
            ordered.extend(
                sorted(column_rows, key=lambda row: (row.bbox[1], row.bbox[0]))
            )
        if len(columns) > 1:
            for block_id in sorted(
                    spanning_ids,
                    key=lambda bid: (
                        blocks[bid][0].bbox[1], blocks[bid][0].bbox[0]
                    ),
            ):
                ordered.extend(
                    sorted(
                        blocks[block_id],
                        key=lambda row: (row.bbox[1], row.bbox[0]),
                    )
                )
        elif not columns:
            for block_id in sorted(
                    spanning_ids,
                    key=lambda bid: (
                        blocks[bid][0].bbox[1], blocks[bid][0].bbox[0]
                    ),
            ):
                ordered.extend(
                    sorted(
                        blocks[block_id],
                        key=lambda row: (row.bbox[1], row.bbox[0]),
                    )
                )
    return ordered


def _mean_block_x(
        block_ids: list[int],
        blocks: dict[int, list[TextLine]],
) -> float:
    return sum(
        blocks[block_id][0].block_bbox[0] for block_id in block_ids
    ) / len(block_ids)


def _looks_like_heading_label(line: str) -> bool:
    t = line.strip().lower().rstrip(":")
    labels = {
        "experience", "work experience", "education", "skills",
        "projects", "summary", "professional summary", "objective",
        "profile", "contact", "interests", "strengths", "languages",
        "about", "about me", "certifications", "references",
    }
    return t in labels
