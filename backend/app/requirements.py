from __future__ import annotations

import re

from app.skills import SkillLexicon

TRIGGER_RE = re.compile(
    r"(?=\b(?:experience\s+with|knowledge\s+of|proficient\s+in|"
    r"certified(?:\s+in)?|must\s+have|required|ability\s+to)\b"
    r"\s*:?\s*([^.;\n]{2,120}))",
    re.I,
)
SPLIT_RE = re.compile(
    r"\s*(?:,|;|\s+\b(?:and|or)\b\s+)\s*",
    re.I,
)
PREFIX_RE = re.compile(
    r"^(?:(?:and|or)\s+|with\s+|experience\s+with\s+|knowledge\s+of\s+|"
    r"proficient\s+in\s+|certified(?:\s+in)?\s+|"
    r"must\s+have\s+|required\s+|ability\s+to\s+|"
    r"use\s+(?:a\s+|an\s+|the\s+)?|"
    r"operate\s+(?:a\s+|an\s+|the\s+)?)+",
    re.I,
)


def extract_unlisted_requirements(
        text: str,
        lexicon: SkillLexicon,
        limit: int = 12,
) -> list[str]:
    requirements: list[str] = []
    seen: set[str] = set()
    for match in TRIGGER_RE.finditer(text):
        candidates = SPLIT_RE.split(match.group(1))
        for index, candidate in enumerate(candidates):
            if len(match.group(1)) >= 120 and index == len(candidates) - 1:
                continue
            phrase = re.sub(
                r"^(?:a|an|the|at least|minimum of)\s+",
                "",
                candidate.strip(),
                flags=re.I,
            )
            previous = None
            while phrase != previous:
                previous = phrase
                phrase = PREFIX_RE.sub("", phrase).strip()
            phrase = re.sub(
                r"^(?:a|an|the)\s+", "", phrase, flags=re.I
            )
            phrase = re.sub(r"\s+", " ", phrase).strip(" :-")
            normalized = phrase.casefold()
            if (
                len(normalized) < 3
                or normalized in seen
                or normalized in lexicon.canonical
            ):
                continue
            if re.fullmatch(r"[\d\s.+-]+", phrase):
                continue
            seen.add(normalized)
            requirements.append(phrase)
            if len(requirements) >= limit:
                return requirements
    return requirements
