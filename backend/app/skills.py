from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SKILLS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "skills.txt"
)
STOP = {
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "in",
    "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "we", "you", "they", "our",
    "your", "their", "i", "it",
}


@dataclass
class SkillLexicon:
    canonical: dict[str, str]
    patterns: list[tuple[str, re.Pattern[str]]]


def load_skills(path: Path | None = None) -> SkillLexicon:
    src = path or SKILLS_PATH
    canonical: dict[str, str] = {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip().lower() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        canon = parts[0]
        for alias in parts:
            if canon in STOP or alias in STOP:
                continue
            canonical[alias] = canon
            canonical[alias.replace(" ", "")] = canon
    patterns: list[tuple[str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for alias, canon in sorted(
            canonical.items(), key=lambda kv: -len(kv[0])
    ):
        if alias in seen or not alias:
            continue
        seen.add(alias)
        escaped = re.escape(alias)
        pat = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)
        patterns.append((canon, pat))
    return SkillLexicon(canonical=canonical, patterns=patterns)


def extract_lexical(text: str, lexicon: SkillLexicon) -> list[str]:
    hits: list[tuple[int, int, int, str]] = []
    for order, (canon, pat) in enumerate(lexicon.patterns):
        if canon in STOP:
            continue
        for match in pat.finditer(text):
            hits.append(
                (match.start(), -len(match.group()), order, canon)
            )
    hits.sort()
    found: list[str] = []
    seen: set[str] = set()
    for _start, _length, _order, canon in hits:
        if canon in seen:
            continue
        seen.add(canon)
        found.append(canon)
    return found


def noun_phrases(text: str) -> list[str]:
    tokens = [
        token for token in
        re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,}", text)
        if len(token) >= 2
    ]
    phrases: list[str] = []
    seen: set[str] = set()
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i].lower() in STOP:
            i += 1
            continue
        buf: list[str] = []
        j = i
        while j < n and j < i + 3 and tokens[j].lower() not in STOP:
            buf.append(tokens[j])
            phrase = " ".join(buf).lower()
            if phrase not in seen and len(phrase) >= 2:
                seen.add(phrase)
                phrases.append(phrase)
            j += 1
        i = j if j > i else i + 1
    return phrases


def semantic_skill_matches(
        unmatched: list[str],
        phrases: list[str],
        skill_emb: np.ndarray,
        phrase_emb: np.ndarray,
        threshold: float,
) -> list[tuple[str, float]]:
    if (
        not unmatched
        or not phrases
        or skill_emb.size == 0
        or phrase_emb.size == 0
    ):
        return []
    sim = cosine_similarity(skill_emb, phrase_emb)
    hits: list[tuple[str, float]] = []
    for i, skill in enumerate(unmatched):
        score = float(sim[i].max())
        if score >= threshold:
            hits.append((skill, score))
    return hits
