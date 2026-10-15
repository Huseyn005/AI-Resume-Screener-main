from __future__ import annotations
import re
import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.skill_context import (
    context_allows,
    domains_from_skills,
    is_occupation_term,
    occupation_domains,
    GENERIC_SKILLS,
)

SKILLS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "skills.txt"
)
STOP = {
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "in",
    "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "we", "you", "they", "our",
    "your", "their", "i", "it",
}
SKILL_STOPLIST = {
    "across", "strong", "plus", "ua", "good", "excellent",
    "ability", "abilities", "familiarity", "knowledge", "required",
    "preferred", "bonus", "various", "related",
}
AMBIGUOUS_SKILLS = {"phoenix", "lion", "flow", "equipment"}
SINGLE_SKILLS = {"c", "r"}
SKILLS_HEADING_RE = re.compile(
    r"^(?:(?:technical|hard|soft)(?:\s+and\s+(?:hard|soft))?\s+)?"
    r"skills$|^core\s+competenc(?:y|ies)$|^technologies$|^tech\s+stack$",
    re.I,
)
OTHER_HEADING_RE = re.compile(
    r"^(?:work\s+)?experience$|^employment$|^professional\s+experience$|"
    r"^work\s+history$|^internships?$|^internship\s+experience$|"
    r"^education$|^training$|^courses$|^certifications?$|^academic$|"
    r"^qualifications$|^degrees?$|^(?:personal\s+|selected\s+|key\s+)?"
    r"projects$|^summary$|^professional\s+summary$|^about\s+me$|"
    r"^objective$|^profile$|^volunteering$|^volunteer\s+experience$|"
    r"^volunteer\s+work$|^organizational\s+involvement$|"
    r"^other\s+experiences$|^achievements$|^publications$|"
    r"^community(?:\s+involvement)?$",
    re.I,
)
QUALIFIED_SINGLE_RE = {
    "c": re.compile(r"\bc\s*/\s*c\+\+|\bc\s+programming\b", re.I),
    "r": re.compile(r"\br\s+language\b", re.I),
}


@dataclass(eq=False)
class SkillLexicon:
    canonical: dict[str, str]
    patterns: list[tuple[str, re.Pattern[str]]]
    skill_types: dict[str, str]
    skill_domains: dict[str, tuple[str, ...]] = field(default_factory=dict)
    specificity: dict[str, int] = field(default_factory=dict)
    ambiguity_rules: dict[str, str] = field(default_factory=dict)
    combined_pattern: re.Pattern[str] | None = None
    aliases_by_canonical: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )


@lru_cache(maxsize=8)
def load_skills(path: Path | None = None) -> SkillLexicon:
    src = path or SKILLS_PATH
    canonical: dict[str, str] = {}
    skill_types: dict[str, str] = {}
    skill_domains: dict[str, tuple[str, ...]] = {}
    data_files = [src]
    for domain_file in sorted((src.parent / "skills").glob("*.txt")):
        data_files.append(domain_file)
    for data_file in data_files:
        for raw in data_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.isdigit():
                continue
            parts = [part.strip().lower() for part in line.split("|")]
            if not parts or not parts[0]:
                continue
            canon = parts[0]
            markers = [part for part in parts[1:] if part in {"hard", "soft"}]
            skill_type = markers[0] if markers else "hard"
            aliases = [
                part for part in parts
                if (
                    part
                    and part not in {"hard", "soft"}
                    and not part.isdigit()
                    and part not in AMBIGUOUS_SKILLS
                    and part not in SKILL_STOPLIST
                )
            ]
            if not aliases or canon in SKILL_STOPLIST:
                continue
            skill_types.setdefault(canon, skill_type)
            skill_domains.setdefault(canon, ("general",))
            for alias in aliases:
                if canon in STOP or alias in STOP:
                    continue
                canonical.setdefault(alias, canon)
                canonical.setdefault(alias.replace(" ", ""), canon)

    meta_path = src.parent / "skills_meta.tsv"
    specificity: dict[str, int] = {}
    ambiguity_rules: dict[str, str] = {}
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                canon = row.get("canonical", "").strip().casefold()
                if not canon:
                    continue
                skill_type = row.get("type", "").strip().casefold()
                if skill_type in {"hard", "soft"}:
                    skill_types[canon] = skill_type
                domains = tuple(
                    value.strip().casefold()
                    for value in row.get("domains", "").split(",")
                    if value.strip()
                )
                if domains:
                    skill_domains[canon] = domains
                value = row.get("specificity", "").strip()
                if value.isdigit():
                    specificity[canon] = min(3, max(1, int(value)))
                rule = row.get("ambiguity_rule", "").strip()
                if rule:
                    ambiguity_rules[canon] = rule
    for alias, canon in (
        ("opc-ua", "opc ua"),
        ("plcscada", "plcscada"),
    ):
        canonical.setdefault(alias, canon)
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
    alternatives = "|".join(
        re.escape(alias)
        for alias in sorted(canonical, key=len, reverse=True)
        if alias
    )
    combined_pattern = (
        re.compile(
            rf"(?<![a-z0-9])(?P<alias>{alternatives})(?![a-z0-9])",
            re.I,
        )
        if alternatives else None
    )
    aliases_by_canonical_mutable: dict[str, list[str]] = {}
    for alias, canon in canonical.items():
        aliases_by_canonical_mutable.setdefault(canon, []).append(alias)
    aliases_by_canonical = {
        canon: tuple(aliases)
        for canon, aliases in aliases_by_canonical_mutable.items()
    }
    return SkillLexicon(
        canonical=canonical,
        patterns=patterns,
        skill_types=skill_types,
        skill_domains=skill_domains,
        specificity=specificity,
        ambiguity_rules=ambiguity_rules,
        combined_pattern=combined_pattern,
        aliases_by_canonical=aliases_by_canonical,
    )


def extract_lexical(text: str, lexicon: SkillLexicon) -> list[str]:
    if len(text) > 50000:
        return _extract_lexical(text, lexicon)
    return list(_extract_lexical_cached(text, lexicon))


@lru_cache(maxsize=512)
def _extract_lexical_cached(
        text: str,
        lexicon: SkillLexicon,
) -> tuple[str, ...]:
    return tuple(_extract_lexical(text, lexicon))


def _extract_lexical(text: str, lexicon: SkillLexicon) -> list[str]:
    hits: list[tuple[int, int, int, str, str]] = []
    skill_ranges = _skills_ranges(text)
    if lexicon.combined_pattern:
        for order, match in enumerate(
                lexicon.combined_pattern.finditer(text)
        ):
            alias = match.group("alias").casefold()
            canon = lexicon.canonical[alias]
            if canon == "plcscada":
                hits.extend((
                    (
                        match.start(), match.start() + 3, order,
                        "plc", match.group()[:3],
                    ),
                    (
                        match.start() + 3, match.end(), order,
                        "scada", match.group()[3:],
                    ),
                ))
                continue
            hits.append(
                (
                    match.start(), match.end(), order, canon,
                    match.group(),
                )
            )
    else:
        for order, (canon, pat) in enumerate(lexicon.patterns):
            if canon in STOP or canon in SKILL_STOPLIST:
                continue
            for match in pat.finditer(text):
                if canon == "plcscada":
                    hits.extend((
                        (
                            match.start(), match.start() + 3, order,
                            "plc", match.group()[:3],
                        ),
                        (
                            match.start() + 3, match.end(), order,
                            "scada", match.group()[3:],
                        ),
                    ))
                    continue
                hits.append(
                    (
                        match.start(), match.end(), order, canon,
                        match.group(),
                    )
                )
    domains = occupation_domains(text)
    for _start, _end, _order, canon, _alias in hits:
        if (
            lexicon.skill_types.get(canon) == "soft"
            or canon in GENERIC_SKILLS
            or canon in lexicon.ambiguity_rules
        ):
            continue
        domains.update(lexicon.skill_domains.get(canon, ("general",)))
    domains = domains or {"general"}
    filtered = []
    for hit in hits:
        start, end, _order, canon, alias = hit
        if canon == "chef":
            if not context_allows(
                    canon, text, start, end, domains, skill_ranges, alias
            ):
                continue
        elif is_occupation_term(canon, alias):
            continue
        if canon in GENERIC_SKILLS:
            continue
        if not context_allows(
                canon, text, start, end, domains, skill_ranges, alias
        ):
            continue
        filtered.append(hit)
    hits = filtered
    hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0]), hit[2]))
    selected: list[tuple[int, int, int, str, str]] = []
    for hit in hits:
        start, end, _order, _canon, _alias = hit
        overlaps = [
            prior for prior in selected
            if start < prior[1] and prior[0] < end
        ]
        if not overlaps:
            selected.append(hit)
            continue
        if any(prior[1] - prior[0] >= end - start for prior in overlaps):
            continue
        selected = [prior for prior in selected if prior not in overlaps]
        selected.append(hit)
    selected.sort(key=lambda hit: (hit[0], hit[1]))
    found: list[str] = []
    seen: set[str] = set()
    for _start, _end, _order, canon, _alias in selected:
        if canon == "plcscada":
            continue
        if canon in seen:
            continue
        seen.add(canon)
        found.append(canon)
    return found


def technical_skills(
        skills: list[str],
        lexicon: SkillLexicon,
) -> list[str]:
    return [
        skill for skill in skills
        if lexicon.skill_types.get(skill, "hard") == "hard"
    ]


def skill_weights(text: str, lexicon: SkillLexicon) -> dict[str, float]:
    weights = {skill: 1.0 for skill in technical_skills(
        extract_lexical(text, lexicon), lexicon
    )}
    optional = re.compile(
        r"\b(?:plus|preferred|nice[\s-]+to[\s-]+have|bonus|familiarity)\b",
        re.I,
    )
    for skill in weights:
        for alias in lexicon.aliases_by_canonical.get(skill, ()):
            pattern = re.compile(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                re.I,
            )
            for match in pattern.finditer(text):
                left = max(
                    text.rfind(mark, 0, match.start())
                    for mark in (".", "!", "?", ";", "\n")
                )
                ends = [
                    index for mark in (".", "!", "?", ";", "\n")
                    if (index := text.find(mark, match.end())) >= 0
                ]
                right = min(ends) if ends else len(text)
                if optional.search(text[left + 1:right]):
                    weights[skill] = 0.5
                    break
            if weights[skill] == 0.5:
                break
    return weights


def _skills_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        heading = line.strip().rstrip(":").strip()
        inline = re.match(
            r"^\s*(?:technical\s+)?skills\s*:\s*(.*)$", line, re.I
        )
        if SKILLS_HEADING_RE.fullmatch(heading):
            if current_start is not None:
                ranges.append((current_start, offset))
            current_start = offset + len(line) - len(line.lstrip())
        elif inline:
            if current_start is not None:
                ranges.append((current_start, offset))
            current_start = offset + inline.start(1)
        elif OTHER_HEADING_RE.fullmatch(heading):
            if current_start is not None:
                ranges.append((current_start, offset))
                current_start = None
        offset += len(raw)
    if current_start is not None:
        ranges.append((current_start, len(text)))
    return ranges


def _single_skill_allowed(
        skill: str,
        position: int,
        text: str,
        skill_ranges: list[tuple[int, int]],
) -> bool:
    if any(start <= position < end for start, end in skill_ranges):
        return True
    return any(
        match.start() <= position < match.end()
        for match in QUALIFIED_SINGLE_RE[skill].finditer(text)
    )


def noun_phrases(
        text: str,
        max_phrases: int = 256,
        max_phrase_chars: int = 128,
) -> list[str]:
    if max_phrases <= 0 or max_phrase_chars < 2:
        return []
    tokens: list[str] = []
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9+#.]{1,}", text):
        token = match.group()
        if len(token) <= 64:
            tokens.append(token)
        if len(tokens) >= max_phrases:
            break
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
            if (
                phrase not in seen
                and len(phrase) >= 2
                and len(phrase) <= max_phrase_chars
            ):
                seen.add(phrase)
                phrases.append(phrase)
                if len(phrases) >= max_phrases:
                    return phrases
            j += 1
        i = j if j > i else i + 1
    return phrases


def semantic_skill_matches(
        unmatched: list[str],
        phrases: list[str],
        skill_emb: np.ndarray,
        phrase_emb: np.ndarray,
        threshold: float,
        lexicon: SkillLexicon,
) -> list[tuple[str, float]]:
    if (
        not unmatched
        or not phrases
        or skill_emb.size == 0
        or phrase_emb.size == 0
    ):
        return []
    eligible = [
        index for index, phrase in enumerate(phrases)
        if extract_lexical(phrase, lexicon)
    ]
    if not eligible:
        return []
    sim = cosine_similarity(skill_emb, phrase_emb[eligible])
    hits: list[tuple[str, float]] = []
    for i, skill in enumerate(unmatched):
        score = float(sim[i].max())
        if score >= threshold:
            hits.append((skill, score))
    return hits
