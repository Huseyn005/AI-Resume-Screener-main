from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

OCCUPATION_TITLES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "occupation_titles.txt"
)
GENERIC_SKILLS = {
    "schedule", "inventory", "safety", "budgeting", "planning",
    "assessment", "station", "supervisor", "instructor", "equipment",
    "product",
}
ROLE_DOMAINS = {
    "accountant": "finance",
    "chef": "culinary",
    "cook": "culinary",
    "electrician": "trades",
    "history teacher": "teaching",
    "hr specialist": "hr",
    "line cook": "culinary",
    "nurse": "healthcare",
    "sales representative": "sales_marketing",
    "social media manager": "sales_marketing",
    "teacher": "teaching",
    "truck driver": "logistics",
}
ROLE_WORDS = {
    "accountant", "administrator", "chef", "cook", "driver",
    "electrician", "foreman", "instructor", "nurse", "supervisor",
    "teacher",
}
CONTEXT_WORDS = {
    "programming": {
        "api", "backend", "code", "coding", "compiler", "developer",
        "frontend", "framework", "golang", "language", "programming",
        "software", "web",
    },
    "spreadsheet": {
        "accounting", "cell", "finance", "financial", "formula",
        "formulas", "pivot", "quickbooks", "spreadsheet", "spreadsheets",
        "workbook", "worksheet",
    },
    "data": {
        "analytics", "big data", "data", "dataframe", "etl", "hadoop",
        "lakehouse", "pipeline", "spark sql", "warehouse",
    },
}


def _normalize_title(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^a-z0-9+#]+", " ", value)
    return " ".join(value.split())


@lru_cache(maxsize=1)
def occupation_data() -> tuple[dict[str, frozenset[str]], set[str]]:
    domains: dict[str, set[str]] = {}
    role_words = set(ROLE_WORDS)
    if OCCUPATION_TITLES_PATH.is_file():
        for raw in OCCUPATION_TITLES_PATH.read_text(
                encoding="utf-8"
        ).splitlines():
            title, _, domain_text = raw.partition("|")
            title = _normalize_title(title)
            if not title:
                continue
            title_domains = {
                value.strip().casefold()
                for value in domain_text.split(",")
                if value.strip()
            }
            domains.setdefault(title, set()).update(title_domains)
            if len(title.split()) == 1:
                role_words.add(title)
    for title, domain in ROLE_DOMAINS.items():
        domains.setdefault(title, set()).add(domain)
    return (
        {key: frozenset(value) for key, value in domains.items()},
        role_words,
    )


def _word_ngrams(text: str, maximum: int = 8) -> list[str]:
    tokens = re.findall(r"[a-z0-9+#]+", text.casefold())
    phrases = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + maximum) + 1):
            phrase = " ".join(tokens[start:end])
            phrases.append(phrase)
            if phrase.endswith("ies"):
                phrases.append(phrase[:-3] + "y")
            elif phrase.endswith("s") and not phrase.endswith("ss"):
                phrases.append(phrase[:-1])
    return phrases


def occupation_domains(text: str) -> set[str]:
    title_domains, _ = occupation_data()
    found: set[str] = set()
    for phrase in _word_ngrams(text):
        found.update(title_domains.get(phrase, ()))
    return found


def is_occupation_term(canonical: str, alias: str) -> bool:
    title_domains, role_words = occupation_data()
    canon = _normalize_title(canonical)
    form = _normalize_title(alias)
    return (
        canon in title_domains
        or form in title_domains
        or canon in role_words
        or form in role_words
    )


def domains_from_skills(
        text: str,
        skills: list[str],
        lexicon: object,
) -> set[str]:
    domains = occupation_domains(text)
    skill_domains = getattr(lexicon, "skill_domains")
    skill_types = getattr(lexicon, "skill_types")
    for skill in skills:
        if skill_types.get(skill) == "soft":
            continue
        domains.update(skill_domains.get(skill, ("general",)))
    return domains or {"general"}


def domain_allows(skill: str, domains: set[str], lexicon: object) -> bool:
    skill_domains = getattr(lexicon, "skill_domains")
    assigned = set(skill_domains.get(skill, ("general",)))
    return "general" in assigned or bool(assigned.intersection(domains))


def context_allows(
        skill: str,
        text: str,
        start: int,
        end: int,
        domains: set[str],
        skill_ranges: list[tuple[int, int]],
        alias: str,
) -> bool:
    lower = text.casefold()
    context = lower[max(0, start - 80):min(len(text), end + 80)]
    in_skill_list = any(
        left <= start < right for left, right in skill_ranges
    )
    if skill in {"c", "r"} and re.search(
            r"\b(?:vitamin\s+c|grade\s+c|c-level)\b", context
    ):
        return False
    if skill == "r" and re.search(r"\br\s*&\s*d\b", context):
        return False
    if skill == "go" and "go-getter" in context:
        return False

    if skill == "chef":
        return bool(re.search(
            r"\b(?:ansible|puppet|cookbook|infrastructure|ci\s*/\s*cd)\b",
            context,
        ))
    if (
        skill == "lesson planning"
        and alias.casefold() in {"lesson", "lessons"}
    ) or (
        skill == "curriculum design"
        and alias.casefold() == "curriculum"
    ):
        return bool(
            domains.intersection({"teaching"})
            or re.search(
                r"\b(?:teacher|classroom|curriculum|instruction)\b",
                context,
            )
        )
    if skill == "plc":
        matched = text[start:end]
        if "programmable logic controller" in alias.casefold():
            return True
        if matched.isupper() or matched.casefold() in {"plc", "plcs"}:
            return True
        return bool(
            domains.intersection({"engineering", "trades"})
            or re.search(
                r"\b(?:industrial|automation|controller|scada)\b", context
            )
        )
    rule = {
        "react": "programming",
        "rust": "programming",
        "swift": "programming",
        "spring": "programming",
    }.get(skill)
    if rule == "programming":
        return bool(CONTEXT_WORDS["programming"].intersection(
            re.findall(r"[a-z0-9+#]+", context)
        ))
    if skill == "excel":
        return bool(CONTEXT_WORDS["spreadsheet"].intersection(
            re.findall(r"[a-z0-9+#]+", context)
        ))
    if skill == "microsoft word" and alias.casefold() == "word":
        return bool(re.search(
            r"\b(?:microsoft|ms|document|word processing|word processor|"
            r"office suite)\b",
            context,
        ))
    if (
        skill == "microsoft powerpoint"
        and alias.casefold() == "powerpoint"
    ):
        return bool(re.search(
            r"\b(?:microsoft|ms|presentation|presentations|slide|slides|"
            r"office suite)\b",
            context,
        ))
    if skill == "go":
        return bool(CONTEXT_WORDS["programming"].intersection(
            re.findall(r"[a-z0-9+#]+", context)
        ))
    if skill == "r":
        if in_skill_list:
            return True
        return bool(re.search(
            r"\b(?:statistic|statistics|statistical|data science|"
            r"regression|rstudio|programming|r language)\b",
            context,
        ))
    if skill == "c":
        if in_skill_list:
            return True
        return bool(re.search(
            r"\b(?:programming|language|compiler|c\+\+|embedded|"
            r"software|developer)\b",
            context,
        ))
    if skill in {"c#", "c++"}:
        return bool(CONTEXT_WORDS["programming"].intersection(
            re.findall(r"[a-z0-9+#]+", context)
        ))
    if skill == "js":
        return in_skill_list or bool(re.search(
            r"\b(?:experience with|javascript|web development)\b",
            context,
        ))
    if alias.casefold() == "js":
        return in_skill_list or bool(re.search(
            r"\b(?:experience with|javascript|web development)\b",
            context,
        ))
    if skill == "rest":
        return bool(
            re.search(
                r"\b(?:apis?|web services?|restful)\b",
                f"{context} {alias.casefold()}",
            )
        )
    if skill in {"spark", "hive"}:
        return bool(CONTEXT_WORDS["data"].intersection(
            re.findall(r"[a-z0-9+#]+", context)
        ))
    if skill == "sap":
        return bool(re.search(
            r"\b(?:erp|s/4hana|business suite|manufacturing)\b", context
        ))
    if skill == "oracle":
        return bool(re.search(
            r"\b(?:database|sql|pl/sql|oracle db|oracle cloud)\b", context
        ))
    if skill == "iot":
        return bool(re.search(
            r"\b(?:internet of things|embedded|sensors|industrial)\b",
            context,
        ))
    if skill == "seo":
        return bool(re.search(
            r"\b(?:search engine|digital marketing|marketing|website)\b",
            context,
        ))
    if len(alias) <= 2:
        return False
    return True
