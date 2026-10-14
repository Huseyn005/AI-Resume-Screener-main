from __future__ import annotations
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from app.preprocess import all_chunks, scrub_pii, split_sections
from app.skills import SkillLexicon, extract_lexical, load_skills
from ml.common import DATA_DIR, write_jsonl

SEED = 42
ROLE_NAMES = {
    "ACCOUNTANT": "Accountant",
    "ADVOCATE": "Advocate",
    "AGRICULTURE": "Agricultural Specialist",
    "APPAREL": "Apparel Professional",
    "ARTS": "Arts Professional",
    "AUTOMOBILE": "Automotive Professional",
    "AVIATION": "Aviation Professional",
    "BANKING": "Banking Professional",
    "BPO": "Business Process Outsourcing Professional",
    "BUSINESS-DEVELOPMENT": "Business Development Professional",
    "CHEF": "Chef",
    "CONSTRUCTION": "Construction Professional",
    "CONSULTANT": "Consultant",
    "DESIGNER": "Designer",
    "DIGITAL-MEDIA": "Digital Media Professional",
    "ENGINEERING": "Engineer",
    "FINANCE": "Finance Professional",
    "FITNESS": "Fitness Professional",
    "HEALTHCARE": "Healthcare Professional",
    "HR": "Human Resources Professional",
    "INFORMATION-TECHNOLOGY": "Information Technology Professional",
    "PUBLIC-RELATIONS": "Public Relations Professional",
    "SALES": "Sales Professional",
    "TEACHER": "Teacher",
}
TEMPLATES_A = (
    "We are hiring a {role} to support a growing team. The successful "
    "candidate will apply {skills} to deliver reliable day-to-day results.",
    "Our organization needs a {role} who can own assigned work, collaborate "
    "across teams, and bring experience with {skills}.",
    "The {role} will contribute to active projects and ongoing operations. "
    "Candidates should be comfortable using {skills}.",
)
TEMPLATES_B = (
    "A team is seeking a {role}. This position calls for practical "
    "experience in {skills} and sound judgment when priorities change.",
    "In this {role} role, you will help improve team outcomes and take "
    "responsibility for quality. Relevant capabilities include {skills}.",
    "We need a dependable {role} to strengthen our operation. Bring "
    "working knowledge of {skills} and a collaborative approach.",
)
NOISE_SENTENCES = (
    "Clear communication and careful documentation are valued.",
    "The position works with colleagues across several teams.",
    "Successful candidates manage priorities and meet agreed deadlines.",
)


def _lexicon_entries(
        lexicon: SkillLexicon,
) -> list[tuple[str, str, re.Pattern[str]]]:
    pattern_lookup = {
        pattern.pattern: (canonical, pattern)
        for canonical, pattern in lexicon.patterns
    }
    entries: list[tuple[str, str, re.Pattern[str]]] = []
    seen: set[str] = set()
    for alias, canonical in sorted(
            lexicon.canonical.items(), key=lambda item: -len(item[0])
    ):
        if not alias or alias in seen:
            continue
        seen.add(alias)
        pattern_text = (
            rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        )
        match = pattern_lookup.get(pattern_text)
        if match is not None:
            entries.append((alias, match[0], match[1]))
    return entries


def _extract_document_skills(
        text: str,
        lexicon: SkillLexicon,
        entries: list[tuple[str, str, re.Pattern[str]]],
) -> list[str]:
    folded = text.casefold()
    filtered = [
        (canonical, pattern)
        for alias, canonical, pattern in entries
        if alias.casefold() in folded
    ]
    return extract_lexical(
        text,
        SkillLexicon(canonical=lexicon.canonical, patterns=filtered),
    )


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_resumes(path: Path = DATA_DIR / "Resume.csv") -> list[dict[str, str]]:
    frame = pd.read_csv(path, usecols=["Resume_str", "Category"])
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for raw_text, raw_category in frame.itertuples(index=False, name=None):
        if pd.isna(raw_text) or pd.isna(raw_category):
            continue
        text = str(raw_text).strip()
        category = str(raw_category).strip().upper()
        if len(text) < 200 or not category:
            continue
        key = normalized_text(text)
        if key in seen:
            continue
        seen.add(key)
        records.append({"text": scrub_pii(text), "category": category})
    if not records:
        raise ValueError(f"No usable resumes found in {path}")
    return records


def stratified_splits(
        records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]],
           list[dict[str, str]]]:
    labels = [row["category"] for row in records]
    train, remaining = train_test_split(
        records,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    val, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=SEED + 1,
        stratify=[row["category"] for row in remaining],
    )
    return list(train), list(val), list(test)


def _log_odds_rate(count: int, size: int) -> float:
    return math.log((count + 0.5) / (size - count + 0.5))


def category_profiles(
        train: list[dict[str, str]],
        lexicon: SkillLexicon,
) -> dict[str, dict[str, list[str]]]:
    categories = sorted({row["category"] for row in train})
    entries = _lexicon_entries(lexicon)
    skill_docs = [
        set(_extract_document_skills(row["text"], lexicon, entries))
        for row in train
    ]
    all_skills = sorted(set().union(*skill_docs))
    category_docs = {
        category: [
            i for i, row in enumerate(train)
            if row["category"] == category
        ]
        for category in categories
    }
    skill_profiles: dict[str, list[str]] = {}
    for category in categories:
        indexes = category_docs[category]
        other_indexes = [
            i for i, row in enumerate(train)
            if row["category"] != category
        ]
        ranked: list[tuple[float, int, str]] = []
        for skill in all_skills:
            own_df = sum(skill in skill_docs[i] for i in indexes)
            other_df = sum(skill in skill_docs[i] for i in other_indexes)
            score = (
                _log_odds_rate(own_df, len(indexes))
                - _log_odds_rate(other_df, len(other_indexes))
            )
            if own_df:
                ranked.append((score, own_df, skill))
        ranked.sort(reverse=True)
        skill_profiles[category] = [
            skill for _score, _df, skill in ranked[:40]
        ]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=2,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform([row["text"] for row in train])
    terms = vectorizer.get_feature_names_out()
    profiles: dict[str, dict[str, list[str]]] = {}
    for category in categories:
        indexes = category_docs[category]
        mean_tfidf = np.asarray(tfidf[indexes].mean(axis=0)).ravel()
        top = np.argsort(mean_tfidf)[::-1][:30]
        profiles[category] = {
            "top_skills": skill_profiles[category],
            "top_tfidf_terms": [str(terms[i]) for i in top if mean_tfidf[i]],
        }
    return profiles


def role_name(category: str) -> str:
    return ROLE_NAMES.get(
        category,
        f"{category.replace('-', ' ').replace('_', ' ').title()} Professional",
    )


def sample_skills(
        rng: np.random.Generator,
        skills: list[str],
        count: int,
) -> list[str]:
    if not skills:
        return []
    count = min(count, len(skills))
    return [
        str(skill)
        for skill in rng.choice(skills, size=count, replace=False)
    ]


def generate_jds(
        profiles: dict[str, dict[str, list[str]]],
) -> dict[str, list[dict[str, object]]]:
    rng = np.random.default_rng(SEED)
    output: dict[str, list[dict[str, object]]] = {
        "train": [], "val": [], "test": [],
    }
    n_by_split = {"train": 15, "val": 4, "test": 4}
    for split, jd_count in n_by_split.items():
        templates = TEMPLATES_A if split == "train" else TEMPLATES_B
        for category in sorted(profiles):
            profile = profiles[category]["top_skills"]
            if not profile:
                continue
            for index in range(jd_count):
                target = int(rng.integers(2, 4)) if index % 5 == 0 else int(
                    rng.integers(4, 9)
                )
                selected = sample_skills(rng, profile, target)
                skill_text = ", ".join(selected)
                role = role_name(category)
                text = templates[index % len(templates)].format(
                    role=role, skills=skill_text,
                )
                if index % 4 == 0:
                    noise = NOISE_SENTENCES[index % len(NOISE_SENTENCES)]
                    text = f"{text} {noise}"
                output[split].append({
                    "jd_id": f"{split}-{category.lower()}-{index:02d}",
                    "category": category,
                    "text": text,
                    "skills": selected,
                    "split": split,
                })
    return output


def _training_chunks(
        train: list[dict[str, str]],
        lexicon: SkillLexicon,
) -> dict[str, list[tuple[str, set[str]]]]:
    output: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    entries = _lexicon_entries(lexicon)
    for resume in train:
        sections = split_sections(resume["text"])
        chunks = all_chunks(sections, size=200, overlap=40)
        for _section, chunk in chunks:
            skills = set(_extract_document_skills(chunk, lexicon, entries))
            output[resume["category"]].append((chunk, skills))
    return output


def make_triplets(
        train: list[dict[str, str]],
        train_jds: list[dict[str, object]],
        lexicon: SkillLexicon,
) -> list[dict[str, str]]:
    chunks = _training_chunks(train, lexicon)
    categories = sorted(chunks)
    triplets: list[dict[str, str]] = []
    for jd in train_jds:
        category = str(jd["category"])
        jd_skills = set(str(skill) for skill in jd["skills"])
        if not jd_skills:
            continue
        positive_chunks = chunks.get(category, [])
        negatives = [
            chunk for other in categories if other != category
            for chunk, skills in chunks[other]
            if jd_skills.intersection(skills)
        ]
        if not positive_chunks or not negatives:
            continue
        hard_negative = max(
            negatives,
            key=lambda item: len(jd_skills.intersection(item[1])),
        )[0]
        ranked_positive = sorted(
            positive_chunks,
            key=lambda item: len(jd_skills.intersection(item[1])),
            reverse=True,
        )
        for positive, _skills in ranked_positive[:4]:
            triplets.append({
                "category": category,
                "jd": str(jd["text"]),
                "positive": positive,
                "negative": hard_negative,
            })
    return triplets


def print_counts(name: str, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["category"] for row in rows)
    print(f"{name} ({len(rows)} resumes)")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")


def main() -> None:
    records = load_resumes()
    train, val, test = stratified_splits(records)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(DATA_DIR / f"{name}.jsonl", rows)
        print_counts(name, rows)
    lexicon = load_skills()
    profiles = category_profiles(train, lexicon)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "category_profiles.json").write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    jds = generate_jds(profiles)
    for split, rows in jds.items():
        write_jsonl(DATA_DIR / f"{split}_jds.jsonl", rows)
    triplets = make_triplets(train, jds["train"], lexicon)
    write_jsonl(DATA_DIR / "triplets.jsonl", triplets)
    print(f"Generated JDs: {sum(map(len, jds.values()))}")
    print(f"Generated training triplets: {len(triplets)}")


if __name__ == "__main__":
    main()
