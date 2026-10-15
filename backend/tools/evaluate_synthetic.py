from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app import main  # noqa: E402
from app.config import Settings  # noqa: E402
from app.skill_context import GENERIC_SKILLS, is_occupation_term  # noqa: E402
from app.scoring import calibrate  # noqa: E402
from app.skills import (  # noqa: E402
    extract_lexical,
    load_skills,
    technical_skills,
)

JD_DIR = BACKEND_ROOT / "tests" / "eval" / "jds"
BASELINE_PATH = BACKEND_ROOT / "tests" / "eval" / "baseline_before.json"
ROLE_NAMES = {
    "accountant": "Accountant",
    "electrician": "Electrician",
    "head_chef": "Head Chef",
    "history_teacher": "History Teacher",
    "hr_specialist": "Human Resources Specialist",
    "java_developer": "Java Developer",
    "line_cook": "Line Cook",
    "nurse": "Registered Nurse",
    "process_automation_engineer": "Process Automation Engineer",
    "sales_rep": "Sales Representative",
    "social_media_manager": "Social Media Manager",
    "truck_driver": "Truck Driver",
}
UNRELATED_ROLE = {
    "accountant": "Graphic Designer",
    "electrician": "Registered Nurse",
    "head_chef": "Java Developer",
    "history_teacher": "Truck Driver",
    "hr_specialist": "Line Cook",
    "java_developer": "Line Cook",
    "line_cook": "Java Developer",
    "nurse": "Social Media Manager",
    "process_automation_engineer": "History Teacher",
    "sales_rep": "Electrician",
    "social_media_manager": "Accountant",
    "truck_driver": "Human Resources Specialist",
}
TOKEN_RE = re.compile(r"[a-z0-9+#]+", re.I)
BASELINE_COMMON_FALSE_POSITIVES = {
    "cargo", "coordinate", "end", "vital",
}


class FakeTokenizer:
    def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
            verbose: bool = True,
    ) -> list[int]:
        tokens = TOKEN_RE.findall(text)
        if add_special_tokens:
            return [1, *range(len(tokens)), 2]
        return list(range(len(tokens)))


class BagOfWordsModel:
    max_seq_length = 384
    tokenizer = FakeTokenizer()

    def encode(
            self,
            texts: list[str],
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % vectors.shape[1]
                vectors[row, index] += 1.0
        if normalize_embeddings:
            lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.maximum(lengths, 1e-12)
        return vectors


def candidate_profiles(
        jd_id: str,
        jd: str,
        lexicon: Any,
) -> list[dict[str, Any]]:
    found = extract_lexical(jd, lexicon)
    hard = technical_skills(found, lexicon)
    soft = [skill for skill in found if skill not in hard]
    mid_hard = hard[:max(1, (len(hard) * 2 + 2) // 3)] if hard else []
    mid_soft = soft[:max(1, len(soft) // 2)] if soft else []
    relevant = hard + soft
    weak_skills = relevant[:max(1, len(relevant) // 3)]
    role = ROLE_NAMES[jd_id]

    strong_text = (
        f"Strong Candidate {jd_id}\n{role}\n"
        f"Experience\nWorked as a {role}. "
        f"Applied {', '.join(relevant)} and delivered reliable results."
    )
    mid_skills = mid_hard + mid_soft
    mid_text = (
        f"Mid Candidate {jd_id}\n{role}\n"
        "Skills\n" + ", ".join(mid_skills) + "\n"
        "Experience\n"
        f"Supported {role} projects and applied {', '.join(mid_skills)}. "
        "Contributed to reliable daily operations and documented results."
    )
    weak_text = (
        f"Coursework Candidate {jd_id}\n{role}\nEducation\n"
        f"Relevant coursework: {', '.join(weak_skills)}. "
        "Completed classroom exercises and introductory assignments."
    )
    unrelated_text = (
        f"Unrelated Candidate {jd_id}\n{UNRELATED_ROLE[jd_id]}\n"
        "Skills\nPython, React, social media, customer service, "
        "inventory control\nExperience\n"
        "Performed unrelated duties and completed projects in a different "
        "occupation. Maintained records and supported routine operations."
    )
    return [
        {
            "id": f"{jd_id}-strong",
            "name": f"Strong {jd_id}",
            "relevance": 3,
            "text": strong_text,
        },
        {
            "id": f"{jd_id}-mid",
            "name": f"Mid {jd_id}",
            "relevance": 2,
            "text": mid_text,
        },
        {
            "id": f"{jd_id}-coursework",
            "name": f"Coursework {jd_id}",
            "relevance": 1,
            "text": weak_text,
        },
        {
            "id": f"{jd_id}-unrelated",
            "name": f"Unrelated {jd_id}",
            "relevance": 0,
            "text": unrelated_text,
        },
    ]


def cook_profiles(jd: str, lexicon: Any) -> list[dict[str, Any]]:
    skills = technical_skills(extract_lexical(jd, lexicon), lexicon)
    if not skills:
        raise ValueError("The line-cook JD has no recognized hard skills")
    count = len(skills)
    cases = [
        (0, 0, "Dishwashing and back-of-house cleaning."),
        (0, 0, "Kitchen porter duties during busy restaurant service."),
        (1, 1, "Coursework covered food service routines."),
        (1, 2, "Used foundational skills during a kitchen shift."),
        (2, 3, "Coursework included ingredient preparation."),
        (2, 4, "Applied skills during supervised kitchen service."),
        (2, 5, "Prepared ingredients on a busy kitchen line."),
        (3, min(6, count), "Worked a line station during dinner service."),
        (3, min(6, count), "Maintained quality during high-volume service."),
        (3, count, "Led reliable line service and trained junior cooks."),
    ]
    profiles = []
    for index, (relevance, skill_count, description) in enumerate(cases):
        selected = skills[:skill_count]
        if index in {2, 4, 8}:
            body = (
                "Education\nCoursework in " + ", ".join(selected)
                + ". " + description
            )
        else:
            body = (
                "Experience\n" + description + " "
                + ", ".join(selected) + "."
            )
        profiles.append({
            "id": f"cook-{index + 1:02d}",
            "name": f"Kitchen Candidate {index + 1:02d}",
            "relevance": relevance,
            "text": (
                f"Kitchen Candidate {index + 1:02d}\nLine Cook\n"
                f"{body}\n"
                "Handled kitchen tasks, communicated with coworkers, "
                "and followed service procedures."
            ),
        })
    return profiles


def pairwise_accuracy(
        scores: list[float],
        relevance: list[int],
) -> float:
    correct = 0.0
    pairs = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            if relevance[left] == relevance[right]:
                continue
            pairs += 1
            if scores[left] == scores[right]:
                correct += 0.5
            elif (
                (scores[left] > scores[right])
                == (relevance[left] > relevance[right])
            ):
                correct += 1
    return correct / pairs if pairs else 1.0


def ndcg_at_10(scores: list[float], relevance: list[int]) -> float:
    order = sorted(
        range(len(scores)), key=lambda index: (-scores[index], index)
    )[:10]
    ideal = sorted(relevance, reverse=True)[:10]
    dcg = sum(
        (2 ** relevance[index] - 1) / np.log2(rank + 2)
        for rank, index in enumerate(order)
    )
    idcg = sum(
        (2 ** value - 1) / np.log2(rank + 2)
        for rank, value in enumerate(ideal)
    )
    return float(dcg / idcg) if idcg else 0.0


def removed_baseline_false_positives(
        baseline_skills: list[str],
        current_skills: list[str],
        lexicon: Any,
) -> list[str]:
    removed = []
    for skill in baseline_skills:
        normalized = skill.casefold()
        if (
            normalized in GENERIC_SKILLS
            or normalized in BASELINE_COMMON_FALSE_POSITIVES
            or is_occupation_term(normalized, normalized)
        ):
            removed.append(skill)
            continue
        if not any(
            re.search(
                rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])",
                current_skill,
                re.I,
            )
            for current_skill in current_skills
        ):
            removed.append(skill)
    return sorted(set(removed))


def baseline_score(
        jd: str,
        profile: str,
        hard_skills: list[str],
        model: BagOfWordsModel,
        settings: Settings,
) -> float:
    jd_vector, resume_vector = model.encode([jd, profile])
    raw_semantic = float(np.dot(jd_vector, resume_vector))
    semantic = calibrate(
        raw_semantic, settings.semantic_low, settings.semantic_high
    )
    matched = [
        skill for skill in hard_skills
        if re.search(
            rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])",
            profile,
            re.I,
        )
    ]
    coverage = len(matched) / len(hard_skills) if hard_skills else 0.0
    experience = profile.split("Experience", 1)[-1]
    section_vector = model.encode([experience])[0]
    section = float(np.dot(jd_vector, section_vector))
    score = 0.5 * semantic + 0.4 * coverage + 0.1 * section
    if coverage == 0:
        score = min(score, 0.35)
    return float(round(score * 100))


async def evaluate_pipeline() -> dict[str, Any]:
    settings = Settings()
    lexicon = load_skills()
    model = BagOfWordsModel()
    previous = {
        key: getattr(main.app.state, key, None)
        for key in ("settings", "model", "calibrator", "lexicon")
    }
    main.app.state.settings = settings
    main.app.state.model = model
    main.app.state.calibrator = None
    main.app.state.lexicon = lexicon
    try:
        output: dict[str, Any] = {
            "jds": {},
            "cook_scenario": [],
            "metrics": {},
            "baseline": json.loads(
                BASELINE_PATH.read_text(encoding="utf-8")
            ),
        }
        all_before_accuracy = []
        all_after_accuracy = []
        before_ndcg = []
        after_ndcg = []
        for path in sorted(JD_DIR.glob("*.txt")):
            jd_id = path.stem
            jd = path.read_text(encoding="utf-8").strip()
            profiles = candidate_profiles(jd_id, jd, lexicon)
            before_row = next(
                row for row in output["baseline"]
                if row["jd"] == jd_id
            )
            current = extract_lexical(jd, lexicon)
            current_hard = technical_skills(current, lexicon)
            current_soft = [
                skill for skill in current if skill not in current_hard
            ]
            before_scores = [
                baseline_score(
                    jd, profile["text"], before_row["hard_skills"],
                    model, settings,
                )
                for profile in profiles
            ]
            after_scores = []
            for profile in profiles:
                result = await main._score_text(
                    f"{profile['id']}.txt",
                    profile["text"],
                    [],
                    profile["name"],
                    jd,
                    settings,
                )
                after_scores.append(result.match_percentage)
            relevance = [profile["relevance"] for profile in profiles]
            before_accuracy = pairwise_accuracy(
                before_scores, relevance
            )
            after_accuracy = pairwise_accuracy(after_scores, relevance)
            before_metric = ndcg_at_10(before_scores, relevance)
            after_metric = ndcg_at_10(after_scores, relevance)
            all_before_accuracy.append(before_accuracy)
            all_after_accuracy.append(after_accuracy)
            before_ndcg.append(before_metric)
            after_ndcg.append(after_metric)
            output["jds"][jd_id] = {
                "profiles": profiles,
                "before_scores": before_scores,
                "after_scores": after_scores,
                "before_pairwise_accuracy": before_accuracy,
                "after_pairwise_accuracy": after_accuracy,
                "before_ndcg": before_metric,
                "after_ndcg": after_metric,
                "before_hard_skills": before_row["hard_skills"],
                "before_soft_skills": before_row["soft_skills"],
                "hard_skills": current_hard,
                "soft_skills": current_soft,
                "false_positives_removed": removed_baseline_false_positives(
                    before_row["hard_skills"] + before_row["soft_skills"],
                    current_hard + current_soft,
                    lexicon,
                ),
                "hard_matched_profile_scores": after_scores[:3],
            }

        cook_jd = (JD_DIR / "line_cook.txt").read_text(
            encoding="utf-8"
        ).strip()
        for profile in cook_profiles(cook_jd, lexicon):
            result = await main._score_text(
                f"{profile['id']}.txt",
                profile["text"],
                [],
                profile["name"],
                cook_jd,
                settings,
            )
            output["cook_scenario"].append({
                **profile,
                "score": result.match_percentage,
            })
        cook_scores = [
            row["score"] for row in output["cook_scenario"]
        ]
        cook_relevance = [
            row["relevance"] for row in output["cook_scenario"]
        ]
        output["metrics"] = {
            "before_pairwise_accuracy": float(np.mean(all_before_accuracy)),
            "after_pairwise_accuracy": float(np.mean(all_after_accuracy)),
            "before_ndcg_at_10": float(np.mean(before_ndcg)),
            "after_ndcg_at_10": float(np.mean(after_ndcg)),
            "cook_pairwise_accuracy": pairwise_accuracy(
                cook_scores, cook_relevance
            ),
            "cook_ndcg_at_10": ndcg_at_10(
                cook_scores, cook_relevance
            ),
        }
        return output
    finally:
        for key, value in previous.items():
            setattr(main.app.state, key, value)


def run_evaluation() -> dict[str, Any]:
    return asyncio.run(evaluate_pipeline())


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic JDs and candidate resumes."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full synthetic evaluation records as JSON.",
    )
    args = parser.parse_args()
    report = run_evaluation()
    metrics = report["metrics"]
    print(
        "Pairwise accuracy: "
        f"before={metrics['before_pairwise_accuracy']:.3f} "
        f"after={metrics['after_pairwise_accuracy']:.3f}"
    )
    print(
        "NDCG@10: "
        f"before={metrics['before_ndcg_at_10']:.3f} "
        f"after={metrics['after_ndcg_at_10']:.3f}"
    )
    print(
        "JD | before hard/soft | after hard/soft | "
        "baseline-only false positives removed"
    )
    for jd_id, row in report["jds"].items():
        print(
            f"{jd_id} | {row['before_hard_skills']} / "
            f"{row['before_soft_skills']} | {row['hard_skills']} / "
            f"{row['soft_skills']} | {row['false_positives_removed']}"
        )
    print("HR line-cook scenario (rank, CV, relevance, score):")
    ranked = sorted(
        report["cook_scenario"],
        key=lambda row: (-row["score"], row["id"]),
    )
    for rank, row in enumerate(ranked, 1):
        print(
            f"{rank:2d}. {row['id']} "
            f"relevance={row['relevance']} score={row['score']}"
        )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main_cli()
