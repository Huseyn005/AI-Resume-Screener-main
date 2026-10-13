from __future__ import annotations

TEMPLATES_MATCH = (
    "This resume aligns most with the {section} section and "
    "covers {skills}."
)
TEMPLATES_NO_MATCH = (
    "This resume has limited overlap with the listed job skills "
    "in the {section} section."
)
TEMPLATES_GAPS = "Key gaps include: {missing}."
TEMPLATES_COVERED = "Required skills appear well covered overall."


def rank_top_skills(
        matched: list[str],
        scores: dict[str, float],
        limit: int = 3,
) -> list[str]:
    ranked = sorted(
        matched,
        key=lambda s: scores.get(s, 0.0),
        reverse=True,
    )
    return ranked[:limit]


def missing_skills(missing: list[str], limit: int = 3) -> list[str]:
    return missing[:limit]


def build_reason(
        matched: list[str],
        missing: list[str],
        best_section: str,
) -> str:
    if matched:
        skills = ", ".join(matched[:3])
        if best_section and best_section != "other":
            first = TEMPLATES_MATCH.format(
                section=best_section, skills=skills
            )
        else:
            first = f"This resume covers {skills}."
    else:
        if best_section and best_section != "other":
            first = TEMPLATES_NO_MATCH.format(section=best_section)
        else:
            first = (
                "This resume has limited overlap with the listed job skills."
            )
    if missing:
        second = TEMPLATES_GAPS.format(
            missing=", ".join(missing[:3])
        )
    else:
        second = TEMPLATES_COVERED
    return f"{first} {second}"


def strengths_and_gaps(
        matched: list[str],
        missing: list[str],
        jd_total: int,
) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    gaps: list[str] = []
    if matched:
        strengths.append(
            f"Covers {len(matched)} of {jd_total or len(matched)} "
            "required skills"
        )
    if missing:
        shown = ", ".join(missing[:3])
        extra = "…" if len(missing) > 3 else ""
        gaps.append(f"Missing {shown}{extra}")
    elif matched:
        gaps.append("No major gaps")
    return strengths, gaps
