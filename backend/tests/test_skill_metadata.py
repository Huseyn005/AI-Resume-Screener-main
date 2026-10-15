from app.skill_context import domains_from_skills
from app.requirements import extract_unlisted_requirements
from app.skills import extract_lexical, load_skills, technical_skills


def test_metadata_sidecar_assigns_soft_types_and_domains() -> None:
    lexicon = load_skills()

    assert lexicon.skill_types["communication"] == "soft"
    assert lexicon.skill_types["time management"] == "soft"
    assert lexicon.skill_domains["react"] == ("software",)
    assert lexicon.ambiguity_rules["chef"] == "devops_context"


def test_legacy_soft_marker_is_not_an_alias() -> None:
    lexicon = load_skills()

    assert "soft" not in lexicon.canonical
    found = extract_lexical("Communication and teamwork", lexicon)
    assert found == ["communication", "teamwork"]
    assert technical_skills(found, lexicon) == []


def test_domain_lexicons_are_precise_for_cook_and_teacher() -> None:
    lexicon = load_skills()
    cook = (
        "Line Cook: use knife skills, follow HACCP and food safety, "
        "maintain mise en place."
    )
    teacher = (
        "History Teacher: plan lessons, design curriculum, manage a "
        "classroom, and use formative assessment."
    )
    cook_skills = extract_lexical(cook, lexicon)
    teacher_skills = extract_lexical(teacher, lexicon)

    assert {"knife skills", "haccp", "food safety"}.issubset(cook_skills)
    assert {"lesson planning", "curriculum design",
            "classroom management"}.issubset(teacher_skills)
    assert not any(
        skill in {"python", "javascript", "java"}
        for skill in cook_skills + teacher_skills
    )
    assert "culinary" in domains_from_skills(cook, cook_skills, lexicon)
    assert "teaching" in domains_from_skills(
        teacher, teacher_skills, lexicon
    )


def test_ambiguous_skills_require_appropriate_context() -> None:
    lexicon = load_skills()

    assert extract_lexical("Head Chef", lexicon) == []
    assert extract_lexical("go-getter", lexicon) == []
    assert extract_lexical("R&D", lexicon) == []
    assert extract_lexical("vitamin C", lexicon) == []
    assert extract_lexical("grade C and C-level", lexicon) == []
    assert extract_lexical("Excel spreadsheets", lexicon) == ["excel"]
    assert "excel" not in extract_lexical(
        "excel at communication", lexicon
    )
    assert extract_lexical("Microsoft Word", lexicon) == [
        "microsoft word"
    ]
    assert "microsoft word" not in extract_lexical(
        "word salad", lexicon
    )
    assert extract_lexical(
        "PowerPoint presentations", lexicon
    ) == ["microsoft powerpoint"]
    assert extract_lexical(
        "stainless steel equipment", lexicon
    ) == []
    assert extract_lexical("Go programming", lexicon) == ["go"]
    assert extract_lexical("Build REST APIs", lexicon) == ["rest"]


def test_unlisted_requirement_phrases_exclude_known_skills() -> None:
    lexicon = load_skills()
    phrases = extract_unlisted_requirements(
        "Certified in HACCP, with experience with atypical kiln calibration, "
        "knowledge of a regulated thermal sensor, and ability to use a "
        "proprietary welding fixture.",
        lexicon,
    )

    assert "atypical kiln calibration" in phrases
    assert "regulated thermal sensor" in phrases
    assert "proprietary welding fixture" in phrases
    assert not any("haccp" in phrase for phrase in phrases)
