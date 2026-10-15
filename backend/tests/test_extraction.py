from __future__ import annotations

import pymupdf

from app.extract import NAME_RE, extract_pdf, name_from_filename
from app.preprocess import split_sections


def make_background_pdf(background: tuple[float, float, float]) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=400, height=200)
    page.draw_rect(
        pymupdf.Rect(20, 20, 380, 90),
        color=background,
        fill=background,
    )
    page.insert_text(
        pymupdf.Point(35, 65),
        "Visible White Name",
        fontsize=24,
        color=(1, 1, 1),
    )
    data = document.tobytes()
    document.close()
    return data


def test_white_text_on_dark_banner_is_visible() -> None:
    result = extract_pdf(
        make_background_pdf((0.2, 0.1, 0.35)),
        min_chars=1,
    )

    assert "Visible White Name" in result.text
    assert "hidden_text" not in result.flags


def test_white_text_on_white_background_is_hidden() -> None:
    result = extract_pdf(make_background_pdf((1, 1, 1)), min_chars=1)

    assert "Visible White Name" not in result.text
    assert "hidden_text" in result.flags


def test_off_page_text_is_hidden() -> None:
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.insert_text(pymupdf.Point(190, 80), "Off page text", fontsize=12)
    data = document.tobytes()
    document.close()

    result = extract_pdf(data, min_chars=1)

    assert "Off page text" not in result.text
    assert "hidden_text" in result.flags


def test_name_detection_supports_uppercase_and_azerbaijani_letters() -> None:
    assert NAME_RE.fullmatch("NOAH WILLIAMS")
    assert NAME_RE.fullmatch("Hüseyn Məmmədli")
    assert NAME_RE.fullmatch("Rəna Rəsulzadə")


def test_name_detection_prefers_name_over_banner_title() -> None:
    document = pymupdf.open()
    page = document.new_page(width=400, height=200)
    page.draw_rect(
        pymupdf.Rect(0, 0, 400, 75),
        color=(0.2, 0.1, 0.35),
        fill=(0.2, 0.1, 0.35),
    )
    page.insert_text(
        pymupdf.Point(25, 30),
        "Lead Process Engineer",
        fontsize=14,
        color=(1, 1, 1),
    )
    page.insert_text(
        pymupdf.Point(25, 60),
        "NOAH WILLIAMS",
        fontsize=24,
        color=(1, 1, 1),
    )
    data = document.tobytes()
    document.close()

    result = extract_pdf(data, min_chars=1)

    assert result.candidate_name == "NOAH WILLIAMS"


def test_filename_fallback_strips_site_title_and_numeric_ids() -> None:
    assert name_from_filename(
        "Upstaff_Java_Software_Engineer_Vladyslav_A._100-136-111.pdf"
    ) == "Vladyslav A."
    assert name_from_filename("automation-engineer-1559104704.pdf") == (
        "Unnamed candidate"
    )
    assert name_from_filename("123456_resume.pdf") == "Unnamed candidate"


def test_two_column_text_reads_main_content_before_sidebar() -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(
        pymupdf.Rect(0, 0, 600, 70),
        color=(0.2, 0.1, 0.35),
        fill=(0.2, 0.1, 0.35),
    )
    page.insert_text(
        pymupdf.Point(25, 45),
        "NOAH WILLIAMS",
        fontsize=22,
        color=(1, 1, 1),
    )
    page.insert_text(pymupdf.Point(220, 125), "EXPERIENCE", fontsize=12)
    page.insert_text(
        pymupdf.Point(220, 150),
        "PLC and SCADA control systems engineering",
        fontsize=11,
    )
    page.insert_text(
        pymupdf.Point(220, 175),
        "DCS, HMI, Modbus and PID process controls",
        fontsize=11,
    )
    page.insert_text(pymupdf.Point(25, 125), "SKILLS", fontsize=12)
    page.insert_text(pymupdf.Point(25, 150), "Automation tools", fontsize=11)
    page.insert_text(pymupdf.Point(25, 175), "MATLAB", fontsize=11)
    data = document.tobytes()
    document.close()

    result = extract_pdf(data, min_chars=1)
    sections = split_sections(result.text)

    assert result.text.index("EXPERIENCE") < result.text.index("SKILLS")
    assert "PLC and SCADA" in sections["experience"]
    assert "Automation tools" not in sections["experience"]
    assert "Automation tools" in sections["skills"]
