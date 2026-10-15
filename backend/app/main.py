from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from app.config import FINETUNED_DIR, Settings, get_settings
from app.embed import encode_texts, load_model, mean_pool, model_name
from app.explain import (
    build_reason,
    missing_skills as pick_missing,
    rank_top_skills,
    strengths_and_gaps,
)
from app.extract import extract_pdf, name_from_filename
from app.preprocess import (
    all_chunks,
    chunk_text,
    guess_role,
    scrub_pii,
    split_sections,
    years_experience,
)
from app.schemas import (
    AnalyzeResponse,
    AnalyzeResult,
    HealthResponse,
    ScoreBreakdown,
)
from app.scoring import (
    combine,
    load_calibrator,
    section_fit_score,
    skill_coverage,
    topk_mean,
)
from app.skills import (
    extract_lexical,
    load_skills,
    noun_phrases,
    semantic_skill_matches,
)

log = logging.getLogger("screener")
ALLOWED_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}
JD_MIN, JD_MAX = 50, 10000
MAX_FILES, MIN_FILES = 30, 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.model_name = model_name(settings)
    app.state.model = load_model(settings)
    app.state.calibrator = (
        load_calibrator(settings)
        if app.state.model_name == str(FINETUNED_DIR) else None
    )
    app.state.calibrator_loaded = app.state.calibrator is not None
    app.state.lexicon = load_skills()
    yield
    app.state.model = None


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Shortlist Screener", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        #allow_origins=settings.origin_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=getattr(app.state, "model_name", ""),
        calibrator_loaded=getattr(app.state, "calibrator_loaded", False),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
        job_description: str = Form(...),
        resumes: list[UploadFile] = File(...),
) -> AnalyzeResponse | JSONResponse:
    jd = (job_description or "").strip()
    if not (JD_MIN <= len(jd) <= JD_MAX):
        return JSONResponse(
            status_code=422,
            content={"detail": "job_description must be 50-10000 chars"},
        )
    files = resumes or []
    if not (MIN_FILES <= len(files) <= MAX_FILES):
        return JSONResponse(
            status_code=422,
            content={"detail": "resumes must contain 1-30 PDFs"},
        )
    results: list[AnalyzeResult] = []
    for upload in files:
        results.append(await _analyze_one(upload, jd))
    results.sort(key=lambda r: r.match_percentage, reverse=True)
    return AnalyzeResponse(results=results)


async def _analyze_one(upload: UploadFile, jd: str) -> AnalyzeResult:
    filename = upload.filename or "resume.pdf"
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    if (
        Path(filename).suffix.lower() != ".pdf"
        or (content_type and content_type not in ALLOWED_TYPES)
    ):
        raise HTTPException(
            status_code=415, detail="Only PDF resumes are supported"
        )
    settings: Settings = app.state.settings
    try:
        data = await upload.read()
        if len(data) > settings.max_pdf_bytes:
            return _stub(filename, ["too_large"], "error")
        if not _valid_pdf(upload.content_type, data):
            return _stub(filename, ["invalid_pdf"], "error")
        extracted = extract_pdf(
            data,
            filename,
            settings.min_text_chars,
            settings.hidden_font_size,
        )
        if extracted.status == "error":
            return _stub(
                filename, extracted.flags, "error",
                extracted.candidate_name,
            )
        if extracted.status == "needs_ocr":
            reason = (
                "This PDF looks scanned or has too little readable "
                "text. OCR is not enabled yet."
            )
            return _stub(
                filename, extracted.flags, "needs_ocr",
                extracted.candidate_name, reason,
            )
        return await _score_text(
            filename, extracted.text, extracted.flags,
            extracted.candidate_name, jd, settings,
        )
    except Exception:
        log.exception("resume failed filename=%s", filename)
        return _stub(filename, ["invalid_pdf"], "error")


def _valid_pdf(content_type: str | None, data: bytes) -> bool:
    if not data.startswith(b"%PDF"):
        return False
    ctype = (content_type or "").split(";")[0].strip().lower()
    if not ctype:
        return True
    return ctype in ALLOWED_TYPES


async def _score_text(
        filename: str,
        raw_text: str,
        flags: list[str],
        candidate_name: str,
        jd: str,
        settings: Settings,
) -> AnalyzeResult:
    model = app.state.model
    lexicon = app.state.lexicon
    model_limit = getattr(model, "max_seq_length", None)
    if model_limit is None:
        model_limit = settings.chunk_tokens
    chunk_size = max(
        1, min(settings.chunk_tokens, int(model_limit))
    )
    tokenizer = getattr(model, "tokenizer", None)
    token_counter = None
    if tokenizer is not None:
        def token_counter(chunk: str) -> int:
            return len(tokenizer.encode(chunk, add_special_tokens=True))

    text = scrub_pii(raw_text)
    jd_clean = scrub_pii(jd)
    sections = split_sections(text)
    chunks = all_chunks(
        sections, chunk_size, settings.chunk_overlap, token_counter
    )
    jd_chunks = chunk_text(
        jd_clean, chunk_size, settings.chunk_overlap, token_counter
    )
    if not jd_chunks:
        jd_chunks = [jd_clean] if jd_clean else []
    jd_skills = extract_lexical(jd_clean, lexicon)
    resume_skills = extract_lexical(text, lexicon)
    resume_set = set(resume_skills)
    lexical = [s for s in jd_skills if s in resume_set]
    unmatched = [s for s in jd_skills if s not in resume_set]
    phrases = noun_phrases(text)
    exp_proj = " ".join(
        p for p in (
            sections.get("experience"), sections.get("projects"),
        ) if p
    )
    exp_chunks = chunk_text(
        exp_proj, chunk_size, settings.chunk_overlap, token_counter
    ) if exp_proj else []
    resume_texts = [c[1] for c in chunks]
    batch = jd_chunks + resume_texts + exp_chunks + jd_skills + phrases
    batch = [text[:1200] for text in batch]
    emb = await encode_texts(model, batch)
    n_jd, n_res, n_exp = (
        len(jd_chunks), len(resume_texts), len(exp_chunks),
    )
    n_sk, n_ph = len(jd_skills), len(phrases)
    i = 0
    jd_emb = emb[i:i + n_jd]
    i += n_jd
    res_emb = emb[i:i + n_res]
    i += n_res
    exp_emb = emb[i:i + n_exp]
    i += n_exp
    skill_emb = emb[i:i + n_sk]
    i += n_sk
    phrase_emb = emb[i:i + n_ph]
    unmatched_idx = [jd_skills.index(s) for s in unmatched]
    un_emb = (
        skill_emb[unmatched_idx]
        if unmatched_idx and skill_emb.size else np.zeros((0, 1))
    )
    sem_hits = semantic_skill_matches(
        unmatched, phrases, un_emb, phrase_emb,
        settings.skill_cosine_threshold,
    )
    matched = list(lexical)
    for skill, _score in sem_hits:
        if skill not in matched:
            matched.append(skill)
    matched_set = set(matched)
    matched = [skill for skill in jd_skills if skill in matched_set]
    missing = [s for s in jd_skills if s not in matched_set]
    skill_scores = {s: 1.0 for s in lexical}
    for skill, score in sem_hits:
        skill_scores[skill] = max(skill_scores.get(skill, 0.0), score)
    if skill_emb.size and res_emb.size:
        sim = skill_emb @ res_emb.T
        for idx, skill in enumerate(jd_skills):
            skill_scores[skill] = max(
                skill_scores.get(skill, 0.0), float(sim[idx].max())
            )
    raw_sem = topk_mean(res_emb, jd_emb, k=3)
    skills_s = skill_coverage(len(matched), len(jd_skills))
    jd_vec = mean_pool(jd_emb)
    sec_vec = mean_pool(exp_emb)
    section_s = section_fit_score(jd_vec, sec_vec)
    parts = combine(
        raw_sem,
        skills_s,
        section_s,
        settings,
        getattr(app.state, "calibrator", None),
    )
    best_section = _best_section(chunks, res_emb, jd_emb)
    top = rank_top_skills(matched, skill_scores, 3)
    miss = pick_missing(missing, 3)
    reason = build_reason(top, miss, best_section)
    strengths, gaps = strengths_and_gaps(matched, missing, len(jd_skills))
    breakdown = ScoreBreakdown(
        semantic=parts.semantic_display,
        skills=parts.skills_display,
        section_fit=parts.section_display,
    )
    return AnalyzeResult(
        filename=filename,
        candidate_name=candidate_name or name_from_filename(filename),
        match_percentage=parts.match_percentage,
        top_skills=top,
        missing_skills=miss,
        score_breakdown=breakdown,
        reason=reason,
        flags=flags,
        status="ok",
        role=guess_role(raw_text),
        experience=years_experience(raw_text),
        skills=resume_skills,
        matched=matched,
        strengths=strengths,
        gaps=gaps,
        summary=reason,
    )


def _best_section(
        chunks: list[tuple[str, str]],
        res_emb: np.ndarray,
        jd_emb: np.ndarray,
) -> str:
    if not chunks or res_emb.size == 0 or jd_emb.size == 0:
        return "experience"
    sim = res_emb @ jd_emb.T
    best = int(sim.max(axis=1).argmax())
    return chunks[best][0]


def _stub(
        filename: str,
        flags: list[str],
        status: str,
        name: str = "",
        reason: str = "",
) -> AnalyzeResult:
    zero = ScoreBreakdown(semantic=0, skills=0, section_fit=0)
    text = reason or "This resume could not be scored."
    return AnalyzeResult(
        filename=filename,
        candidate_name=name or name_from_filename(filename),
        match_percentage=0,
        top_skills=[],
        missing_skills=[],
        score_breakdown=zero,
        reason=text,
        flags=flags,
        status=status,
        summary=text,
        gaps=["Text could not be read from this PDF"],
    )
