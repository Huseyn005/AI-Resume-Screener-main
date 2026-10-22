from __future__ import annotations
import asyncio
import hashlib
import logging
import math
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
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
    strip_invisible_unicode,
)
from app.schemas import (
    AnalyzeResponse,
    AnalyzeResult,
    HealthResponse,
    ScoreBreakdown,
)
from app.scoring import (
    combine_with_signal,
    evidence_weighted_coverage,
    load_calibrator,
    section_fit_score,
    topk_mean,
)
from app.skills import (
    extract_lexical,
    load_skills,
    noun_phrases,
    semantic_skill_matches,
    skill_weights,
    technical_skills,
)
from app.requirements import extract_unlisted_requirements
from app.skill_context import domain_allows, domains_from_skills

log = logging.getLogger("screener")
ALLOWED_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}
JD_MIN, JD_MAX = 50, 10000
MAX_FILES, MIN_FILES = 30, 1
RATE_WINDOW_SECONDS = 60
MAX_TRACKED_CLIENTS = 10_000


@dataclass
class ScorePreparation:
    chunks: list[tuple[str, str]]
    jd_chunks: list[str]
    resume_texts: list[str]
    exp_chunks: list[str]
    jd_skills: list[str]
    phrases: list[str]
    unlisted_requirements: list[str]
    semantic_unmatched: list[str]
    lexical: list[str]
    resume_skills: list[str]
    sections: dict[str, str]
    jd_skill_weights: dict[str, float]
    jd_soft_skills: list[str]
    hard_skill_count: int
    batch: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.analysis_semaphore = asyncio.Semaphore(
        settings.max_concurrent_analyses
    )
    app.state.rate_limit_hits = {}
    app.state.rate_limit_lock = asyncio.Lock()
    app.state.model_name = model_name(settings)
    app.state.model_source = settings.model_source()
    app.state.model = load_model(settings)
    app.state.calibrator = (
        load_calibrator(settings)
        if app.state.model_name == str(FINETUNED_DIR) else None
    )
    app.state.calibrator_loaded = app.state.calibrator is not None
    app.state.lexicon = load_skills()
    log.info(
        "model startup source=%s name=%s calibrator_loaded=%s",
        settings.model_source(),
        app.state.model_name,
        "yes" if app.state.calibrator_loaded else "no",
    )
    if not app.state.calibrator_loaded:
        log.warning("semantic calibrator is off")
    yield
    app.state.model = None


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    log.setLevel(settings.log_level.upper())
    app = FastAPI(title="Shortlist Screener", lifespan=lifespan)

    @app.middleware("http")
    async def reject_oversized_request(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/analyze":
            active_settings = (
                getattr(request.app.state, "settings", None)
                or get_settings()
            )
            if active_settings.log_level.upper() == "DEBUG":
                client, forwarded_count = _resolved_client(
                    request, active_settings
                )
                client_hash = hashlib.sha256(
                    client.encode("utf-8", errors="replace")
                ).hexdigest()[:10]
                log.debug(
                    "analyze client forwarded_for_entries=%d "
                    "client_hash=%s",
                    forwarded_count,
                    client_hash,
                )
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > active_settings.max_request_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": "Request exceeds the maximum size"
                            },
                        )
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length"},
                    )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list(),
        allow_origin_regex=settings.allowed_origin_regex or None,
        allow_credentials=False,
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
        model_source=getattr(
            app.state, "model_source", get_settings().model_source()
        ),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
        request: Request,
        job_description: str = Form(...),
        resumes: list[UploadFile] = File(...),
        debug: bool = Query(False),
) -> AnalyzeResponse | JSONResponse:
    settings = getattr(app.state, "settings", None) or get_settings()
    retry_after = await _rate_limit_retry_after(request, settings)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"detail": "Analysis rate limit exceeded"},
        )
    if debug and not settings.debug_endpoints:
        raise HTTPException(
            status_code=403,
            detail="Score explanations are disabled",
        )
    jd = (job_description or "").strip()
    if not (JD_MIN <= len(jd) <= JD_MAX):
        return JSONResponse(
            status_code=422,
            content={"detail": "job_description must be 50-10000 chars"},
        )
    files = resumes or []
    if not (MIN_FILES <= len(files) <= settings.max_files):
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"resumes must contain 1-{settings.max_files} PDFs"
            },
        )
    semaphore = getattr(app.state, "analysis_semaphore", None)
    if semaphore is None:
        semaphore = asyncio.Semaphore(settings.max_concurrent_analyses)
        app.state.analysis_semaphore = semaphore
    try:
        await asyncio.wait_for(
            semaphore.acquire(), timeout=settings.queue_timeout_seconds
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "1"},
            content={"detail": "Analysis queue is full; retry shortly"},
        )
    results: list[AnalyzeResult] = []
    explanations: list[dict[str, Any]] = []
    byte_budget = [settings.max_request_bytes]
    try:
        for index, upload in enumerate(files):
            if debug:
                result, detail = await _analyze_one(
                    upload, jd, explain=True,
                    file_index=index, byte_budget=byte_budget,
                )
                results.append(result)
                explanations.append(detail)
            else:
                results.append(await _analyze_one(
                    upload, jd, file_index=index, byte_budget=byte_budget,
                ))
    except HTTPException:
        raise
    finally:
        semaphore.release()
    order = sorted(
        range(len(results)),
        key=lambda index: results[index].match_percentage,
        reverse=True,
    )
    results = [results[index] for index in order]
    if debug:
        explanations = [explanations[index] for index in order]
        return JSONResponse(content={
            "results": [result.model_dump() for result in results],
            "debug": explanations,
        })
    return AnalyzeResponse(results=results)


async def _analyze_one(
        upload: UploadFile,
        jd: str,
        explain: bool = False,
        file_index: int = 0,
        byte_budget: list[int] | None = None,
) -> AnalyzeResult | tuple[AnalyzeResult, dict[str, Any]]:
    filename = upload.filename or "resume.pdf"
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    unsupported = (
        Path(filename).suffix.lower() != ".pdf"
        or (content_type and content_type not in ALLOWED_TYPES)
    )
    settings: Settings = app.state.settings
    try:
        budget = byte_budget or [settings.max_request_bytes]
        data = await _read_upload(upload, settings, budget)
        if unsupported:
            result = _stub(
                filename, ["unsupported_type"], "error",
                reason="Only PDF resumes are supported.",
            )
            return _with_explanation(result, explain, settings, jd)
        if len(data) > settings.max_pdf_bytes:
            result = _stub(filename, ["too_large"], "error")
            return _with_explanation(result, explain, settings, jd)
        if not _valid_pdf(upload.content_type, data):
            result = _stub(filename, ["invalid_pdf"], "error")
            return _with_explanation(result, explain, settings, jd)
        extracted = await asyncio.to_thread(
            extract_pdf,
            data,
            filename,
            settings.min_text_chars,
            settings.hidden_font_size,
            settings.max_pdf_pages,
        )
        if extracted.status == "error":
            result = _stub(
                filename, extracted.flags, "error",
                extracted.candidate_name,
            )
            return _with_explanation(result, explain, settings, jd)
        if extracted.status == "needs_ocr":
            reason = (
                "This PDF looks scanned or has too little readable "
                "text. OCR is not enabled yet."
            )
            result = _stub(
                filename, extracted.flags, "needs_ocr",
                extracted.candidate_name, reason,
            )
            return _with_explanation(
                result, explain, settings, jd, extracted.text, reason
            )
        return await _score_text(
            filename, extracted.text, extracted.flags,
            extracted.candidate_name, jd, settings,
            include_debug=explain,
        )
    except HTTPException:
        raise
    except Exception:
        filename_hash = hashlib.sha256(
            filename.encode("utf-8", errors="replace")
        ).hexdigest()[:10]
        log.exception(
            "resume failed file_index=%d filename_hash=%s",
            file_index,
            filename_hash,
        )
        result = _stub(filename, ["invalid_pdf"], "error")
        return _with_explanation(result, explain, settings, jd)


async def _read_upload(
        upload: UploadFile,
        settings: Settings,
        byte_budget: list[int],
) -> bytes:
    request_remaining = byte_budget[0]
    limit = min(settings.max_pdf_bytes, request_remaining)
    data = bytearray()
    while len(data) <= limit:
        chunk = await upload.read(min(64 * 1024, limit + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    if len(data) > request_remaining:
        raise HTTPException(
            status_code=413,
            detail="Request exceeds the maximum size",
        )
    byte_budget[0] -= len(data)
    return bytes(data)


async def _rate_limit_retry_after(
        request: Request,
        settings: Settings,
) -> int | None:
    limit = settings.rate_limit_per_minute
    if limit == 0:
        return None
    client, _forwarded_count = _resolved_client(request, settings)
    hits = app.state.rate_limit_hits
    now = time.monotonic()
    async with app.state.rate_limit_lock:
        for known_client, known_hits in list(hits.items()):
            while (
                known_hits
                and now - known_hits[0] >= RATE_WINDOW_SECONDS
            ):
                known_hits.popleft()
            if not known_hits:
                del hits[known_client]
        while client not in hits and len(hits) >= MAX_TRACKED_CLIENTS:
            least_recent_client = min(
                hits,
                key=lambda known_client: hits[known_client][-1],
            )
            del hits[least_recent_client]
        client_hits = hits.setdefault(client, deque())
        if len(client_hits) >= limit:
            return max(
                1, math.ceil(RATE_WINDOW_SECONDS - (now - client_hits[0]))
            )
        client_hits.append(now)
    return None


def _resolved_client(
        request: Request,
        settings: Settings,
) -> tuple[str, int]:
    forwarded_values = [
        value.strip()
        for value in request.headers.get("x-forwarded-for", "").split(",")
        if value.strip()
    ]
    forwarded_count = len(forwarded_values)
    trusted_count = settings.trusted_proxy_count
    if trusted_count and forwarded_count >= trusted_count:
        client = forwarded_values[-trusted_count]
    else:
        client = request.client.host if request.client else "unknown"
    return client, forwarded_count


def _with_explanation(
        result: AnalyzeResult,
        explain: bool,
        settings: Settings,
        jd: str,
        text: str = "",
        message: str = "",
) -> AnalyzeResult | tuple[AnalyzeResult, dict[str, Any]]:
    if not explain:
        return result
    sections = split_sections(scrub_pii(text))
    lexicon = getattr(app.state, "lexicon", None) or load_skills()
    return result, {
        "filename": result.filename,
        "candidate_name": result.candidate_name,
        "role": result.role or guess_role(text),
        "flags": result.flags,
        "sections": {
            name: len(body)
            for name, body in sections.items()
            if body
        },
        "jd_skills": extract_lexical(scrub_pii(jd), lexicon),
        "matched_skills": {"lexical": [], "semantic": []},
        "missing_skills": [],
        "top_chunks": [],
        "raw_semantic": None,
        "calibrated_semantic": None,
        "skills_coverage": None,
        "section_fit": None,
        "weights": {
            "semantic": settings.semantic_weight,
            "skills": settings.skill_weight,
            "section_fit": settings.section_weight,
        },
        "formula": None,
        "status": result.status,
        "message": message or (
            f"Scoring unavailable; extracted {len(text)} characters."
        ),
    }


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
        include_debug: bool = False,
) -> AnalyzeResult | tuple[AnalyzeResult, dict[str, Any]]:
    model = app.state.model
    lexicon = app.state.lexicon
    prep = await asyncio.to_thread(
        _prepare_score, raw_text, candidate_name, jd, settings, model, lexicon
    )
    chunks = prep.chunks
    jd_chunks = prep.jd_chunks
    resume_texts = prep.resume_texts
    exp_chunks = prep.exp_chunks
    jd_skills = prep.jd_skills
    phrases = prep.phrases
    unlisted_requirements = prep.unlisted_requirements
    semantic_unmatched = prep.semantic_unmatched
    lexical = prep.lexical
    resume_skills = prep.resume_skills
    sections = prep.sections
    jd_skill_weights = prep.jd_skill_weights
    jd_soft_skills = prep.jd_soft_skills
    hard_skill_count = prep.hard_skill_count
    emb = await encode_texts(model, prep.batch)
    n_jd, n_res, n_exp = (
        len(jd_chunks), len(resume_texts), len(exp_chunks),
    )
    n_sk, n_ph = len(jd_skills), len(phrases)
    n_req = len(unlisted_requirements)
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
    i += n_ph
    requirement_emb = emb[i:i + n_req]
    unmatched_idx = [jd_skills.index(s) for s in semantic_unmatched]
    un_emb = (
        skill_emb[unmatched_idx]
        if unmatched_idx and skill_emb.size else np.zeros((0, 1))
    )
    sem_hits = semantic_skill_matches(
        semantic_unmatched, phrases, un_emb, phrase_emb,
        settings.skill_cosine_threshold,
        lexicon,
    )
    matched = list(lexical)
    for skill, _score in sem_hits:
        if skill not in matched:
            matched.append(skill)
    matched_set = set(matched)
    matched = [skill for skill in jd_skills if skill in matched_set]
    missing = [s for s in jd_skills if s not in matched_set]
    skill_scores = {s: 1.0 for s in lexical}
    skill_cosines: dict[str, float] = {}
    sim = np.zeros((0, 0), dtype=np.float32)
    for skill, score in sem_hits:
        skill_scores[skill] = max(skill_scores.get(skill, 0.0), score)
    if skill_emb.size and res_emb.size:
        sim = skill_emb @ res_emb.T
        for idx, skill in enumerate(jd_skills):
            skill_cosines[skill] = float(sim[idx].max())
            skill_scores[skill] = max(
                skill_scores.get(skill, 0.0), skill_cosines[skill]
            )
    evidence_by_section = {
        "experience": settings.experience_skill_weight,
        "projects": settings.project_skill_weight,
        "skills": settings.skills_skill_weight,
        "summary": settings.summary_skill_weight,
        "education": settings.education_skill_weight,
        "extras": settings.other_skill_weight,
        "other": settings.other_skill_weight,
    }
    skill_evidence: dict[str, float] = {}
    for section_name, section_text in sections.items():
        location_weight = evidence_by_section.get(
            section_name, settings.other_skill_weight
        )
        for skill in extract_lexical(section_text, lexicon):
            if skill not in jd_skill_weights:
                continue
            skill_evidence[skill] = max(
                skill_evidence.get(skill, 0.0), location_weight
            )
    for skill, _score in sem_hits:
        if skill in skill_evidence or not sim.size:
            continue
        skill_index = jd_skills.index(skill)
        chunk_index = int(sim[skill_index].argmax())
        section_name = chunks[chunk_index][0]
        skill_evidence[skill] = evidence_by_section.get(
            section_name, settings.other_skill_weight
        )
    raw_sem = topk_mean(res_emb, jd_emb, k=3)
    matched_evidence = {
        skill: skill_evidence.get(skill, 0.0) for skill in matched
    }
    skills_s = evidence_weighted_coverage(
        jd_skill_weights, matched_evidence
    )
    requirement_scores: list[float] = []
    if requirement_emb.size and res_emb.size:
        requirement_sim = requirement_emb @ res_emb.T
        requirement_scores = [
            float(np.clip(row.max(), 0.0, 1.0))
            for row in requirement_sim
        ]
    else:
        requirement_scores = [0.0] * n_req
    if requirement_scores:
        known_weight = sum(jd_skill_weights.values())
        requirement_weight = 0.5 * len(requirement_scores)
        skills_s = (
            skills_s * known_weight
            + 0.5 * sum(requirement_scores)
        ) / (known_weight + requirement_weight)
    jd_vec = mean_pool(jd_emb)
    sec_vec = mean_pool(exp_emb)
    section_s = section_fit_score(jd_vec, sec_vec)
    hard_matched_count = sum(
        lexicon.skill_types.get(skill, "hard") == "hard"
        for skill in matched
    )
    parts = combine_with_signal(
        raw_sem,
        skills_s,
        section_s,
        settings,
        hard_skill_count,
        len(jd_soft_skills),
        hard_matched_count,
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
    result = AnalyzeResult(
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
    if not include_debug:
        return result

    weighted_semantic = parts.semantic_weight * parts.semantic
    weighted_skills = parts.skill_weight * skills_s
    weighted_section_fit = parts.section_weight * section_s
    pre_clipped = (
        weighted_semantic + weighted_skills + weighted_section_fit
    )
    similarities = (
        res_emb @ jd_emb.T
        if res_emb.size and jd_emb.size
        else np.zeros((len(chunks), 0))
    )
    top_chunk_indices = []
    if similarities.size:
        chunk_scores = similarities.max(axis=1)
        top_chunk_indices = sorted(
            range(len(chunks)),
            key=lambda index: (-float(chunk_scores[index]), index),
        )[:3]
    semantic_set = {skill for skill, _cosine in sem_hits}
    detail = {
        "filename": filename,
        "candidate_name": result.candidate_name,
        "role": result.role,
        "flags": flags,
        "sections": {
            name: len(body)
            for name, body in sections.items()
            if body
        },
        "jd_skills": jd_skills,
        "jd_soft_skills": jd_soft_skills,
        "matched_skills": {
            "lexical": [
                {
                    "skill": skill,
                    "cosine": skill_cosines.get(skill),
                }
                for skill in lexical
            ],
            "semantic": [
                {"skill": skill, "cosine": float(cosine)}
                for skill, cosine in sem_hits
                if skill in semantic_set
            ],
        },
        "missing_skills": missing,
        "top_chunks": [
            {
                "section": chunks[index][0],
                "similarity": float(similarities[index].max()),
                "text": (
                    chunks[index][1][:120]
                    + ("..." if len(chunks[index][1]) > 120 else "")
                ),
            }
            for index in top_chunk_indices
        ],
        "raw_semantic": raw_sem,
        "calibrated_semantic": parts.calibrated_semantic,
        "linear_semantic": parts.semantic,
        "jd_skill_weights": jd_skill_weights,
        "unlisted_requirements": [
            {
                "label": "unlisted requirements",
                "requirement": phrase,
                "similarity": score,
                "credit": 0.5 * score,
            }
            for phrase, score in zip(
                unlisted_requirements, requirement_scores
            )
        ],
        "limited_skill_signal": parts.limited_skill_signal,
        "skill_evidence_weights": matched_evidence,
        "skills_coverage": skills_s,
        "section_fit": section_s,
        "weights": {
            "semantic": parts.semantic_weight,
            "skills": parts.skill_weight,
            "section_fit": parts.section_weight,
        },
        "formula": {
            "weighted_semantic": weighted_semantic,
            "weighted_skills": weighted_skills,
            "weighted_section_fit": weighted_section_fit,
            "pre_clipped_score": pre_clipped,
            "final_score": parts.final,
            "hard_skill_penalty": parts.hard_skill_penalty,
            "limited_skill_signal": parts.limited_skill_signal,
            "match_percentage": parts.match_percentage,
            "expression": (
                "round(100 * clip("
                "semantic_weight * linear_semantic + "
                "skill_weight * skills_coverage + "
                "section_weight * section_fit, 0, 1) * "
                "hard_skill_penalty)"
            ),
        },
        "status": result.status,
    }
    return result, detail


def _prepare_score(
        raw_text: str,
        candidate_name: str,
        jd: str,
        settings: Settings,
        model: Any,
        lexicon: Any,
) -> ScorePreparation:
    model_limit = getattr(model, "max_seq_length", None)
    if model_limit is None:
        model_limit = settings.chunk_tokens
    chunk_size = max(1, min(settings.chunk_tokens, int(model_limit)))
    tokenizer = getattr(model, "tokenizer", None)
    token_counter = None
    if tokenizer is not None:
        def token_counter(chunk: str) -> int:
            return len(tokenizer.encode(
                chunk, add_special_tokens=True, verbose=False
            ))

    name_key = candidate_name.strip().casefold()
    scoring_text = "\n".join(
        line for line in raw_text.splitlines()
        if not name_key or line.strip().casefold() != name_key
    )
    text = scrub_pii(strip_invisible_unicode(scoring_text))
    jd_clean = scrub_pii(strip_invisible_unicode(jd))
    sections = split_sections(text)
    chunks = all_chunks(
        sections, chunk_size, settings.chunk_overlap, token_counter
    )
    jd_chunks = chunk_text(
        jd_clean, chunk_size, settings.chunk_overlap, token_counter
    )
    if not jd_chunks:
        jd_chunks = [jd_clean] if jd_clean else []
    jd_all_skills = extract_lexical(jd_clean, lexicon)
    jd_skill_weights = skill_weights(jd_clean, lexicon)
    hard_skill_count = len(jd_skill_weights)
    jd_soft_skills = [
        skill for skill in jd_all_skills
        if lexicon.skill_types.get(skill) == "soft"
    ]
    if hard_skill_count < settings.min_hard_skills:
        for skill in jd_soft_skills:
            jd_skill_weights.setdefault(skill, 0.5)
    jd_skills = list(jd_skill_weights)
    jd_domains = domains_from_skills(jd_clean, jd_all_skills, lexicon)
    resume_skills = technical_skills(
        extract_lexical(text, lexicon), lexicon
    )
    resume_set = set(extract_lexical(text, lexicon))
    lexical = [skill for skill in jd_skills if skill in resume_set]
    unmatched = [skill for skill in jd_skills if skill not in resume_set]
    semantic_unmatched = [
        skill for skill in unmatched
        if (
            lexicon.skill_types.get(skill, "hard") == "hard"
            and domain_allows(skill, jd_domains, lexicon)
        )
    ]
    unlisted_requirements = extract_unlisted_requirements(jd_clean, lexicon)
    phrases = [
        phrase for phrase in noun_phrases(text)
        if extract_lexical(phrase, lexicon)
    ]
    exp_proj = " ".join(
        part for part in (
            sections.get("experience"), sections.get("projects"),
        ) if part
    )
    section_text = exp_proj or text
    exp_chunks = chunk_text(
        section_text, chunk_size, settings.chunk_overlap, token_counter
    ) if section_text else []
    resume_texts = [chunk[1] for chunk in chunks]
    batch = (
        jd_chunks + resume_texts + exp_chunks + jd_skills + phrases
        + unlisted_requirements
    )
    return ScorePreparation(
        chunks=chunks,
        jd_chunks=jd_chunks,
        resume_texts=resume_texts,
        exp_chunks=exp_chunks,
        jd_skills=jd_skills,
        phrases=phrases,
        unlisted_requirements=unlisted_requirements,
        semantic_unmatched=semantic_unmatched,
        lexical=lexical,
        resume_skills=resume_skills,
        sections=sections,
        jd_skill_weights=jd_skill_weights,
        jd_soft_skills=jd_soft_skills,
        hard_skill_count=hard_skill_count,
        batch=batch,
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
