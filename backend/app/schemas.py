from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    semantic: int
    skills: int
    section_fit: int


class AnalyzeResult(BaseModel):
    filename: str
    candidate_name: str
    match_percentage: int
    top_skills: list[str]
    missing_skills: list[str]
    score_breakdown: ScoreBreakdown
    reason: str
    flags: list[str]
    status: str
    role: str = ""
    experience: str = ""
    skills: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    breakdown: ScoreBreakdown | None = None


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResult]


class HealthResponse(BaseModel):
    status: str
    model: str
