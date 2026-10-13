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
    matched: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown
    reason: str
    flags: list[str]
    status: str
    role: str = ""
    experience: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResult]


class HealthResponse(BaseModel):
    status: str
    model: str
