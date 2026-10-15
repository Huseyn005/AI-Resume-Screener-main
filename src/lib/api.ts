export type BackendAnalysisResult = {
  filename: string;
  candidate_name: string;
  match_percentage: number;
  top_skills: string[];
  missing_skills: string[];
  matched: string[];
  skills: string[];
  score_breakdown: { semantic: number; skills: number; section_fit: number };
  reason: string;
  flags: string[];
  status: "ok" | "needs_ocr" | "error";
  role: string;
  experience: string;
  strengths: string[];
  gaps: string[];
  summary: string;
};

export class BackendNotReachableError extends Error {
  constructor() {
    super("Backend not reachable on localhost:8000");
    this.name = "BackendNotReachableError";
  }
}

export async function analyzeResumes(jd: string, files: File[]): Promise<unknown> {
  const baseUrl = import.meta.env["VITE_API_URL"]?.replace(/\/+$/, "");
  if (!baseUrl) throw new Error("VITE_API_URL is not configured.");

  const formData = new FormData();
  formData.append("job_description", jd);
  files.forEach((file) => formData.append("resumes", file, file.name));

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/analyze`, { method: "POST", body: formData });
  } catch {
    throw new BackendNotReachableError();
  }

  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const message =
      typeof body === "object" && body !== null
        ? "detail" in body
          ? body.detail
          : "error" in body
            ? body.error
            : "message" in body
              ? body.message
              : undefined
        : undefined;
    const detail =
      typeof message === "string" ? message : message === undefined ? "" : JSON.stringify(message);
    throw new Error(detail || `Analysis failed with status ${response.status}.`);
  }

  return response.json();
}
