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
  constructor(baseUrl: string) {
    super(`Backend not reachable at ${baseUrl}`);
    this.name = "BackendNotReachableError";
  }
}

type AnalyzeOptions = { timeoutMs?: number };

const RETRYABLE_STATUSES = new Set([502, 503, 504]);
const DEFAULT_TIMEOUT_MS = 120_000;

function errorMessage(status: number, detail: string): string {
  if (status === 413) {
    return "The uploaded files exceed the server's size limit. Choose smaller PDFs.";
  }
  if (status === 429) {
    return "Too many analyses were submitted. Wait a minute before trying again.";
  }
  if (status === 503) {
    return "The analysis service is busy. Please try again shortly.";
  }
  return detail || `Analysis failed with status ${status}.`;
}

function waitBeforeRetry(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 500));
}

export async function analyzeResumes(
  jd: string,
  files: File[],
  options: AnalyzeOptions = {},
): Promise<unknown> {
  const baseUrl = import.meta.env["VITE_API_URL"]?.replace(/\/+$/, "");
  if (!baseUrl) throw new Error("VITE_API_URL is not configured.");

  const formData = new FormData();
  formData.append("job_description", jd);
  files.forEach((file) => formData.append("resumes", file, file.name));

  const hostname = new URL(baseUrl, window.location.origin).hostname;
  const needsNgrokWarningHeader = [".ngrok-free.app", ".ngrok-free.dev", ".ngrok.app"].some(
    (suffix) => hostname.endsWith(suffix),
  );
  const requestOptions: RequestInit = {
    method: "POST",
    body: formData,
    ...(needsNgrokWarningHeader ? { headers: { "ngrok-skip-browser-warning": "true" } } : {}),
  };
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  let response: Response | undefined;
  for (let attempt = 0; attempt < 2; attempt++) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      response = await fetch(`${baseUrl}/analyze`, {
        ...requestOptions,
        signal: controller.signal,
      });
    } catch {
      if (controller.signal.aborted) {
        throw new Error("Analysis timed out. Please try again.");
      }
      if (attempt === 0) {
        await waitBeforeRetry();
        continue;
      }
      throw new BackendNotReachableError(baseUrl);
    } finally {
      clearTimeout(timeout);
    }
    if (RETRYABLE_STATUSES.has(response.status) && attempt === 0) {
      await response.body?.cancel();
      await waitBeforeRetry();
      continue;
    }
    break;
  }

  if (!response) throw new BackendNotReachableError(baseUrl);
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
    throw new Error(errorMessage(response.status, detail));
  }

  return response.json();
}
