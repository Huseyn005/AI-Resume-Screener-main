import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import { extractPdfText, scoreResume } from "./mock/scoring";
import { analyzeResumes, type BackendAnalysisResult } from "./api";
import { clearResumes, deleteResumes, getResume, putResume } from "./resume-store";

export type Candidate = {
  id: string;
  name: string;
  role: string;
  score: number | null;
  skills: string[];
  topSkills?: string[];
  experience: string;
  summary: string;
  resumeId?: string;
  matched?: string[];
  missing?: string[];
  strengths?: string[];
  gaps?: string[];
  breakdown?: { skills: number; experience: number; education: number };
  scoreBreakdown?: { semantic: number; skills: number; sectionFit: number };
  reason?: string;
  flags?: string[];
  status?: BackendAnalysisResult["status"];
};

export type ResumeFile = { id: string; name: string; size: number; file?: File };
export type Job = {
  id: string;
  title: string;
  employment: string;
  jd: string;
  date: string;
  candidates: Candidate[];
  files: ResumeFile[];
  demo?: boolean;
};
export type Weights = { skills: number; experience: number; education: number };
export type Profile = { name: string; email: string };

export type Level = "great" | "good" | "low";
export const levelOf = (score: number): Level =>
  score >= 85 ? "great" : score >= 70 ? "good" : "low";
export const levelLabel: Record<Level, string> = {
  great: "Great match",
  good: "Good match",
  low: "Weak match",
};
export const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((n) => n[0])
    .join("")
    .toUpperCase();
export const average = (cs: Candidate[]) => {
  const scored = cs.filter((c): c is Candidate & { score: number } => c.score !== null);
  return scored.length
    ? Math.round(scored.reduce((sum, c) => sum + c.score, 0) / scored.length)
    : 0;
};
export const formatSize = (b: number) =>
  b < 1024 * 1024
    ? `${Math.max(1, Math.round(b / 1024))} KB`
    : `${(b / 1024 / 1024).toFixed(1)} MB`;
export const EMPLOYMENT = ["Full-time", "Part-time", "Contract", "Internship"] as const;

const DEFAULT_WEIGHTS: Weights = { skills: 60, experience: 30, education: 10 };
const UPLOAD_BATCH_SIZE = 5;
const demo = (
  id: string,
  name: string,
  role: string,
  score: number,
  matched: string[],
  missing: string[],
  yrs: number,
  b: [number, number, number],
): Candidate => ({
  id,
  name,
  role,
  score,
  skills: matched,
  matched,
  missing,
  experience: `${yrs} years`,
  summary: `Demo data: ${yrs} years of experience with ${matched.slice(0, 2).join(" and ")}.`,
  strengths: [`Covers ${matched.length} of 4 required skills`, `${yrs} years of experience`],
  gaps: missing.length ? [`Missing ${missing.join(", ")}`] : ["No major gaps"],
  breakdown: { skills: b[0], experience: b[1], education: b[2] },
});
const demoJob: Job = {
  id: "demo-design",
  title: "Product Designer",
  employment: "Full-time",
  date: "2026-10-01T09:00:00.000Z",
  files: [],
  demo: true,
  jd: "Product designer with 5 years of experience in Figma, UX Research, Design Systems and Prototyping.",
  candidates: [
    demo(
      "demo-maya",
      "Maya Chen",
      "Senior Product Designer",
      96,
      ["Figma", "UX Research", "Design Systems", "Prototyping"],
      [],
      8,
      [100, 100, 90],
    ),
    demo(
      "demo-alex",
      "Alex Morgan",
      "Product Designer",
      84,
      ["Figma", "Prototyping", "Design Systems"],
      ["UX Research"],
      6,
      [75, 100, 75],
    ),
    demo(
      "demo-sofia",
      "Sofia Patel",
      "UX Designer",
      71,
      ["UX Research", "Figma", "Prototyping"],
      ["Design Systems"],
      4,
      [75, 80, 75],
    ),
  ],
};

type Saved = {
  jobs: Job[];
  shortlist: string[];
  rejected: string[];
  weights: Weights;
  profile: Profile;
};
const KEY = "shortlist-workspace-v1";

type Ctx = {
  jd: string;
  setJD: (v: string) => void;
  title: string;
  setTitle: (v: string) => void;
  employment: string;
  setEmployment: (v: string) => void;
  files: ResumeFile[];
  setFiles: React.Dispatch<React.SetStateAction<ResumeFile[]>>;
  candidates: Candidate[];
  setCandidates: (c: Candidate[]) => void;
  shortlist: string[];
  toggleShortlist: (id: string) => void;
  rejected: string[];
  reject: (id: string) => void;
  jobs: Job[];
  activeJob: string | null;
  openJob: (id: string) => void;
  newJob: () => void;
  runScreening: (onProgress: (done: number, total: number) => void) => Promise<Candidate[]>;
  deleteJob: (id: string) => Promise<void>;
  duplicateJob: (id: string) => void;
  weights: Weights;
  setWeights: (v: Weights) => void;
  profile: Profile;
  setProfile: (v: Profile) => void;
  signOut: () => void;
  clearData: () => Promise<void>;
};

const ScreeningContext = createContext<Ctx | null>(null);

export function ScreeningProvider({ children }: { children: ReactNode }) {
  const [jd, setJD] = useState("");
  const [title, setTitle] = useState("");
  const [employment, setEmployment] = useState("Full-time");
  const [files, setFiles] = useState<ResumeFile[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [shortlist, setShortlist] = useState<string[]>([]);
  const [rejected, setRejected] = useState<string[]>([]);
  const [jobs, setJobs] = useState<Job[]>([demoJob]);
  const [activeJob, setActiveJob] = useState<string | null>(null);
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [profile, setProfile] = useState<Profile>({ name: "", email: "" });
  const loaded = useRef(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const s = JSON.parse(raw) as Partial<Saved>;
        if (s.jobs) setJobs(s.jobs);
        setShortlist(s.shortlist ?? []);
        setRejected(s.rejected ?? []);
        setWeights(s.weights ?? DEFAULT_WEIGHTS);
        setProfile(s.profile ?? { name: "", email: "" });
      }
    } catch {
      /* ignore corrupt storage */
    }
    loaded.current = true;
  }, []);

  useEffect(() => {
    if (!loaded.current) return;
    const cleanJobs = jobs.map((j) => ({ ...j, files: j.files.map(({ file: _f, ...f }) => f) }));
    localStorage.setItem(
      KEY,
      JSON.stringify({ jobs: cleanJobs, shortlist, rejected, weights, profile }),
    );
  }, [jobs, shortlist, rejected, weights, profile]);

  const newJob = () => {
    setActiveJob(null);
    setJD("");
    setTitle("");
    setEmployment("Full-time");
    setFiles([]);
    setCandidates([]);
  };
  const openJob = (id: string) => {
    const j = jobs.find((j) => j.id === id);
    if (!j) return;
    setActiveJob(id);
    setJD(j.jd);
    setTitle(j.title);
    setEmployment(j.employment);
    setFiles(j.files);
    setCandidates(j.candidates);
  };

  async function runScreening(onProgress: (done: number, total: number) => void) {
    const id = activeJob && activeJob !== demoJob.id ? activeJob : crypto.randomUUID();
    const prev = jobs.find((j) => j.id === id)?.candidates ?? [];
    const results: Candidate[] = [];
    if (import.meta.env["VITE_USE_MOCK"] === "true") {
      for (let i = 0; i < files.length; i++) {
        const f = files[i]!;
        let text = "";
        if (f.file) {
          await putResume(f.id, f.file);
          text = await extractPdfText(f.file);
        } else {
          const blob = await getResume(f.id);
          if (blob) text = await extractPdfText(blob);
        }
        const s = scoreResume(text, f.name, jd, weights);
        results.push({
          ...s,
          id: prev.find((c) => c.resumeId === f.id)?.id ?? f.id,
          resumeId: f.id,
        });
        onProgress(i + 1, files.length);
      }
    } else {
      const failures: string[] = [];
      for (let start = 0; start < files.length; start += UPLOAD_BATCH_SIZE) {
        const batchFiles = files.slice(start, start + UPLOAD_BATCH_SIZE);
        try {
          const uploads = await Promise.all(
            batchFiles.map(async (f) => {
              if (f.file) {
                await putResume(f.id, f.file);
                return f.file;
              }
              const blob = await getResume(f.id);
              if (!blob) {
                throw new Error(`Resume file "${f.name}" is no longer available.`);
              }
              return new File([blob], f.name, { type: blob.type || "application/pdf" });
            }),
          );
          const payload = await analyzeResumes(jd, uploads);
          const analyses = getAnalysisResults(payload);
          if (analyses.length !== batchFiles.length) {
            throw new Error(
              `The backend returned ${analyses.length} results for ${batchFiles.length} resumes.`,
            );
          }
          analyses.forEach((analysis, index) => {
            const fileIndex = batchFiles.findIndex((f) => f.name === analysis.filename);
            const file = batchFiles[fileIndex >= 0 ? fileIndex : index]!;
            results.push(
              mapAnalysis(
                analysis,
                file.id,
                prev.find((c) => c.resumeId === file.id)?.id ?? file.id,
              ),
            );
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : "Unknown analysis error";
          failures.push(message);
          batchFiles.forEach((file) => {
            results.push({
              id: prev.find((c) => c.resumeId === file.id)?.id ?? file.id,
              resumeId: file.id,
              name: file.name,
              role: "",
              score: null,
              skills: [],
              experience: "",
              summary: `This resume could not be analyzed: ${message}`,
              reason: `Analysis failed: ${message}`,
              flags: [],
              status: "error",
            });
          });
        } finally {
          onProgress(Math.min(start + batchFiles.length, files.length), files.length);
        }
      }
      if (failures.length === Math.ceil(files.length / UPLOAD_BATCH_SIZE)) {
        throw new Error(`All resume batches failed. ${failures.join(" ")}`);
      }
    }
    results.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
    const job: Job = {
      id,
      title: title.trim() || "Untitled job",
      employment,
      jd,
      date: new Date().toISOString(),
      candidates: results,
      files: files.map(({ file: _f, ...f }) => f),
    };
    setJobs((p) => [job, ...p.filter((j) => j.id !== id)]);
    setActiveJob(id);
    setCandidates(results);
    return results;
  }

  async function deleteJob(id: string) {
    const j = jobs.find((j) => j.id === id);
    const used = jobs.filter((x) => x.id !== id).flatMap((x) => x.files.map((f) => f.id));
    await deleteResumes((j?.files ?? []).map((f) => f.id).filter((f) => !used.includes(f)));
    setJobs((p) => p.filter((x) => x.id !== id));
    const ids = j?.candidates.map((c) => c.id) ?? [];
    setShortlist((p) => p.filter((x) => !ids.includes(x)));
    setRejected((p) => p.filter((x) => !ids.includes(x)));
    if (activeJob === id) newJob();
    toast.success("Job deleted");
  }

  const duplicateJob = (id: string) => {
    const j = jobs.find((j) => j.id === id);
    if (!j) return;
    const copy: Job = {
      ...j,
      id: crypto.randomUUID(),
      title: `${j.title} (copy)`,
      date: new Date().toISOString(),
      demo: false,
      candidates: j.candidates.map((c) => ({ ...c, id: crypto.randomUUID() })),
    };
    setJobs((p) => [copy, ...p]);
    toast.success("Job duplicated");
  };

  async function clearData() {
    await clearResumes();
    setJobs([]);
    setShortlist([]);
    setRejected([]);
    newJob();
    toast.success("All screening data cleared");
  }

  const toggleShortlist = (id: string) => {
    const was = shortlist.includes(id);
    setShortlist((s) => (was ? s.filter((x) => x !== id) : [...s, id]));
    setRejected((s) => s.filter((x) => x !== id));
    toast.success(was ? "Removed from shortlist" : "Candidate shortlisted");
  };
  const reject = (id: string) => {
    const was = rejected.includes(id);
    setRejected((s) => (was ? s.filter((x) => x !== id) : [...s, id]));
    setShortlist((s) => s.filter((x) => x !== id));
    toast.success(was ? "Candidate restored" : "Candidate rejected");
  };
  const signOut = () => {
    setProfile({ name: "", email: "" });
    toast.success("Signed out");
  };

  return (
    <ScreeningContext.Provider
      value={{
        jd,
        setJD,
        title,
        setTitle,
        employment,
        setEmployment,
        files,
        setFiles,
        candidates,
        setCandidates,
        shortlist,
        toggleShortlist,
        rejected,
        reject,
        jobs,
        activeJob,
        openJob,
        newJob,
        runScreening,
        deleteJob,
        duplicateJob,
        weights,
        setWeights,
        profile,
        setProfile,
        signOut,
        clearData,
      }}
    >
      {children}
    </ScreeningContext.Provider>
  );
}

function getAnalysisResults(payload: unknown): BackendAnalysisResult[] {
  const results = Array.isArray(payload)
    ? payload
    : typeof payload === "object" && payload !== null && "results" in payload
      ? payload.results
      : undefined;
  if (!Array.isArray(results) || !results.every(isBackendAnalysisResult)) {
    throw new Error("The backend returned an invalid analysis response.");
  }
  return results;
}

function isBackendAnalysisResult(value: unknown): value is BackendAnalysisResult {
  if (typeof value !== "object" || value === null) return false;
  const result = value as Record<string, unknown>;
  const breakdown = result["score_breakdown"];
  const isStringArray = (items: unknown) =>
    Array.isArray(items) && items.every((item) => typeof item === "string");
  return (
    typeof result["filename"] === "string" &&
    typeof result["candidate_name"] === "string" &&
    typeof result["match_percentage"] === "number" &&
    isStringArray(result["top_skills"]) &&
    isStringArray(result["missing_skills"]) &&
    isStringArray(result["matched"]) &&
    isStringArray(result["skills"]) &&
    typeof breakdown === "object" &&
    breakdown !== null &&
    "semantic" in breakdown &&
    typeof breakdown["semantic"] === "number" &&
    "skills" in breakdown &&
    typeof breakdown["skills"] === "number" &&
    "section_fit" in breakdown &&
    typeof breakdown["section_fit"] === "number" &&
    typeof result["reason"] === "string" &&
    isStringArray(result["flags"]) &&
    (result["status"] === "ok" ||
      result["status"] === "needs_ocr" ||
      result["status"] === "error") &&
    typeof result["role"] === "string" &&
    typeof result["experience"] === "string" &&
    isStringArray(result["strengths"]) &&
    isStringArray(result["gaps"]) &&
    typeof result["summary"] === "string"
  );
}

function mapAnalysis(result: BackendAnalysisResult, resumeId: string, id: string): Candidate {
  return {
    id,
    resumeId,
    name: result.candidate_name || result.filename,
    role: result.role,
    score: result.status === "ok" ? result.match_percentage : null,
    skills: result.skills,
    topSkills: result.top_skills,
    experience: result.experience,
    summary: result.summary,
    matched: result.matched,
    missing: result.missing_skills,
    strengths: result.strengths,
    gaps: result.gaps,
    scoreBreakdown: {
      semantic: result.score_breakdown.semantic,
      skills: result.score_breakdown.skills,
      sectionFit: result.score_breakdown.section_fit,
    },
    reason: result.reason,
    flags: result.flags,
    status: result.status,
  };
}

export function useScreening() {
  const c = useContext(ScreeningContext);
  if (!c) throw new Error("useScreening must be used inside ScreeningProvider");
  return c;
}
