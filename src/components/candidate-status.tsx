import type { Candidate } from "@/lib/screening";

type CandidateStatusPart = "all" | "badges" | "message";

export function CandidateStatus({ candidate, part = "all" }: { candidate: Candidate; part?: CandidateStatusPart }) {
  const status = candidate.status;
  const hiddenText = candidate.flags?.includes("hidden_text");
  if (status !== "needs_ocr" && status !== "error" && !hiddenText) return null;

  return (
    <>
      {part !== "message" && (
        <span className="inline-flex flex-wrap items-center gap-2">
          {status === "needs_ocr" && <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">Needs OCR</span>}
          {status === "error" && <span className="rounded-full bg-danger-soft px-2.5 py-1 text-xs font-semibold text-destructive">Analysis error</span>}
          {hiddenText && <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">Hidden text detected</span>}
        </span>
      )}
      {part !== "badges" && (status === "needs_ocr" || status === "error") && (
        <p className="text-sm text-muted-foreground">
          {candidate.reason || (status === "needs_ocr" ? "This resume needs OCR before it can be scored." : "This resume could not be analyzed.")}
        </p>
      )}
    </>
  );
}
