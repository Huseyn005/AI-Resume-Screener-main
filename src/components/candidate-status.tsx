import type { Candidate } from "@/lib/screening";

export function CandidateStatus({ candidate }: { candidate: Candidate }) {
  const status = candidate.status;
  const hiddenText = candidate.flags?.includes("hidden_text");
  if (status !== "needs_ocr" && status !== "error" && !hiddenText) return null;

  return (
    <div className="space-y-2">
      {status === "needs_ocr" && (
        <div>
          <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">Needs OCR</span>
          <p className="mt-2 text-sm text-muted-foreground">{candidate.reason || "This resume needs OCR before it can be scored."}</p>
        </div>
      )}
      {status === "error" && (
        <div>
          <span className="rounded-full bg-danger-soft px-2.5 py-1 text-xs font-semibold text-destructive">Analysis error</span>
          <p className="mt-2 text-sm text-muted-foreground">{candidate.reason || "This resume could not be analyzed."}</p>
        </div>
      )}
      {hiddenText && <span className="inline-flex rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">Hidden text detected</span>}
    </div>
  );
}
