import type { Candidate } from "@/features/screening/screening";

type CandidateStatusPart = "all" | "badges" | "message";

const FLAG_MESSAGES: Record<string, string> = {
  too_many_pages: "This PDF exceeds the 20-page limit. Split it into smaller files.",
  unsupported_type: "This file is not a supported PDF.",
  too_large: "This PDF exceeds the 10 MB per-file limit.",
  invalid_pdf: "This file is not a readable PDF.",
  scanned_pdf: "This PDF has no readable text and needs OCR.",
};

export function CandidateStatus({
  candidate,
  part = "all",
}: {
  candidate: Candidate;
  part?: CandidateStatusPart;
}) {
  const status = candidate.status;
  const hiddenText = candidate.flags?.includes("hidden_text");
  const flagMessage = candidate.flags
    ?.map((flag) => FLAG_MESSAGES[flag])
    .find((message) => message !== undefined);
  if (status !== "needs_ocr" && status !== "error" && !hiddenText) return null;

  return (
    <>
      {part !== "message" && (
        <span className="inline-flex flex-wrap items-center gap-2">
          {status === "needs_ocr" && (
            <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">
              Needs OCR
            </span>
          )}
          {status === "error" && (
            <span className="rounded-full bg-danger-soft px-2.5 py-1 text-xs font-semibold text-destructive">
              Analysis error
            </span>
          )}
          {hiddenText && (
            <span className="rounded-full bg-warning-soft px-2.5 py-1 text-xs font-semibold text-foreground">
              Hidden text detected
            </span>
          )}
        </span>
      )}
      {part !== "badges" && (status === "needs_ocr" || status === "error") && (
        <p className="text-sm text-muted-foreground">
          {flagMessage ||
            candidate.reason ||
            (status === "needs_ocr"
              ? "This resume needs OCR before it can be scored."
              : "This resume could not be analyzed.")}
        </p>
      )}
    </>
  );
}
