import { useState } from "react";
import { ChevronDown, Star } from "lucide-react";
import { cn } from "@/lib/utils";
import { levelLabel, levelOf, useScreening, type Candidate } from "@/lib/screening";
import { CandidateStatus } from "@/components/candidate-status";

const tone = {
  great: { ring: "text-success", chip: "bg-success-soft text-success", bar: "bg-success" },
  good: { ring: "text-warning", chip: "bg-warning-soft text-foreground", bar: "bg-warning" },
  low: { ring: "text-destructive", chip: "bg-danger-soft text-destructive", bar: "bg-destructive" },
};

export function CandidateCard({ c, rank }: { c: Candidate; rank: number }) {
  const { shortlist, toggleShortlist } = useScreening();
  const [open, setOpen] = useState(false);
  const t = tone[c.score === null ? "low" : levelOf(c.score)];
  const saved = shortlist.includes(c.id);

  return (
    <article className="rounded-2xl border bg-card p-4 shadow-soft transition-transform hover:-translate-y-0.5">
      <div className="flex items-center gap-4">
        {c.score === null ? <CandidateStatus candidate={c} /> : (
          <div className={cn("relative grid size-14 shrink-0 place-items-center", t.ring)}>
            <svg viewBox="0 0 36 36" className="absolute inset-0 -rotate-90">
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="currentColor" strokeOpacity=".15" strokeWidth="3" />
              <circle cx="18" cy="18" r="15.5" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round"
                strokeDasharray={`${(c.score / 100) * 97.4} 97.4`} />
            </svg>
            <span className="font-display text-base font-bold text-foreground">{c.score}</span>
          </div>
        )}

        <button onClick={() => setOpen(!open)} className="min-w-0 flex-1 text-left">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-muted-foreground">#{rank}</span>
            <h3 className="truncate font-display text-lg font-semibold capitalize">{c.name}</h3>
            {c.score !== null && <span className={cn("rounded-full px-2.5 py-0.5 text-xs font-semibold", t.chip)}>{levelLabel[levelOf(c.score)]}</span>}
          </div>
          <p className="text-sm text-muted-foreground">{c.role} · {c.experience}</p>
        </button>

        <button
          onClick={() => toggleShortlist(c.id)}
          aria-label={saved ? "Remove from shortlist" : "Add to shortlist"}
          className={cn(
            "grid size-10 place-items-center rounded-full border transition-colors",
            saved ? "border-accent bg-accent text-accent-foreground" : "hover:border-accent hover:text-accent",
          )}
        >
          <Star className={cn("size-5", saved && "fill-current")} />
        </button>
        <button onClick={() => setOpen(!open)} aria-label="Show details" className="text-muted-foreground">
          <ChevronDown className={cn("size-5 transition-transform", open && "rotate-180")} />
        </button>
      </div>

      {open && (
        <div className="mt-4 space-y-3 border-t pt-4 text-sm">
          <p>{c.summary}</p>
          <div className="flex flex-wrap gap-2">
            {c.skills.map((s) => (
              <span key={s} className="rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground">{s}</span>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}
