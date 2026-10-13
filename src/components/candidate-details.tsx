import { useEffect, useState } from "react";
import { Star, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { Progress } from "@/components/ui/progress";
import { getResume } from "@/lib/resume-store";
import { useScreening, type Candidate } from "@/lib/screening";

function Chips({ items, empty, muted }: { items: string[]; empty: string; muted?: boolean }) {
  if (!items.length) return <p className="text-sm text-muted-foreground">{empty}</p>;
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((s) => (
        <span key={s} className={muted ? "rounded-full bg-danger-soft px-3 py-1 text-xs text-destructive" : "rounded-full bg-secondary px-3 py-1 text-xs text-secondary-foreground"}>{s}</span>
      ))}
    </div>
  );
}

export function CandidateDetails({ candidate, onClose }: { candidate: Candidate | null; onClose: () => void }) {
  const { jobs, files, shortlist, toggleShortlist, rejected, reject } = useScreening();
  const [url, setUrl] = useState<string>();
  const [loading, setLoading] = useState(false);
  const job = jobs.find((j) => j.candidates.some((c) => c.id === candidate?.id));
  const resumeId = candidate?.resumeId;
  const resume = [...files, ...jobs.flatMap((j) => j.files)].find((f) => f.id === resumeId);

  useEffect(() => {
    setUrl(undefined);
    if (!resumeId) return;
    let alive = true;
    let objectUrl: string | undefined;
    setLoading(true);
    const local = files.find((f) => f.id === resumeId)?.file;
    void (local ? Promise.resolve(local as Blob) : getResume(resumeId)).then((blob) => {
      if (!alive) return;
      setLoading(false);
      if (blob) { objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); }
    });
    return () => { alive = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [resumeId, files]);

  const b = candidate?.breakdown;
  return (
    <Sheet open={Boolean(candidate)} onOpenChange={(open) => { if (!open) onClose(); }}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        {candidate && (
          <div className="space-y-7 pb-8">
            <div className="pr-8">
              <SheetTitle className="font-display text-2xl capitalize">{candidate.name}</SheetTitle>
              <SheetDescription>{candidate.role}{job ? ` · ${job.title}` : ""}</SheetDescription>
            </div>
            <div>
              <p className="font-display text-6xl font-bold text-primary">{candidate.score}%</p>
              <p className="mt-1 text-sm text-muted-foreground">Match score{job?.demo ? " · demo data" : ""}</p>
            </div>
            <section><h3 className="mb-2 font-semibold">Summary</h3><p className="text-sm text-muted-foreground">{candidate.summary}</p></section>
            <div className="grid gap-6 sm:grid-cols-2">
              <section><h3 className="mb-2 font-semibold">Strengths</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">{(candidate.strengths?.length ? candidate.strengths : ["None found"]).map((s) => <li key={s}>{s}</li>)}</ul></section>
              <section><h3 className="mb-2 font-semibold">Gaps</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">{(candidate.gaps?.length ? candidate.gaps : ["No major gaps"]).map((s) => <li key={s}>{s}</li>)}</ul></section>
            </div>
            <section><h3 className="mb-2 font-semibold">Matched skills</h3><Chips items={candidate.matched ?? candidate.skills} empty="No required skills matched" /></section>
            <section><h3 className="mb-2 font-semibold">Missing skills</h3><Chips items={candidate.missing ?? []} empty="Nothing missing" muted /></section>
            {b && (
              <section className="space-y-3"><h3 className="font-semibold">Score breakdown</h3>
                {([["Skills", b.skills], ["Experience", b.experience], ["Education", b.education]] as const).map(([label, v]) => (
                  <div key={label}>
                    <div className="mb-1 flex justify-between text-sm"><span>{label}</span><span className="text-muted-foreground">{v}%</span></div>
                    <Progress value={v} className="h-1.5" />
                  </div>
                ))}
              </section>
            )}
            <section><h3 className="mb-3 font-semibold">Original resume</h3>
              {url ? (
                <>
                  <a href={url} target="_blank" rel="noreferrer" className="text-sm text-primary underline">Open {resume?.name ?? "resume"}</a>
                  <iframe src={url} title={`Resume for ${candidate.name}`} className="mt-3 h-96 w-full rounded-2xl border bg-card" />
                </>
              ) : (
                <p className="text-sm text-muted-foreground">{loading ? "Loading resume…" : resumeId ? "Resume file is no longer available." : "No resume attached to this demo candidate."}</p>
              )}
            </section>
            <div className="flex gap-3">
              <Button className="rounded-full" onClick={() => toggleShortlist(candidate.id)}><Star />{shortlist.includes(candidate.id) ? "Shortlisted" : "Shortlist"}</Button>
              <Button variant="outline" className="rounded-full" onClick={() => reject(candidate.id)}><X />{rejected.includes(candidate.id) ? "Restore" : "Reject"}</Button>
            </div>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
