import { createFileRoute, Link } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { Check, FileText, LoaderCircle, RotateCcw, Sparkles, UploadCloud, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { CandidateCard } from "@/components/candidate-card";
import { cn } from "@/lib/utils";
import { average, EMPLOYMENT, formatSize, levelOf, useScreening } from "@/lib/screening";

export const Route = createFileRoute("/")({
  component: ScreenPage,
});

function Step({ n, title, done, children }: { n: number; title: string; done: boolean; children: React.ReactNode }) {
  return (
    <section className="rounded-3xl border bg-card p-6 shadow-soft">
      <h2 className="mb-4 flex items-center gap-3 font-display text-xl font-semibold">
        <span className={cn("grid size-8 place-items-center rounded-full text-sm font-bold transition-colors",
          done ? "bg-success text-primary-foreground" : "bg-secondary text-secondary-foreground")}>
          {done ? <Check className="size-4" /> : n}
        </span>
        {title}
      </h2>
      {children}
    </section>
  );
}

function ScreenPage() {
  const { jd, setJD, title, setTitle, employment, setEmployment, files, setFiles, candidates, runScreening, newJob } = useScreening();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [filter, setFilter] = useState<"all" | "great" | "good" | "low">("all");

  function add(list: File[]) {
    const ok = list.filter((f) => f.name.toLowerCase().endsWith(".pdf") && f.size <= 10 * 1024 * 1024);
    if (ok.length !== list.length) toast.error("Only PDF files up to 10 MB");
    setFiles((cur) => [
      ...cur,
      ...ok.filter((f) => !cur.some((c) => c.name === f.name)).map((f) => ({ id: crypto.randomUUID(), name: f.name, size: f.size, file: f })),
    ]);
  }

  async function run() {
    setBusy(true); setProgress(0);
    try {
      const results = await runScreening((d, t) => setProgress(Math.round((d / t) * 100)));
      toast.success(`${results.length} resumes analyzed successfully.`);
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Something went wrong while checking resumes.");
    } finally {
      setBusy(false);
    }
  }

  const ready = jd.trim().length > 20 && files.length > 0;
  const shown = candidates.filter((c) => c.score === null ? filter === "all" : filter === "all" || levelOf(c.score) === filter);
  const top = candidates.find((c) => c.score !== null);

  return (
    <main className="mx-auto max-w-3xl space-y-6 px-4 py-10">
      <div className="text-center">
        <h1 className="font-display text-4xl font-bold sm:text-5xl">
          Find your best <span className="bg-hero bg-clip-text text-transparent">candidates</span>
        </h1>
        <p className="mt-3 text-lg text-muted-foreground">Three quick steps. No setup needed.</p>
      </div>

      <Step n={1} title="Describe the job" done={jd.trim().length > 20}>
        <Input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={100} placeholder="Job title, e.g. Product Designer"
          className="mb-3 h-11 rounded-2xl text-base" aria-label="Job title" />
        <div className="mb-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Employment type">
          {EMPLOYMENT.map((t) => (
            <button key={t} type="button" role="radio" aria-checked={employment === t} onClick={() => setEmployment(t)}
              className={cn("rounded-full px-3 py-1.5 text-sm font-semibold transition-colors",
                employment === t ? "bg-secondary text-secondary-foreground" : "text-muted-foreground hover:bg-muted")}>
              {t}
            </button>
          ))}
        </div>
        <Textarea value={jd} onChange={(e) => setJD(e.target.value)} rows={6}
          placeholder="Paste the job description — role, responsibilities and must-have skills…" className="rounded-2xl text-base" />
      </Step>

      <Step n={2} title="Add resumes" done={files.length > 0}>
        <div
          onClick={() => input.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); add(Array.from(e.dataTransfer.files)); }}
          className={cn("cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-colors",
            dragging ? "border-primary bg-secondary" : "hover:border-primary hover:bg-secondary/50")}
        >
          <UploadCloud className="mx-auto size-10 text-primary" />
          <p className="mt-2 font-semibold">Drop PDF resumes here or click to choose</p>
          <p className="text-sm text-muted-foreground">Up to 10 MB each</p>
          <input ref={input} type="file" accept=".pdf" multiple hidden onChange={(e) => { add(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
        </div>
        {files.length > 0 && (
          <ul className="mt-4 divide-y rounded-2xl border">
            {files.map((f) => (
              <li key={f.id} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                <FileText className="size-4 shrink-0 text-primary" />
                <span className="min-w-0 flex-1 truncate">{f.name}</span>
                <span className="text-xs text-muted-foreground">{formatSize(f.size)}</span>
                <button aria-label={`Remove ${f.name}`} onClick={() => setFiles((c) => c.filter((x) => x.id !== f.id))}
                  className="grid size-7 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground">
                  <X className="size-4" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </Step>

      <Step n={3} title="See who fits" done={candidates.length > 0}>
        <Button size="lg" disabled={!ready || busy} onClick={run}
          className="h-14 w-full rounded-2xl bg-hero text-base font-semibold text-primary-foreground hover:opacity-90">
          {busy ? <LoaderCircle className="animate-spin" /> : <Sparkles />}
          {busy ? "Checking resumes…" : candidates.length ? "Check again" : "Find matches"}
        </Button>
        {busy && <Progress value={progress} className="mt-4 h-1" />}
        {!ready && !busy && <p className="mt-2 text-center text-sm text-muted-foreground">Finish steps 1 and 2 to continue.</p>}
      </Step>

      {busy && (
        <div className="space-y-3 pt-4" aria-label="Loading results">
          {files.slice(0, 4).map((f) => <Skeleton key={f.id} className="h-20 w-full rounded-2xl" />)}
        </div>
      )}

      {!busy && candidates.length > 0 && (
        <section id="results" className="space-y-4 pt-4">
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: "Resumes scanned", value: String(candidates.length) },
              { label: "Average match", value: candidates.some((c) => c.score !== null) ? `${average(candidates)}%` : "—" },
              { label: "Top candidate", value: top?.name ?? "—" },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border bg-card p-4">
                <p className="text-xs text-muted-foreground">{s.label}</p>
                <p className="mt-1 truncate font-display text-xl font-bold capitalize">{s.value}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
            <Link to="/shortlist" className="text-sm font-semibold text-primary hover:underline">Open full shortlist</Link>
            <div className="flex gap-1 rounded-full bg-muted p-1">
              {([["all", "All"], ["great", "Great"], ["good", "Good"], ["low", "Weak"]] as const).map(([k, label]) => (
                <button key={k} onClick={() => setFilter(k)}
                  className={cn("rounded-full px-3 py-1.5 text-sm font-semibold transition-colors",
                    filter === k ? "bg-card text-foreground shadow-soft" : "text-muted-foreground hover:text-foreground")}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          {shown.map((c) => <CandidateCard key={c.id} c={c} rank={candidates.indexOf(c) + 1} />)}
          {shown.length === 0 && <p className="py-6 text-center text-muted-foreground">No candidates in this group.</p>}
          <button onClick={() => { newJob(); window.scrollTo({ top: 0, behavior: "smooth" }); }}
            className="mx-auto flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground">
            <RotateCcw className="size-4" /> Start a new screening
          </button>
        </section>
      )}
    </main>
  );
}
