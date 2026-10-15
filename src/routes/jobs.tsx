import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Copy, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { average, useScreening } from "@/lib/screening";

export const Route = createFileRoute("/jobs")({
  component: JobsPage,
});

function JobsPage() {
  const { jobs, newJob, openJob, duplicateJob, deleteJob } = useScreening();
  const navigate = useNavigate();
  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <div className="flex items-center justify-between gap-4">
        <h1 className="font-display text-4xl font-bold">Jobs</h1>
        <Button asChild className="rounded-full">
          <Link to="/" onClick={newJob}>
            <Plus />
            New job
          </Link>
        </Button>
      </div>
      {!jobs.length ? (
        <p className="py-16 text-center text-muted-foreground">
          No saved jobs yet. Use New job to start your first screening.
        </p>
      ) : (
        <div className="grid gap-5 sm:grid-cols-2">
          {jobs.map((j) => (
            <article
              key={j.id}
              className="cursor-pointer rounded-2xl border bg-card p-6 shadow-soft transition-transform hover:-translate-y-0.5"
              onClick={() => {
                openJob(j.id);
                void navigate({ to: "/shortlist" });
              }}
            >
              <div className="flex justify-between gap-3">
                <div>
                  <h2 className="font-display text-xl font-semibold">{j.title}</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {new Date(j.date).toLocaleDateString("en-GB", {
                      day: "numeric",
                      month: "short",
                      year: "numeric",
                    })}{" "}
                    · {j.employment}
                  </p>
                </div>
                <div className="flex" onClick={(e) => e.stopPropagation()}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-full"
                    aria-label={`Duplicate ${j.title}`}
                    onClick={() => duplicateJob(j.id)}
                  >
                    <Copy />
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="rounded-full"
                        aria-label={`Delete ${j.title}`}
                      >
                        <Trash2 />
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent className="rounded-3xl">
                      <AlertDialogTitle>Delete {j.title}?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This removes the job and its screening results.
                      </AlertDialogDescription>
                      <AlertDialogFooter>
                        <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          className="rounded-full"
                          onClick={() => void deleteJob(j.id)}
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </div>
              <div className="mt-6 flex gap-8 text-sm">
                <span>
                  <strong className="font-display text-xl">{j.candidates.length}</strong>{" "}
                  <span className="text-muted-foreground">screened</span>
                </span>
                <span>
                  <strong className="font-display text-xl">
                    {j.candidates.some((c) => c.score !== null) ? `${average(j.candidates)}%` : "—"}
                  </strong>{" "}
                  <span className="text-muted-foreground">average match</span>
                </span>
              </div>
              {j.demo && <p className="mt-4 text-xs text-muted-foreground">Demo data</p>}
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
