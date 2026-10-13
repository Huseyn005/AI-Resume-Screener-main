import { createFileRoute, Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { CandidateTable } from "@/components/candidate-table";
import { useScreening } from "@/lib/screening";
import { pageHead } from "@/lib/page-head";

export const Route = createFileRoute("/candidates")({
  head: () => pageHead("All candidates", "Search every candidate across all of your saved jobs."),
  component: CandidatesPage,
});

function CandidatesPage() {
  const { jobs } = useScreening();
  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-10">
      <h1 className="font-display text-4xl font-bold">Candidates</h1>
      {jobs.some((j) => j.candidates.length) ? (
        <CandidateTable allJobs />
      ) : (
        <div className="py-16 text-center">
          <p className="text-muted-foreground">No candidates screened yet.</p>
          <Button asChild className="mt-5 rounded-full"><Link to="/">Screen resumes</Link></Button>
        </div>
      )}
    </main>
  );
}
