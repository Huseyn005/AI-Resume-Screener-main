import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { useScreening, type Weights } from "@/lib/screening";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});

function SettingsPage() {
  const { profile, setProfile, weights, setWeights, clearData } = useScreening();
  const [name, setName] = useState(profile.name);
  const [email, setEmail] = useState(profile.email);
  const [draft, setDraft] = useState(weights);
  useEffect(() => { setName(profile.name); setEmail(profile.email); }, [profile]);
  useEffect(() => setDraft(weights), [weights]);
  const total = draft.skills + draft.experience + draft.education;

  return (
    <main className="mx-auto max-w-3xl space-y-8 px-4 py-10">
      <h1 className="font-display text-4xl font-bold">Settings</h1>
      <form className="space-y-8" onSubmit={(e) => {
        e.preventDefault();
        if (total !== 100) { toast.error("Scoring weights must add up to 100%."); return; }
        setProfile({ name: name.trim(), email: email.trim() }); setWeights(draft); toast.success("Settings saved");
      }}>
        <section className="space-y-4 border-b pb-8">
          <h2 className="font-display text-xl font-semibold">Profile</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-sm">Name<Input value={name} onChange={(e) => setName(e.target.value)} maxLength={100} placeholder="Your name" className="mt-2 rounded-xl" /></label>
            <label className="text-sm">Email<Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} maxLength={255} placeholder="you@company.com" className="mt-2 rounded-xl" /></label>
          </div>
        </section>
        <section className="space-y-6 border-b pb-8">
          <h2 className="font-display text-xl font-semibold">Scoring weights</h2>
          {(Object.keys(draft) as (keyof Weights)[]).map((key) => (
            <div key={key}>
              <div className="mb-3 flex justify-between text-sm"><label htmlFor={`w-${key}`} className="capitalize">{key}</label><span>{draft[key]}%</span></div>
              <Slider id={`w-${key}`} aria-label={`${key} weight`} min={0} max={100} step={5} value={[draft[key]]} onValueChange={(v) => setDraft((p) => ({ ...p, [key]: v[0] ?? 0 }))} />
            </div>
          ))}
          <p className="text-sm text-muted-foreground">Total {total}%{total !== 100 ? " · must equal 100%" : ""}. Applies the next time you check resumes.</p>
        </section>
        <Button className="rounded-full" disabled={total !== 100}>Save changes</Button>
      </form>
      <section className="space-y-4 border-t pt-8">
        <h2 className="font-display text-xl font-semibold">Data</h2>
        <p className="text-sm text-muted-foreground">Everything is stored in this browser only.</p>
        <AlertDialog>
          <AlertDialogTrigger asChild><Button variant="outline" className="rounded-full text-destructive"><Trash2 />Clear all data</Button></AlertDialogTrigger>
          <AlertDialogContent className="rounded-3xl">
            <AlertDialogTitle>Clear all data?</AlertDialogTitle>
            <AlertDialogDescription>All jobs, candidates and resumes will be permanently removed.</AlertDialogDescription>
            <AlertDialogFooter>
              <AlertDialogCancel className="rounded-full">Cancel</AlertDialogCancel>
              <AlertDialogAction className="rounded-full" onClick={() => void clearData()}>Clear data</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </section>
    </main>
  );
}
