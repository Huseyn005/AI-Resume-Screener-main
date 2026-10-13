import { useMemo, useState } from "react";
import { Download, Search, SlidersHorizontal, Star, X } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { CandidateDetails } from "@/components/candidate-details";
import { CandidateStatus } from "@/components/candidate-status";
import { initials, useScreening, type Candidate } from "@/lib/screening";
import { cn } from "@/lib/utils";

export function CandidateTable({allJobs=false}: {allJobs?:boolean}) {
 const {jobs,candidates,shortlist,rejected,activeJob}=useScreening();
 const [query,setQuery]=useState("");const [tab,setTab]=useState("all");const [sort,setSort]=useState("score");
 const [min,setMin]=useState(0);const [max,setMax]=useState(100);const [skill,setSkill]=useState("");const [detail,setDetail]=useState<Candidate|null>(null);
 const rows=useMemo(()=>allJobs?jobs.flatMap(j=>j.candidates.map(c=>({...c,job:j.title,demo:j.demo}))):candidates.length?candidates.map(c=>({...c,job:jobs.find(j=>j.id===activeJob)?.title??"",demo:false})):jobs.flatMap(j=>j.candidates.map(c=>({...c,job:j.title,demo:j.demo}))),[jobs,candidates,allJobs,activeJob]);
 const skills=[...new Set(rows.flatMap(c=>c.skills))].sort();
 const shown=rows.filter(c=>(`${c.name} ${c.role} ${c.job}`.toLowerCase().includes(query.toLowerCase()))&&(c.score===null||(c.score>=min&&c.score<=max))&&(!skill||c.skills.includes(skill))&&(tab==="all"||tab==="shortlisted"&&shortlist.includes(c.id)||tab==="rejected"&&rejected.includes(c.id))).sort((a,b)=>sort==="name"?a.name.localeCompare(b.name):(b.score??-1)-(a.score??-1));
 function exportCSV(){
  const cell=(s:string|number)=>`"${String(s).replace(/^[=+@-]/,"'$&").replaceAll('"','""')}"`;
  const csv=[["Candidate","Role","Job","Match score","Skills","Status"],...shown.map(c=>[c.name,c.role,c.job,c.score ?? "",c.skills.join("; "),shortlist.includes(c.id)?"Shortlisted":rejected.includes(c.id)?"Rejected":"Pending"])].map(r=>r.map(cell).join(",")).join("\r\n");
  const url=URL.createObjectURL(new Blob(["\uFEFF",csv],{type:"text/csv;charset=utf-8"}));const a=document.createElement("a");a.href=url;a.download="shortlist.csv";a.click();URL.revokeObjectURL(url);toast.success("CSV exported");
 }
 return <div className="space-y-5">
  {!allJobs&&<div className="flex flex-wrap gap-1 rounded-full bg-muted p-1 w-fit">{["all","shortlisted","rejected"].map(t=><Button key={t} variant="ghost" className={cn("rounded-full capitalize",tab===t&&"bg-card shadow-soft")} onClick={()=>setTab(t)}>{t==="all"?"All candidates":t}</Button>)}</div>}
  <div className="flex flex-wrap gap-3"><div className="relative min-w-40 flex-1"><Search className="absolute left-3 top-3 size-4 text-muted-foreground"/><Input aria-label="Search candidates" placeholder="Search candidates…" value={query} onChange={e=>setQuery(e.target.value)} className="h-10 rounded-full pl-10"/></div>
   <Popover><PopoverTrigger asChild><Button variant="outline" className="h-10 rounded-full"><SlidersHorizontal/>Filters{min>0||max<100||skill?" · active":""}</Button></PopoverTrigger><PopoverContent className="space-y-4 rounded-2xl"><h3 className="font-semibold">Match score</h3><div className="flex gap-3">{[{label:"Minimum score",v:min,set:setMin},{label:"Maximum score",v:max,set:setMax}].map(x=><label key={x.label} className="text-xs">{x.label}<Input type="number" min={0} max={100} value={x.v} onChange={e=>x.set(Math.min(100,Math.max(0,Number(e.target.value))))}/></label>)}</div><label className="block text-sm">Skill<select aria-label="Filter by skill" className="mt-2 w-full rounded-lg border bg-background p-2" value={skill} onChange={e=>setSkill(e.target.value)}><option value="">All skills</option>{skills.map(s=><option key={s}>{s}</option>)}</select></label><Button variant="ghost" onClick={()=>{setMin(0);setMax(100);setSkill("");}}>Reset filters</Button></PopoverContent></Popover>
   <select aria-label="Sort candidates" className="h-10 rounded-full border bg-background px-3 text-sm" value={sort} onChange={e=>setSort(e.target.value)}><option value="score">Highest score</option><option value="name">Name A–Z</option></select><Button disabled={!shown.length} onClick={exportCSV} className="h-10 rounded-full"><Download/>Export CSV</Button>
  </div>
  {rows.some(c=>c.demo)&&<p className="text-xs text-muted-foreground">Includes demo data</p>}
  <div className="overflow-x-auto rounded-2xl border bg-card"><table className="w-full min-w-[680px] text-left text-sm"><thead className="border-b bg-muted/50 text-xs text-muted-foreground"><tr><th className="p-4 font-medium">Candidate</th>{allJobs&&<th className="p-4 font-medium">Job</th>}<th className="p-4 font-medium">Match score</th><th className="p-4 font-medium">Top skills</th><th className="p-4 font-medium">Decision</th><th className="p-4 font-medium"><span className="sr-only">Details</span></th></tr></thead><tbody>{shown.map(c=><Row key={c.id} c={c} allJobs={allJobs} onDetails={()=>setDetail(c)}/>)}</tbody></table>{!shown.length&&<p className="p-10 text-center text-muted-foreground">No candidates match your search.</p>}</div>
  <CandidateDetails candidate={detail} onClose={()=>setDetail(null)}/>
 </div>;
}
function Row({c,allJobs,onDetails}:{c:Candidate&{job:string};allJobs:boolean;onDetails:()=>void}){
 const {shortlist,toggleShortlist,rejected,reject}=useScreening();
 return <tr className="border-b last:border-0 hover:bg-muted/40"><td className="p-4"><div className="flex items-center gap-3"><span className="grid size-10 shrink-0 place-items-center rounded-full bg-secondary text-xs font-bold text-secondary-foreground">{initials(c.name)}</span><div><p className="font-semibold capitalize">{c.name}</p><p className="text-xs text-muted-foreground">{c.role}</p></div></div></td>{allJobs&&<td className="p-4">{c.job}</td>}<td className="min-w-28 p-4">{c.score === null?<CandidateStatus candidate={c}/>:<><p className={cn("mb-2 font-semibold",c.score>=85?"text-success":c.score>=70?"text-foreground":"text-destructive")}>{c.score}%</p><Progress value={c.score} className="h-1"/></>}</td><td className="p-4"><div className="flex max-w-52 flex-wrap gap-1.5">{(c.topSkills ?? c.skills).slice(0,3).map(s=><span key={s} className="rounded-full bg-secondary px-2 py-1 text-xs text-secondary-foreground">{s}</span>)}</div></td><td className="p-4"><div className="flex gap-1"><Button variant="ghost" size="icon" title={shortlist.includes(c.id)?"Remove from shortlist":"Shortlist candidate"} aria-label={`Shortlist ${c.name}`} aria-pressed={shortlist.includes(c.id)} className={cn("rounded-full",shortlist.includes(c.id)&&"bg-accent text-accent-foreground")} onClick={()=>toggleShortlist(c.id)}><Star className={cn(shortlist.includes(c.id)&&"fill-current")}/></Button><Button variant="ghost" size="icon" title={rejected.includes(c.id)?"Restore candidate":"Reject candidate"} aria-label={`Reject ${c.name}`} aria-pressed={rejected.includes(c.id)} className={cn("rounded-full",rejected.includes(c.id)&&"bg-danger-soft text-destructive")} onClick={()=>reject(c.id)}><X/></Button></div></td><td className="p-4"><Button variant="link" onClick={onDetails}>View details</Button></td></tr>;
}