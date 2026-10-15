// Client-side resume parsing and matching against a job description.
import type { Weights } from "../screening";

const SKILLS = [
  "Figma",
  "Sketch",
  "UX Research",
  "UI Design",
  "Design Systems",
  "Prototyping",
  "Wireframing",
  "Interaction Design",
  "Visual Design",
  "Branding",
  "Illustration",
  "Adobe CC",
  "Photoshop",
  "Illustrator",
  "Usability Testing",
  "JavaScript",
  "TypeScript",
  "React",
  "Next.js",
  "Vue",
  "Angular",
  "Node.js",
  "Python",
  "Java",
  "Go",
  "Rust",
  "C#",
  "C++",
  "PHP",
  "Ruby",
  "Swift",
  "Kotlin",
  "SQL",
  "PostgreSQL",
  "MySQL",
  "MongoDB",
  "GraphQL",
  "REST",
  "AWS",
  "Azure",
  "GCP",
  "Docker",
  "Kubernetes",
  "Terraform",
  "CI/CD",
  "Git",
  "Linux",
  "HTML",
  "CSS",
  "Tailwind",
  "Machine Learning",
  "Data Analysis",
  "Excel",
  "Tableau",
  "Power BI",
  "Statistics",
  "Product Management",
  "Agile",
  "Scrum",
  "Jira",
  "Roadmapping",
  "Stakeholder Management",
  "Marketing",
  "SEO",
  "Content Strategy",
  "Copywriting",
  "Sales",
  "CRM",
  "Salesforce",
  "Customer Success",
  "Project Management",
  "Leadership",
  "Communication",
  "Accounting",
  "Finance",
  "Recruiting",
];

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const has = (text: string, skill: string) =>
  new RegExp(`(^|[^a-z0-9])${esc(skill.toLowerCase())}([^a-z0-9]|$)`).test(text);

export const findSkills = (text: string) => {
  const t = text.toLowerCase();
  return SKILLS.filter((s) => has(t, s));
};

export async function extractPdfText(file: Blob): Promise<string> {
  try {
    const pdfjs = await import("pdfjs-dist");
    const worker = (await import("pdfjs-dist/build/pdf.worker.min.mjs?url")).default;
    pdfjs.GlobalWorkerOptions.workerSrc = worker;
    const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
    const pages: string[] = [];
    for (let i = 1; i <= Math.min(doc.numPages, 10); i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      const lines: string[] = [];
      let last: number | null = null;
      for (const item of content.items as { str: string; transform: number[] }[]) {
        const y = item.transform[5] ?? 0;
        if (last !== null && Math.abs(y - last) > 2) lines.push("\n");
        lines.push(item.str);
        last = y;
      }
      pages.push(lines.join(" "));
    }
    return pages.join("\n");
  } catch {
    return "";
  }
}

const years = (text: string) => {
  const m = [...text.matchAll(/(\d{1,2})\s*\+?\s*(?:years|yrs)/gi)].map((x) => Number(x[1]));
  return m.length ? Math.max(...m.filter((n) => n < 45)) : 0;
};

const education = (text: string): [number, string] => {
  const t = text.toLowerCase();
  if (/ph\.?d|doctorate/.test(t)) return [100, "Doctorate"];
  if (/master|m\.sc|msc|mba|m\.a\b/.test(t)) return [90, "Master's degree"];
  if (/bachelor|b\.sc|bsc|b\.a\b|b\.s\b|university|college/.test(t))
    return [75, "Bachelor's degree"];
  return [35, "No degree found"];
};

const guessName = (text: string, fileName: string) => {
  const first = text
    .split("\n")
    .map((l) => l.trim())
    .find((l) => /^[A-Za-zÀ-ž'.-]+(\s+[A-Za-zÀ-ž'.-]+){1,3}$/.test(l) && l.length < 40);
  if (first) return first.replace(/\s+/g, " ");
  return (
    fileName
      .replace(/\.pdf$/i, "")
      .replace(/[_-]+/g, " ")
      .replace(/\b(resume|cv)\b/gi, "")
      .trim() || "Unnamed candidate"
  );
};

const guessRole = (text: string) => {
  const line = text
    .split("\n")
    .map((l) => l.trim())
    .find(
      (l) =>
        l.length < 60 &&
        /(designer|engineer|developer|manager|analyst|scientist|lead|director|specialist|consultant|architect|marketer|recruiter|accountant)/i.test(
          l,
        ),
    );
  return line?.replace(/\s+/g, " ") ?? "Role not listed";
};

export type Scored = {
  name: string;
  role: string;
  score: number;
  skills: string[];
  matched: string[];
  missing: string[];
  experience: string;
  summary: string;
  strengths: string[];
  gaps: string[];
  breakdown: { skills: number; experience: number; education: number };
};

export function scoreResume(text: string, fileName: string, jd: string, w: Weights): Scored {
  const required = findSkills(jd);
  const found = findSkills(text);
  const matched = required.filter((s) => found.includes(s));
  const missing = required.filter((s) => !found.includes(s));
  const needYears = years(jd) || 3;
  const yrs = years(text);
  const [edu, eduLabel] = education(text);
  const skillsScore = required.length
    ? Math.round((matched.length / required.length) * 100)
    : Math.min(100, found.length * 15);
  const expScore = Math.min(100, Math.round((yrs / needYears) * 100));
  const total = w.skills + w.experience + w.education || 1;
  const score = text
    ? Math.round((skillsScore * w.skills + expScore * w.experience + edu * w.education) / total)
    : 0;
  const skills = [...matched, ...found.filter((s) => !matched.includes(s))];
  const strengths: string[] = [];
  const gaps: string[] = [];
  if (matched.length)
    strengths.push(
      `Covers ${matched.length} of ${required.length || matched.length} required skills`,
    );
  if (yrs >= needYears) strengths.push(`${yrs} years of experience (role asks for ${needYears})`);
  if (edu >= 75) strengths.push(eduLabel);
  if (missing.length)
    gaps.push(`Missing ${missing.slice(0, 3).join(", ")}${missing.length > 3 ? "…" : ""}`);
  if (yrs < needYears)
    gaps.push(
      yrs
        ? `${yrs} years of experience, role asks for ${needYears}`
        : "Years of experience not stated",
    );
  if (edu < 75) gaps.push("No degree mentioned");
  if (!text) gaps.unshift("Text could not be read from this PDF");
  const summary = !text
    ? "We couldn't read text from this PDF — it may be a scanned image. Open the resume to review it manually."
    : `${yrs ? `${yrs} years of experience` : "Experience not stated"}${matched.length ? ` with ${matched.slice(0, 2).join(" and ")}` : ""}. ${
        score >= 85
          ? "Very close to what this role needs."
          : score >= 70
            ? "A solid match worth a conversation."
            : "Some gaps against the key requirements."
      }`;
  return {
    name: guessName(text, fileName),
    role: guessRole(text),
    score,
    skills,
    matched,
    missing,
    experience: yrs ? `${yrs} years` : "Not stated",
    summary,
    strengths,
    gaps,
    breakdown: { skills: skillsScore, experience: expScore, education: edu },
  };
}
