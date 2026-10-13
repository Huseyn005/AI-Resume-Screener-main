import httpx
from pathlib import Path
JD = ("We need a Machine Learning Engineer with Python, PyTorch, NLP "
      "and FastAPI experience to build and deploy ranking models.")
fs = [("resumes", (p.name, p.read_bytes(), "application/pdf"))
      for p in sorted(Path("samples").glob("*.pdf"))]
r = httpx.post("http://localhost:8000/analyze", data={
    "job_description": JD}, files=fs, timeout=300)
print(r.status_code)
for x in r.json()["results"]:
  print(x["match_percentage"], x["filename"], x["status"],
        x["flags"], x["top_skills"])