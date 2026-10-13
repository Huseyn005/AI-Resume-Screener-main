import httpx, json
from pathlib import Path
JD = ("We need a Machine Learning Engineer with Python, PyTorch, NLP "
      "and FastAPI experience to build and deploy ranking models.")
p = Path("samples/ml_engineer.pdf")
f = [("resumes", (p.name, p.read_bytes(), "application/pdf"))]
r = httpx.post("http://localhost:8000/analyze",
               data={"job_description": JD}, files=f, timeout=300)
x = r.json()["results"][0]
keys = ("score_breakdown", "matched", "missing_skills", "reason")
print(json.dumps({k: x[k] for k in keys}, indent=1))