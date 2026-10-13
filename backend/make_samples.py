import pymupdf
from pathlib import Path
R = {
  "ml_engineer.pdf": "Jane Doe\nML Engineer\nSkills: Python, PyTorch, "
  "scikit-learn, NLP, FastAPI, Docker, SQL\nExperience\nBuilt NLP ranking "
  "models with transformers and deployed FastAPI services in Docker.\n"
  "Education\nBSc Computer Science. " * 3,
  "frontend_dev.pdf": "Sam Lee\nFrontend Developer\nSkills: React, "
  "TypeScript, CSS, Figma, Tailwind\nExperience\nBuilt dashboards and "
  "design systems in React. Education\nBA Design. " * 3,
  "chef.pdf": "Maria Rossi\nHead Chef\nSkills: Menu planning, Food "
  "safety, Pastry, Kitchen management\nExperience\nRan a 40-seat "
  "restaurant kitchen. Education\nCulinary diploma. " * 3,
}
out = Path("samples")
out.mkdir(exist_ok=True)
for name, text in R.items():
  d = pymupdf.open()
  d.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 790), text)
  d.save(str(out / name))
d = pymupdf.open()
p = d.new_page()
p.insert_textbox(pymupdf.Rect(50, 50, 550, 300), R["chef.pdf"])
p.insert_text((50, 400), "python pytorch nlp fastapi docker sql",
              fontsize=11, color=(1, 1, 1))
d.save(str(out / "hidden_text.pdf"))
d = pymupdf.open()
d.new_page()
d.save(str(out / "blank_scanned.pdf"))
Path(out / "notes.txt").write_text("not a pdf")