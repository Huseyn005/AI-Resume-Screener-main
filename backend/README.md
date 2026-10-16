# Resume Screener Backend

## Setup and run

From the repository root, create and activate the backend virtual environment,
install dependencies, and copy the environment template. For local GPU
training, install `requirements-ml.txt` before `requirements.txt`; server and
CI environments use CPU-only torch before `requirements.txt` instead:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt  # local GPU/training
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For a CPU-only server or CI environment, replace the two install commands
above with:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

By default, `ALLOWED_ORIGINS` includes `https://pushcv.vercel.app`,
`https://statcv.vercel.app`, `https://pullcv.vercel.app`,
`https://taphire.vercel.app`, `https://dragcv.vercel.app`,
`http://localhost:5173`, and `http://localhost:8080`. Override it with a
comma-separated list of allowed browser origins for your deployment.
`ALLOWED_ORIGIN_REGEX` defaults to empty and can allow Vercel previews, for
example `^https://pushcv(-[a-z0-9-]+)*\.vercel\.app$`.
`EMBED_MODEL` defaults to `sentence-transformers/all-MiniLM-L6-v2`; the model
is baked into the Docker image at build time and used offline at runtime.
Resource and rate-limit defaults are documented in
[docs/deployment.md](../docs/deployment.md).

Start the API from `backend/`:

```powershell
uvicorn app.main:app --port 8000
```

The API exposes `GET /health` and `POST /analyze`. Health reports the loaded
model and whether the semantic calibrator is loaded. The analyze endpoint
accepts a `job_description` form field and one or more PDF files in the
`resumes` form field. Job descriptions must contain 50-10,000 characters.
Container deployments bake the configured `EMBED_MODEL` into the image for
offline cold starts.

Score explanations are disabled by default. Set `DEBUG_ENDPOINTS=1` to enable
the additional `debug` field from `POST /analyze?debug=true`; the normal
response remains unchanged when `debug` is omitted or false.

Resume headings are normalized from common experience, project, education,
skills, summary, and extras labels. When experience/projects are absent,
section fit compares the job description with the whole resume rather than
returning zero solely due to a missing section. PDF extraction currently
flattens positioned text to lines before section parsing, so section assignment
in multi-column PDFs follows extracted text order and is not guaranteed to
follow the nearest heading in the same geometric column.

Technical skill coverage discounts skills listed only in skills sections
(0.6) and summary, education, or other sections (0.3); experience and project
evidence receives full credit. Skills marked preferred, bonus, nice-to-have,
or familiarity receive half weight. The default composite weights are 0.5,
0.4, and 0.1 for semantic similarity, skill coverage, and section fit.
When a job description has fewer than four recognized hard skills, soft
skills contribute at half weight; if the combined signal is still too small,
the coverage weight is redistributed to semantic and section fit. A job with
at least four hard skills and no hard-skill match receives an order-preserving
0.6 multiplier instead of a flat score cap.

## Skill lexicon maintenance

The checked-in skill lexicon combines the legacy list, curated domain lists,
and filtered O*NET examples. O*NET attribution is in
`data/skills/ATTRIBUTION.md`; the downloaded source archive is kept under the
ignored `samples/onet/` directory and is not committed.

Build-time tools use `wordfreq` from `requirements-dev.txt`; it is not a
runtime dependency:

```powershell
python -m pip install -r requirements-dev.txt
python tools\clean_lexicon.py
python tools\build_onet_lexicon.py --onet-dir samples\onet --release 31.0
```

The O*NET builder can be rerun with another extracted release by passing its
folder through `--onet-dir`. It discovers tables from their headers rather
than fixed release-specific filenames.

To inspect scores locally without changing API responses, run the diagnostic
tool from the repository root:

```powershell
python backend\tools\explain_scores.py --jd-file path\to\job-description.txt --pdfs "backend\samples\*.pdf"
```

Add `--no-calibrator` to use the linear semantic fallback. The tool reads PDF
files locally and prints extraction, skill, chunk-similarity, and score-formula
details as JSON. Add `--out path\to\scores.json` to write the details for all
matched PDFs as a UTF-8 JSON array instead of printing them.

## Analyze result schema

Each item in the `results` array contains exactly these fields:

| Field | Description |
| --- | --- |
| `filename` | Uploaded PDF filename |
| `candidate_name` | Name detected in the resume or filename |
| `match_percentage` | Combined match score from 0 to 100 |
| `top_skills` | Up to three matched job skills |
| `missing_skills` | Up to three required skills not found |
| `matched` | All matched job skills |
| `skills` | Skills identified in the resume |
| `score_breakdown` | `semantic`, `skills`, and `section_fit` scores |
| `reason` | Two-sentence explanation of the score |
| `flags` | Extraction or screening flags |
| `status` | Screening status |
| `role` | Role inferred from the resume |
| `experience` | Experience duration, when stated |
| `strengths` | Brief strengths summary |
| `gaps` | Brief gaps summary |
| `summary` | Short explanation summary |

The response does not include duplicate `breakdown` or `missing` fields.

## Tests

Run the backend tests from this directory:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## ML data, training, and evaluation

Place the public Kaggle Resume Dataset at `ml/data/Resume.csv`. The source CSV,
generated split data, and trained models are ignored by git. From `backend/`,
run the pipeline in this order:

```powershell
python -m ml.prepare_data
python -m ml.train --max-pairs 200 --epochs 1
python -m ml.train --epochs 2 --batch-size 16 --lr 2e-5 --max-pairs 3000
python -m ml.evaluate
python -m ml.calibrate
python -m pytest
```

The first training command is a fast smoke test; the second performs the
configured training run. Evaluation writes `ml/results.json`,
`ml/REPORT.md`, and `ml/score_distribution.png`. Training and calibration
write `models/finetuned/` and `models/calibrator.pkl`, which the API loads
automatically when present.
