# Resume Screener Backend

## Setup and run

From the repository root, create and activate the backend virtual environment,
install the pinned dependencies, and copy the environment template:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements-ml.txt
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

By default, `ALLOWED_ORIGINS` includes `https://pushcv.vercel.app`,
`http://localhost:5173`, and `http://localhost:8080`. Override it with a
comma-separated list of allowed browser origins for your deployment.
`EMBED_MODEL` defaults to `sentence-transformers/all-MiniLM-L6-v2`; the model
is downloaded on first use if it is not cached locally.

Start the API from `backend/`:

```powershell
uvicorn app.main:app --port 8000
```

The API exposes `GET /health` and `POST /analyze`. Health reports the loaded
model and whether the semantic calibrator is loaded. The analyze endpoint
accepts a `job_description` form field and one or more PDF files in the
`resumes` form field. Job descriptions must contain 50-10,000 characters.

Score explanations are disabled by default. Set `DEBUG_ENDPOINTS=1` to enable
the additional `debug` field from `POST /analyze?debug=true`; the normal
response remains unchanged when `debug` is omitted or false.

To inspect scores locally without changing API responses, run the diagnostic
tool from the repository root:

```powershell
python backend\tools\explain_scores.py --jd-file path\to\job-description.txt --pdfs "backend\samples\*.pdf"
```

Add `--no-calibrator` to use the linear semantic fallback. The tool reads PDF
files locally and prints extraction, skill, chunk-similarity, and score-formula
details as JSON.

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
