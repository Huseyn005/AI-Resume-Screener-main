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

Set `ALLOWED_ORIGINS` to a comma-separated list of allowed browser origins.
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
