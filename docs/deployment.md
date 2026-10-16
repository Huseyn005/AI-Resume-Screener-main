# Deployment

## Vercel frontend

The root [`vercel.json`](../vercel.json) keeps the repository root as the
Vercel project directory and configures these commands:

```text
installCommand: npm --prefix frontend ci
buildCommand: npm --prefix frontend run build
outputDirectory: frontend/dist
```

It also rewrites client-side routes to `/index.html`. Configure
`VITE_API_URL` in the Vercel Preview and Production environments as appropriate;
Vite embeds this value at build time, so redeploy after changing it.

If the preview build does not work with the root project configuration, use
this dashboard fallback:

1. Set the Vercel project's **Root Directory** to `frontend`.
2. Keep **Include source files outside of the Root Directory** disabled.
3. Use `npm ci` for Install Command and `npm run build` for Build Command, with
   `dist` as the output directory.
4. Keep `frontend/vercel.json` in place for the SPA rewrite.

## FastAPI backend

Build and run the CPU-only image from the repository root:

```powershell
Copy-Item backend\.env.example backend\.env
docker build -t shortlist-backend backend
docker run --rm -p 8000:8000 --env-file backend\.env shortlist-backend
```

The image pre-downloads `EMBED_MODEL` and runs offline; the default is
`sentence-transformers/all-MiniLM-L6-v2`. To bake a fine-tuned model from the
Hugging Face Hub, pass `--build-arg EMBED_MODEL=owner/model-name`. For local
fine-tuned weights, include `backend/models/finetuned/` in the build context
(remove its `.dockerignore` exclusion) and add
`COPY models/finetuned/ ./models/finetuned/` to the Dockerfile. Do not copy
models by default. Local GPU training uses `requirements-ml.txt` followed by
`requirements.txt`; the server and CI use CPU torch 2.9.1 followed by
`requirements.txt`.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `https://pushcv.vercel.app,https://statcv.vercel.app,https://pullcv.vercel.app,https://taphire.vercel.app,https://dragcv.vercel.app,http://localhost:5173,http://localhost:8080` | Comma-separated exact browser origins |
| `ALLOWED_ORIGIN_REGEX` | empty | Optional origin regex, e.g. `^https://pushcv(-[a-z0-9-]+)*\.vercel\.app$` |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model baked into the image |
| `DEBUG_ENDPOINTS` | `0` | Enables optional score explanations |
| `MAX_PDF_BYTES` | `10485760` | Maximum bytes per PDF (10 MiB) |
| `MAX_PDF_PAGES` | `20` | Maximum pages per PDF |
| `MAX_REQUEST_BYTES` | `62914560` | Maximum request size (60 MiB) |
| `MAX_FILES` | `30` | Maximum PDFs per request |
| `MAX_CONCURRENT_ANALYSES` | `2` | Active requests per process |
| `QUEUE_TIMEOUT_SECONDS` | `30` | Wait time for an analysis slot |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-client analyze limit; `0` disables it |
| `LOG_LEVEL` | `INFO` | Python logging level |

For production, set `ALLOWED_ORIGINS` to the actual frontend origin(s) that
should call the API.

One Uvicorn process is intentional: every worker would load another embedding
model and copy of the 10k-skill lexicon, multiplying memory use. Host the
container on a Docker-capable service with enough RAM for CPU torch, MiniLM,
and request processing; terminate TLS with HTTPS, provide the environment
variables, and configure the platform health probe at `/health`. Allow the
image's 180-second Docker health-check start period for loading the model and
lexicon.
Use a keep-alive or minimum-instance option if cold starts are undesirable.

Set `VITE_API_URL` in the Vercel Production and Preview environments to the
public HTTPS API base URL, then redeploy because Vite embeds it at build time.
The active Vercel configuration is the root [`vercel.json`](../vercel.json)
when the Vercel Root Directory is the repository root. Keep
[`frontend/vercel.json`](../frontend/vercel.json) as the dashboard fallback
when the Root Directory is `frontend`; do not delete either file.

Check a live API with:

```powershell
python backend\tools\smoke_test.py --url https://api.example.com
```

See [local testing with ngrok](local-testing-with-ngrok.md) for temporary
preview testing.
