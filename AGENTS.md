# Agent and developer guidance

## Repository layout

- `src/`, `public/`, and the Vite configuration contain the React SPA.
- `backend/` contains the FastAPI service. See [backend/README.md](backend/README.md)
  for setup and test instructions.
- `docs/` contains architecture, deployment, and frontend notes.

## Commands

Run frontend commands from the repository root:

```sh
npm ci
npm run dev
npm run lint
npm test
npm run build
```

Vite uses its default development port, 5173.

Run backend commands from `backend/`:

```sh
python -m pytest
uvicorn app.main:app --port 8000
```

## Conventions and safety

- Frontend: follow the existing Prettier and ESLint configuration.
- Backend: follow PEP 8 and keep lines within 79 characters.
- Preserve the API contract: `POST {VITE_API_URL}/analyze`, multipart fields
  `job_description` and `resumes`, and the response type in `src/lib/api.ts`.
- Never commit real CVs or anything under `backend/samples/`.
- Do not force-push shared branches.
- Ask before touching `backend/` when a task is frontend-only.
