#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$root/backend/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$root/backend/venv/bin/activate"
fi

(
  cd "$root/backend"
  uvicorn app.main:app --port 8000
) &
backend_pid=$!

cleanup() {
  kill "$backend_pid" 2>/dev/null || true
  wait "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

npm --prefix "$root/frontend" run dev
