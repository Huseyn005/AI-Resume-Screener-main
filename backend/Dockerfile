FROM python:3.12-slim

ARG TORCH_VERSION=2.9.1
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV HF_HOME=/opt/huggingface
ENV EMBED_MODEL=${EMBED_MODEL}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/backend

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
# GPU training uses requirements-ml.txt; server and CI install CPU torch first.
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

RUN mkdir -p "${HF_HOME}" \
    && python -c "from sentence_transformers import SentenceTransformer; import os; SentenceTransformer(os.environ['EMBED_MODEL'], device='cpu')" \
    && mkdir -p /app/backend/models/finetuned \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app "${HF_HOME}"

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)" || exit 1

CMD ["python", "-m", "app.serve"]
