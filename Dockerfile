# Two images from one file (phase 6.2):
#
#   docker build --target api    -t sawti-api .     # FastAPI: no torch, no sentence-transformers
#   docker build --target worker -t sawti-worker .  # Celery: the graph + the embedding model
#
# Both install exactly the versions in uv.lock. The api image prunes
# sentence-transformers and everything only it needs (torch, transformers,
# scipy, ...) — the API never runs the graph or embeds anything
# (tests/api/test_main.py checks it does not even import them).
#
# Built on linux/arm64 (Docker on Apple Silicon), where torch's PyPI wheel is
# CPU-only and uv.lock's CUDA packages are excluded by their x86_64 markers.
# An x86_64 build would pull the CUDA wheels into the worker image.

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN uv venv /opt/venv
COPY --chmod=644 pyproject.toml uv.lock README.md alembic.ini ./

# --- api ---------------------------------------------------------------------
FROM base AS api
RUN uv export --frozen --no-dev --no-hashes --no-emit-project --prune sentence-transformers \
        --format requirements.txt -o /tmp/requirements.txt \
    && uv pip install --no-deps -r /tmp/requirements.txt
COPY src ./src
RUN uv pip install --no-deps . && rm -rf src
RUN useradd --create-home sawti
USER sawti
EXPOSE 8000
ENV SAWTI_API_HOST=0.0.0.0 SAWTI_API_PORT=8000
CMD ["sawti"]

# --- worker --------------------------------------------------------------------
FROM base AS worker
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
        --format requirements.txt -o /tmp/requirements.txt \
    && uv pip install --no-deps -r /tmp/requirements.txt
RUN useradd --create-home sawti
USER sawti
# Bake the embedding model into the image: cold start is then a load from
# local disk, not a download, and the worker needs no network for it.
ENV HF_HOME=/home/sawti/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
COPY --chown=sawti src ./src
USER root
RUN uv pip install --no-deps . && rm -rf src
USER sawti
ENV HF_HUB_OFFLINE=1
CMD ["celery", "-A", "sawti.api.tasks", "worker", "--loglevel=INFO", "--concurrency=1"]
