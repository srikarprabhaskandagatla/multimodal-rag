# ── Why this base image? ──────────────────────────────────────────────────────
# python:3.11-slim-bookworm is the smallest Debian image with Python 3.11.
# 3.11 specifically: asyncio is ~25% faster than 3.10 due to the task scheduler
# rewrite. We use slim (not alpine) because FAISS requires glibc — musl libc
# (alpine) causes undefined symbol errors at runtime with faiss-cpu wheels.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

# System deps needed by Pillow (libjpeg), psycopg2 (libpq), and torch (libgomp)
# We install only what's strictly required to keep the image lean (~900MB total)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first — Docker layer caching means pip install only reruns
# when requirements.txt changes, not on every source code change
COPY requirements.txt .

# Why --no-cache-dir? Unity clusters have limited ephemeral disk; pip's cache
# would waste ~300MB. Why --extra-index-url? PyTorch's official index is
# required for the correct CUDA/CPU wheel selection.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY indexing/ ./indexing/
COPY infra/ ./infra/

# Non-root user: Unity clusters enforce security policies that reject root-owned
# processes in containers. UID 1000 is the safe default.
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# FAISS index and model cache live on a mounted volume (see docker-compose.yml)
# so they persist across container restarts and don't bloat the image
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV FAISS_INDEX_PATH=/app/data/faiss_index

EXPOSE 8000

# Why uvicorn directly instead of gunicorn + uvicorn workers?
# On Unity EC2 (typically c5.2xlarge, 8 vCPUs), a single uvicorn process with
# --workers=4 gives better memory isolation than gunicorn's forking model,
# which would duplicate FAISS index memory (4GB+) per worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]