FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

    COPY app/ ./app/
COPY indexing/ ./indexing/
COPY infra/ ./infra/

RUN useradd -m -u 1000 raguser \
    && mkdir -p /app/data/faiss_index /app/model_cache /app/raw_dataset \
    && chown -R raguser:raguser /app
USER raguser

ENV TRANSFORMERS_CACHE=/app/model_cache
ENV FAISS_INDEX_PATH=/app/data/faiss_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]