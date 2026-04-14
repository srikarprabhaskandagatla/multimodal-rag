<h1 align="center">
  <br>
  Multimodal RAG
  <br>
</h1>

<h4 align="center">A production-ready RAG system that searches a 50,000-document corpus using both text and images — embedded with CLIP, retrieved via FAISS HNSW, and answered by a LangChain agent backed by Azure OpenAI GPT-4o.</h4>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://openai.com/">
    <img src="https://img.shields.io/badge/-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" alt="GPT-4o">
  </a>
  <a href="https://github.com/facebookresearch/faiss">
    <img src="https://img.shields.io/badge/-FAISS%20HNSW-4267B2?style=flat-square&logo=meta&logoColor=white" alt="FAISS">
  </a>
  <a href="https://huggingface.co/openai/clip-vit-base-patch32">
    <img src="https://img.shields.io/badge/-CLIP%20ViT--B%2F32-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="CLIP">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  </a>
  <a href="https://redis.io/">
    <img src="https://img.shields.io/badge/-Redis-DC382D?style=flat-square&logo=redis&logoColor=white" alt="Redis">
  </a>
  <a href="https://www.postgresql.org/">
    <img src="https://img.shields.io/badge/-PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL">
  </a>
</p>

<p align="center">
  <a href="#how-it-works">How It Works</a>
  •
  <a href="#repository-structure">Structure</a>
  •
  <a href="#design-choices">Design Choices</a>
  •
  <a href="#setup">Setup</a>
  •
  <a href="#indexing">Indexing</a>
  •
  <a href="#running-the-app">Running</a>
  •
  <a href="#api-endpoints">API</a>
</p>

---

## How It Works

### 1. Query Intake — FastAPI

The user sends a request to one of two endpoints: `/query/text` for a text-only query, or `/query/multimodal` for a query that includes an image. FastAPI validates the request with Pydantic and passes it to the LangChain agent.

### 2. Routing — LangChain Agent with GPT-4o

The agent receives the query and uses GPT-4o's function-calling API to decide which retrieval tool to invoke:
- Text-only query → `text_retriever_tool`
- Image with no text → `image_retriever_tool`
- Both text and image → `multimodal_retriever_tool`

Function calling is used instead of prompt-based routing because it forces the model to emit a structured JSON tool name rather than free text — making routing deterministic and schema-validated.

### 3. Embedding — CLIP ViT-B/32

The selected tool calls `embed_query()`, which runs the query through CLIP (Contrastive Language-Image Pretraining). CLIP encodes both text and images into the same 512-dimensional vector space, which is the property that makes cross-modal retrieval possible.

- Text input → CLIP text encoder → 512-dim vector
- Image input → CLIP image encoder → 512-dim vector
- Both → average of the two vectors, re-normalized

CLIP runs locally via HuggingFace, loaded once per uvicorn worker process and cached in memory for its lifetime.

### 4. Cache Check — Redis

Before hitting FAISS, the system checks Redis for a cached result. The cache key is `rag:{SHA-256(query)[:16]}:{has_image}`. On a cache hit, the full result is returned immediately. On a miss, results are stored in Redis after retrieval with a 1-hour TTL.

### 5. Vector Search — FAISS HNSW

The 512-dim query vector is searched against the pre-built FAISS index using HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search. HNSW returns the top-20 most similar document IDs in under 1ms.

The FAISS index lives in-process — no network hop. Each document in the 50k corpus was embedded at index-build time and stored as an L2-normalized float32 vector.

### 6. Metadata Fetch — PostgreSQL

FAISS returns integer IDs. These are mapped back to `doc_id` strings via `id_map.pkl`, then a single `WHERE doc_id = ANY(...)` query fetches full metadata (title, source, doc_type, content preview, flexible JSONB metadata) from PostgreSQL.

### 7. Answer Synthesis — GPT-4o

The retrieved documents are passed back to the LangChain agent, which synthesizes a grounded natural-language answer citing the relevant `doc_id` and source fields. The agent never fabricates information not present in the retrieved documents.

---

## Repository Structure

```
multimodal-rag/
├── app/
│   ├── main.py          # FastAPI entrypoint, endpoints, lifespan
│   ├── agent.py         # LangChain agent + tool definitions
│   ├── retriever.py     # FAISS HNSW wrapper, singleton loader
│   ├── embeddings.py    # CLIP text/image embedding, async wrappers
│   ├── cache.py         # Redis query cache (SHA-256 keyed, 1h TTL)
│   └── db.py            # PostgreSQL async metadata store (asyncpg/SQLAlchemy)
│
├── indexing/
│   ├── build_index.py       # CPU indexer — used by Docker (single container)
│   ├── build_index_gpu.py   # GPU indexer — used on HPC (SLURM sbatch)
│   ├── insert_postgres.py   # Standalone DB insert — use after GPU indexing
│   └── slurm_index.sh       # SLURM job script for HPC GPU node
│
├── infra/
│   ├── nginx.conf           # Reverse proxy, rate limiting, TLS termination
│   └── postgre_init.sql     # Schema: documents table, JSONB + GIN indexes
│
├── raw_dataset/
│   ├── data/
│   │   ├── corpus.jsonl     # 50k documents (text + image_url/image_path)
│   │   └── images/          # Local image files (if any)
│   ├── faiss_index/         # Output: index.faiss + id_map.pkl (bind-mounted)
│   └── logs/                # Indexer logs
│
├── data/
│   └── prepare_dataset.py   # Dataset preparation script
│
├── tests/
│   └── test_retriever.py
│
├── docker-compose.yaml          # Main app stack (app + postgres + redis + nginx)
├── docker-compose.indexer.yaml  # One-time indexer (run before main stack)
├── Dockerfile
├── requirements.txt
└── .env
```

---

## Design Choices

### Why CLIP for embeddings?

CLIP (Contrastive Language-Image Pretraining) is the only off-the-shelf model that maps **both text and images into the same vector space**. This is what makes cross-modal retrieval work: the text query "a dog running on a beach" and a photo of that scene will be geometrically close in CLIP space. Text-only models (SBERT, OpenAI text-embedding-3) have no image tower and are disqualified for this use case.

We use `openai/clip-vit-base-patch32` via HuggingFace — runs locally, no per-call API cost, weights cached on first load.

### Why FAISS HNSW (not Pinecone, pgvector, or Chroma)?

| Option | Latency | Why not |
|---|---|---|
| FAISS HNSW (in-process) | ~1ms | **Used** — zero network hop |
| Pinecone / Weaviate | ~50-100ms | Network RTT + egress cost |
| pgvector | ~10-30ms | Sequential scan fallback at 50k scale |
| Chroma | ~5ms | Single-threaded, no production ANN |
| Milvus | ~2ms | Requires separate cluster deployment |

HNSW gives ~99% recall at O(log n) query time. For 50k × 512-dim vectors it fits in ~200MB RAM and answers queries in under 1ms.

### Why LangChain agent with function calling?

The agent routes queries to one of three tools based on modality. Function calling (OpenAI tool_use API) forces the model to emit structured JSON with a typed tool name — no fragile ReAct prompt parsing. GPT-4o misroutes <5% of queries; GPT-3.5-turbo misroutes ~18%.

### Why FastAPI (not Flask or Django)?

FastAPI is async-first. CLIP inference, FAISS search, Redis, and PostgreSQL all benefit from non-blocking I/O. Flask blocks the worker thread on every operation; Django's ORM doesn't support asyncpg natively.

### Why Redis for caching?

With 4 uvicorn workers, an in-process dict gives ~25% cache hit rate (each worker has its own dict). Redis is shared across all workers via TCP, supports per-key TTL, and persists across restarts.

### Why PostgreSQL (not SQLite or MongoDB)?

SQLite's single-writer lock causes contention with 4 workers. MongoDB requires application-side joining of FAISS results to metadata. PostgreSQL JSONB with GIN indexing supports flexible metadata filtering in a single indexed scan (`metadata @> '{"source": "arxiv"}'`).

### Why Nginx in front of uvicorn?

Uvicorn should never be exposed directly to the internet. Nginx handles TLS termination, rate limiting (100 req/min per IP), and request buffering so slow clients don't hold uvicorn event loop threads.

---

## Setup

### Prerequisites

- Docker Desktop installed and running
- `.env` file configured (see below)
- FAISS index built (see [Indexing](#indexing))

### Environment Variables (`.env`)

```env
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=your_password

AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

---

## Indexing

The FAISS index must be built **once** before starting the app. The corpus and index files are large (hundreds of MBs) and are excluded from the repository via `.gitignore` — you must generate them locally using one of the options below.

### Option A — Docker CPU (~1 hour for 50k docs)

Does everything in one step: embeds the corpus, builds the FAISS index, and populates PostgreSQL.

```bash
# 1. Build the index and populate the database
docker compose -f docker-compose.indexer.yaml up --build
```

Wait for: `Done. 50000 documents indexed and stored.`

The index writes to `raw_dataset/faiss_index/` on your host via bind mount. Then skip to [Running the App](#running-the-app).

### Option B — HPC GPU (~15-20 minutes on V100/A100)

Use this if you have access to a GPU cluster via SLURM. The GPU indexer (`build_index_gpu.py`) only builds the FAISS index — it does **not** populate PostgreSQL, so there is a separate step for that.

**Step 1** — Submit the SLURM job on the HPC login node:

```bash
sbatch indexing/slurm_index.sh
```

Wait for: `Done. 50000 documents indexed.`

**Step 2** — Copy the two output files back to your laptop:

```bash
scp user@hpc:/path/to/project/faiss_index/index.faiss raw_dataset/faiss_index/
scp user@hpc:/path/to/project/faiss_index/id_map.pkl  raw_dataset/faiss_index/
```

**Step 3** — Populate PostgreSQL using the standalone insert script (reads `corpus.jsonl` directly, no re-embedding):

```bash
docker compose -f docker-compose.indexer.yaml up postgres -d
docker compose -f docker-compose.indexer.yaml run --rm indexer \
    python indexing/insert_postgres.py
```

Wait for: `Done. 50000 documents inserted into PostgreSQL.`

> **Important:** Do not run `build_index.py` after copying the HPC index files — it re-embeds the entire corpus and overwrites `index.faiss`. Use `insert_postgres.py` to populate the database independently.

---

## Running the App

After indexing is complete:

```bash
docker compose up --build
```

Services started:
- `multimodal_rag_app` — FastAPI on port 8000
- `multimodal_rag_postgres` — PostgreSQL on port 5432
- `multimodal_rag_redis` — Redis on port 6379
- `multimodal_rag_nginx` — Nginx on ports 80/443

Confirm the app is healthy:

```bash
curl http://localhost/health
# {"status": "ok", "index_size": 50000}
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns FAISS index size |
| `POST` | `/query/text` | Text query → agent → grounded answer |
| `POST` | `/query/multimodal` | Text + image → direct retrieval |
| `GET` | `/index/stats` | FAISS index statistics |
| `GET` | `/docs` | OpenAPI interactive docs (Swagger UI) |

### Example: Text Query

```bash
curl -X POST http://localhost/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "what is Frankenstein about"}'
```

```json
{
  "answer": "\"Frankenstein; Or, The Modern Prometheus\" by Mary Wollstonecraft Shelley is a Gothic novel written in 1818. It tells the story of Victor Frankenstein, a scientist who creates a sentient creature through an unorthodox experiment. The novel explores themes of ambition, the consequences of playing God, and the isolation and suffering of both the creator and his creation.\n\nFor more details, you can access the full text here (doc_id: doc_00092, source: Gutenberg).",
  "tool_calls": [
    {
      "tool": "text_retriever_tool",
      "input": {"query": "Frankenstein plot summary"}
    }
  ]
}
```

### Example: Multimodal Query

```bash
curl -X POST http://localhost/query/multimodal \
  -F "query=find similar architectural diagrams" \
  -F "image=@/path/to/query_image.jpg"
```

---

## Corpus Format

`raw_dataset/data/corpus.jsonl` — one JSON object per line:

```jsonl
{"doc_id": "doc_00001", "title": "Urban Growth Study", "source": "arxiv", "doc_type": "text", "content": "...", "image_url": null}
{"doc_id": "doc_00002", "title": "Satellite Photo", "source": "nasa", "doc_type": "image", "content": "", "image_url": "https://..."}
{"doc_id": "doc_00003", "title": "Report with Figure", "source": "ieee", "doc_type": "multimodal", "content": "...", "image_path": "raw_dataset/data/images/doc_00003.jpg"}
```

---

## Volume and Mount Strategy

| Data | Type | Host Path | Container Path |
|---|---|---|---|
| FAISS index | bind mount | `./raw_dataset/faiss_index` | `/app/data/faiss_index` |
| Corpus + images | bind mount (ro) | `./raw_dataset/data` | `/app/raw_dataset` |
| CLIP model weights | named volume | `model_cache` | `/app/model_cache` |
| PostgreSQL data | named volume | `postgres_data` | `/var/lib/postgresql/data` |
| Redis data | named volume | `redis_data` | `/data` |

Bind mounts for the FAISS index and corpus allow files built outside Docker (on HPC) to be used directly. Named volumes for model weights and database data persist across container restarts.
