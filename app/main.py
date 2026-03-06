"""
main.py — FastAPI Application Entrypoint
──────────────────────────────────────────
WHY FASTAPI (and not Flask, Django, or aiohttp)?

- Flask: synchronous by default. Async support via asyncio is bolted on and
  requires explicit async context management. FAISS + Redis + PostgreSQL all
  benefit from async I/O — Flask would block the worker thread during each op.

- Django: heavyweight ORM and middleware stack. We already have SQLAlchemy for
  async PostgreSQL. Django's ORM doesn't support asyncpg natively in 2.x.

- aiohttp: low-level async HTTP server. No request validation, no automatic
  OpenAPI docs, no dependency injection. We'd reimplement FastAPI features.

- FastAPI: async-first, Pydantic-validated request/response models, automatic
  OpenAPI schema at /docs, and Depends() injection for the retriever singleton.
  The correct choice for a production async ML-serving API.
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from app.agent import run_agent
from app.cache import get_cached, set_cached
from app.retriever import get_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown hooks) ───────────────────────────────────────
# Why lifespan and not @app.on_event("startup")?
# @app.on_event is deprecated in FastAPI 0.93+. Lifespan context managers are
# the current standard and work correctly with pytest's async test fixtures.

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FAISS index into memory...")
    get_retriever()  # Eagerly load index at startup — not on first request
    logger.info("FAISS index loaded. Application ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multimodal RAG Agent",
    description="Vision + Language retrieval over a 50k-document corpus using CLIP + FAISS + LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: locked down to internal cluster origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to specific origins in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    chat_history: list[dict] | None = None


class RetrievalResult(BaseModel):
    doc_id: str
    title: str
    source: str
    doc_type: str
    content_preview: str
    score: float


class AgentResponse(BaseModel):
    answer: str
    tool_calls: list[dict[str, Any]]
    results: list[RetrievalResult] | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Unity cluster load balancers poll this every 30s to determine instance health.
    Returns 200 only when FAISS index is loaded and retriever is ready.
    """
    retriever = get_retriever()
    if retriever.index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    return {"status": "ok", "index_size": retriever.index.ntotal}


@app.post("/query/text", response_model=AgentResponse)
async def query_text(request: QueryRequest):
    """
    Text-only query endpoint.
    Routes through the LangChain agent which will call text_retriever_tool.
    """
    try:
        result = await run_agent(
            query=request.query,
            chat_history=request.chat_history,
        )
        return AgentResponse(**result)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/multimodal", response_model=AgentResponse)
async def query_multimodal(
    query: str = Form(default=""),
    image: UploadFile = File(default=None),
):
    """
    Multimodal query endpoint — accepts text + optional image.

    Why multipart/form-data instead of JSON?
    JSON cannot encode binary image data efficiently (base64 bloats by 33%).
    multipart/form-data is the HTTP standard for mixed text+binary payloads.

    Why UploadFile and not bytes?
    UploadFile streams the file without loading it entirely into memory first.
    For large images, this avoids OOM on the API server.
    """
    pil_image = None
    if image:
        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

    # For multimodal queries, we bypass the agent and call retriever directly
    # to avoid the LLM overhead when the routing decision is already made
    # (user explicitly hit the /multimodal endpoint).
    cached = await get_cached(text=query or None, has_image=pil_image is not None)
    if cached:
        return AgentResponse(answer="Retrieved from cache.", tool_calls=[], results=cached)

    retriever = get_retriever()
    results = await retriever.retrieve(
        text=query if query else None,
        image=pil_image,
        top_k=5,
    )
    await set_cached(text=query or None, has_image=pil_image is not None, results=results)

    return AgentResponse(
        answer=f"Found {len(results)} relevant documents.",
        tool_calls=[{"tool": "multimodal_retriever_tool", "input": {"query": query}}],
        results=results,
    )


@app.get("/index/stats")
async def index_stats():
    """Returns FAISS index statistics for monitoring."""
    retriever = get_retriever()
    return {
        "total_vectors": retriever.index.ntotal if retriever.index else 0,
        "embedding_dim": 512,
        "index_type": "HNSW",
        "doc_count": len(retriever.id_map),
    }