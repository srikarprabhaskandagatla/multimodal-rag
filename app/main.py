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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FAISS index into Memory.")
    get_retriever()  
    logger.info("FAISS Index Loaded Successfully. Application is Ready.")
    yield
    logger.info("Shutting Down!")


app = FastAPI(
    title="Multimodal RAG Agent",
    description="Vision + Language retrieval over a 50k-document corpus using CLIP + FAISS + LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.get("/health")
async def health():
    retriever = get_retriever()
    if retriever.index is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    return {"status": "ok", "index_size": retriever.index.ntotal}


@app.post("/query/text", response_model=AgentResponse)
async def query_text(request: QueryRequest):
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
    pil_image = None
    if image:
        contents = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

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
    retriever = get_retriever()
    return {
        "total_vectors": retriever.index.ntotal if retriever.index else 0,
        "embedding_dim": 512,
        "index_type": "HNSW",
        "doc_count": len(retriever.id_map),
    }