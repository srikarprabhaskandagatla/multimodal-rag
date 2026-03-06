"""
retriever.py — FAISS Vector Retrieval
───────────────────────────────────────
WHY FAISS (and not Pinecone, Weaviate, Chroma, Milvus, or pgvector)?

- FAISS (Facebook AI Similarity Search) runs entirely in-process. No network
  hop = microsecond latency on the retrieval step itself.

- Pinecone/Weaviate are managed cloud services. On Unity (AWS EC2), every query
  would add ~50-100ms of network RTT plus egress cost. Disqualified.

- Chroma: excellent for prototyping, but single-threaded and lacks production
  ANN algorithms (HNSW, IVF) that FAISS has tuned for years.

- pgvector: PostgreSQL extension for vector similarity. Excellent for small
  corpora (<100k), but at 50k docs with 512-dim CLIP vectors it's 10-30x slower
  than FAISS's HNSW due to sequential scan fallback.

- Milvus: production-grade, but requires a separate cluster deployment.
  FAISS embedded in-process is simpler and sufficient at 50k scale.

INDEX TYPE: IndexHNSWFlat
  - HNSW (Hierarchical Navigable Small World) gives ~99% recall at <1ms query
    time for 50k vectors of 512 dims.
  - IndexFlatIP (exact search) would be more accurate but O(n) — at 50k docs
    and 100 QPS that's 5M dot products per second. HNSW is O(log n).
"""

import os
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.embeddings import embed_query
from app.db import get_doc_metadata

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
TOP_K = int(os.getenv("FAISS_TOP_K", "20"))
EMBEDDING_DIM = 512  # CLIP ViT-B/32 output dimension


class FAISSRetriever:
    """
    Wraps a FAISS HNSW index with doc ID mapping and async retrieval interface.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []  # Maps FAISS int ID → doc_id string

    def load(self):
        """
        Load pre-built FAISS index from disk.

        Why load at startup (not lazily)?
        The HNSW graph structure is ~200MB. Loading it on the first request
        would add 3-5 seconds of latency to that request. Loading at startup
        amortizes this cost before any traffic arrives.
        """
        index_file = Path(FAISS_INDEX_PATH) / "index.faiss"
        idmap_file = Path(FAISS_INDEX_PATH) / "id_map.pkl"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Run `python indexing/build_index.py` first."
            )

        self.index = faiss.read_index(str(index_file))
        with open(idmap_file, "rb") as f:
            self.id_map = pickle.load(f)

    async def retrieve(
        self,
        text: str | None = None,
        image=None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Embed query → search FAISS → fetch metadata → return results.

        top_k pipeline:
        We retrieve FAISS_TOP_K=20 candidates, fetch their full metadata from
        PostgreSQL, then return top_k=5 after any post-retrieval reranking.
        This two-stage approach (ANN → rerank) is standard practice because
        ANN search sacrifices some recall for speed.
        """
        if self.index is None:
            self.load()

        # 1. Embed
        query_vec = await embed_query(text=text, image=image)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        # 2. ANN search in FAISS
        # faiss.search returns (distances, indices) — distances are cosine
        # similarities (since vectors are L2-normalized + IndexFlatIP)
        distances, indices = self.index.search(query_vec, TOP_K)

        # 3. Map integer indices back to doc_id strings
        candidate_ids = [
            self.id_map[idx]
            for idx in indices[0]
            if idx != -1  # FAISS returns -1 for empty slots
        ]

        # 4. Fetch metadata from PostgreSQL
        docs = await get_doc_metadata(candidate_ids)

        # 5. Attach similarity scores and return top_k
        score_map = {
            self.id_map[idx]: float(distances[0][i])
            for i, idx in enumerate(indices[0])
            if idx != -1
        }
        for doc in docs:
            doc["score"] = score_map.get(doc["doc_id"], 0.0)

        docs.sort(key=lambda d: d["score"], reverse=True)
        return docs[:top_k]


# Singleton — shared across all requests in a worker process
# Why singleton? FAISS index is 200MB. One per process, not one per request.
_retriever_instance: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FAISSRetriever()
        _retriever_instance.load()
    return _retriever_instance