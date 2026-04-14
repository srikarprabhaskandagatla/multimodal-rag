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
    def __init__(self):
        self.index: faiss.Index | None = None
        self.id_map: list[str] = []  # Maps FAISS int ID → doc_id string

    def load(self):
        index_dir = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
        index_file = Path(index_dir) / "index.faiss"
        idmap_file = Path(index_dir) / "id_map.pkl"

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

_retriever_instance: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FAISSRetriever()
        _retriever_instance.load()
    return _retriever_instance