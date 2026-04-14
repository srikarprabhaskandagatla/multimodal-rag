"""
build_index.py — Offline FAISS Index Builder
──────────────────────────────────────────────
Run this ONCE before starting the API server to:
  1. Load your document corpus (text + images)
  2. Embed every document with CLIP
  3. Build a FAISS HNSW index
  4. Insert document metadata into PostgreSQL
  5. Save index.faiss + id_map.pkl to FAISS_INDEX_PATH

Usage:
    python indexing/build_index.py

Expected corpus format (JSONL, one document per line):
    {"doc_id": "doc_001", "title": "...", "source": "...", "doc_type": "text",
     "content": "...", "image_path": null}
    {"doc_id": "doc_002", "title": "...", "source": "...", "doc_type": "image",
     "content": "", "image_path": "/data/images/doc_002.jpg"}

WHY HNSW (not IVF or Flat)?
- IndexFlatIP: exact search, O(n) per query. At 50k docs × 100 QPS = 5M dot
  products/sec. Fine for <10k docs; too slow beyond that.
- IndexIVFFlat: requires training (k-means clustering). Good for >1M vectors.
  At 50k, the overhead isn't justified.
- IndexHNSWFlat: approximate nearest neighbor with ~99% recall at <1ms. No
  training required. Optimal for the 10k-500k range.
"""

import asyncio
import json
import logging
import os
import pickle
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Must import after dotenv so env vars are set before module-level reads
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings import embed_query
from app.db import insert_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")
EMBEDDING_DIM = 512   # CLIP ViT-B/32 output dimension
HNSW_M = 32          # HNSW connectivity parameter — higher = better recall, more memory
BATCH_SIZE = 64       # Documents embedded per batch (tune to GPU memory)


def build_faiss_index(dim: int) -> faiss.Index:
    """
    Create an HNSW index wrapped in IndexIDMap so we can use arbitrary int IDs.

    Why IndexIDMap?
    HNSW natively uses sequential integer IDs (0, 1, 2, ...). IndexIDMap lets
    us assign explicit int64 IDs that map to our doc_id strings via id_map.
    """
    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200  # Higher = better graph quality at build time
    hnsw.hnsw.efSearch = 64         # Higher = better recall at query time
    index = faiss.IndexIDMap(hnsw)
    return index


def _fetch_image(url: str) -> Image.Image | None:
    """Fetch image from URL, return PIL Image or None on failure."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "multimodal-rag/1.0"})
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


async def embed_document(doc: dict) -> np.ndarray | None:
    """
    Embed a single document. Returns None if embedding fails (skip, don't crash).
    Supports both local image_path and remote image_url.
    """
    try:
        text = doc.get("content") or doc.get("title") or None
        image = None

        # Prefer local path, fall back to URL fetch
        if doc.get("image_path") and Path(doc["image_path"]).exists():
            image = Image.open(doc["image_path"]).convert("RGB")
        elif doc.get("image_url"):
            image = _fetch_image(doc["image_url"])

        return await embed_query(text=text, image=image)
    except Exception as e:
        logger.warning("Failed to embed doc %s: %s", doc.get("doc_id"), e)
        return None


async def main():
    corpus_path = Path(CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. "
            "Set CORPUS_PATH env var or place corpus.jsonl in data/."
        )

    output_dir = Path(FAISS_INDEX_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = build_faiss_index(EMBEDDING_DIM)
    id_map: list[str] = []   # id_map[i] = doc_id string for FAISS integer ID i

    logger.info("Loading corpus from %s ...", corpus_path)
    with open(corpus_path) as f:
        docs = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d documents", len(docs))

    vectors = []
    faiss_ids = []
    db_inserts = []

    for i, doc in enumerate(docs):
        vec = await embed_document(doc)
        if vec is None:
            continue

        faiss_int_id = len(id_map)
        id_map.append(doc["doc_id"])
        vectors.append(vec.astype(np.float32))
        faiss_ids.append(faiss_int_id)

        # Collect metadata for PostgreSQL batch insert
        content = doc.get("content", "")
        # Keep image_url in metadata so the API can return it to the frontend
        extra_meta = doc.get("metadata", {})
        if doc.get("image_url"):
            extra_meta["image_url"] = doc["image_url"]
        db_inserts.append({
            "doc_id": doc["doc_id"],
            "title": doc.get("title", ""),
            "source": doc.get("source", ""),
            "doc_type": doc.get("doc_type", "text"),
            "content_preview": content[:200],
            "metadata": extra_meta,
        })

        if (i + 1) % BATCH_SIZE == 0:
            # Add batch to FAISS
            batch_vecs = np.stack(vectors[-BATCH_SIZE:])
            batch_ids = np.array(faiss_ids[-BATCH_SIZE:], dtype=np.int64)
            index.add_with_ids(batch_vecs, batch_ids)
            logger.info("Indexed %d / %d documents", i + 1, len(docs))

    # Add remaining documents
    remainder = len(vectors) % BATCH_SIZE
    if remainder:
        batch_vecs = np.stack(vectors[-remainder:])
        batch_ids = np.array(faiss_ids[-remainder:], dtype=np.int64)
        index.add_with_ids(batch_vecs, batch_ids)

    logger.info("FAISS index built: %d vectors", index.ntotal)

    # Save FAISS index and id_map
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "id_map.pkl", "wb") as f:
        pickle.dump(id_map, f)
    logger.info("Saved index to %s", output_dir)

    # Insert metadata into PostgreSQL
    logger.info("Inserting %d documents into PostgreSQL ...", len(db_inserts))
    for doc_meta in db_inserts:
        await insert_document(**doc_meta)
    logger.info("Done. %d documents indexed and stored.", len(db_inserts))


if __name__ == "__main__":
    asyncio.run(main())
