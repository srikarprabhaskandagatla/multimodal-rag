import asyncio
import json
import logging
import math
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embeddings import embed_query
from app.db import insert_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/data/faiss_index")
CORPUS_PATH = os.getenv("CORPUS_PATH", "data/corpus.jsonl")
EMBEDDING_DIM = 512   
HNSW_M = 32          
BATCH_SIZE = 64       

CHUNK_INDEX = int(os.getenv("CHUNK_INDEX", "0"))
CHUNK_TOTAL = int(os.getenv("CHUNK_TOTAL", "1"))


def build_faiss_index(dim: int) -> faiss.Index:
    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200  
    hnsw.hnsw.efSearch = 64        
    index = faiss.IndexIDMap(hnsw)
    return index


def _fetch_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "multimodal-rag/1.0"})
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        pass
    return None


async def embed_document(doc: dict) -> np.ndarray | None:
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
    id_map: list[str] = []   

    logger.info("Loading corpus from %s ...", corpus_path)
    with open(corpus_path) as f:
        all_docs = [json.loads(line) for line in f if line.strip()]

    if CHUNK_TOTAL > 1:
        chunk_size = math.ceil(len(all_docs) / CHUNK_TOTAL)
        start = CHUNK_INDEX * chunk_size
        end = min(start + chunk_size, len(all_docs))
        docs = all_docs[start:end]
        logger.info("Chunk %d/%d — processing docs %d..%d (%d total)",
                    CHUNK_INDEX, CHUNK_TOTAL - 1, start, end - 1, len(docs))
    else:
        docs = all_docs
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

        content = doc.get("content", "")
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
            batch_vecs = np.stack(vectors[-BATCH_SIZE:])
            batch_ids = np.array(faiss_ids[-BATCH_SIZE:], dtype=np.int64)
            index.add_with_ids(batch_vecs, batch_ids)
            logger.info("Indexed %d / %d documents", i + 1, len(docs))

    remainder = len(vectors) % BATCH_SIZE
    if remainder:
        batch_vecs = np.stack(vectors[-remainder:])
        batch_ids = np.array(faiss_ids[-remainder:], dtype=np.int64)
        index.add_with_ids(batch_vecs, batch_ids)

    logger.info("FAISS index built: %d vectors", index.ntotal)

    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "id_map.pkl", "wb") as f:
        pickle.dump(id_map, f)
    logger.info("Saved index to %s", output_dir)

    logger.info("Inserting %d documents into PostgreSQL ...", len(db_inserts))
    for doc_meta in db_inserts:
        await insert_document(**doc_meta)
    logger.info("Done. %d documents indexed and stored.", len(db_inserts))


if __name__ == "__main__":
    asyncio.run(main())
