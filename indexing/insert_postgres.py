import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import insert_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = os.getenv("CORPUS_PATH", "raw_dataset/data/corpus.jsonl")
BATCH_LOG_EVERY = 1000


async def main():
    corpus_path = Path(CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with open(corpus_path) as f:
        docs = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d documents from corpus.", len(docs))

    inserted = 0
    for doc in docs:
        content = doc.get("content", "")
        extra_meta = doc.get("metadata", {})
        if doc.get("image_url"):
            extra_meta["image_url"] = doc["image_url"]

        await insert_document(
            doc_id=doc["doc_id"],
            title=doc.get("title", ""),
            source=doc.get("source", ""),
            doc_type=doc.get("doc_type", "text"),
            content_preview=content[:200],
            metadata=extra_meta,
        )
        inserted += 1
        if inserted % BATCH_LOG_EVERY == 0:
            logger.info("Inserted %d / %d documents", inserted, len(docs))

    logger.info("Done. %d documents inserted into PostgreSQL.", inserted)


if __name__ == "__main__":
    asyncio.run(main())
