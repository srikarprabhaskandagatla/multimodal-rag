"""
db.py — PostgreSQL Metadata Store
────────────────────────────────────
WHY POSTGRESQL (not SQLite, MySQL, or MongoDB)?

- SQLite: single-writer lock. With 4 uvicorn workers issuing concurrent reads
  this causes contention. Also, no JSONB with GIN indexing.

- MySQL: JSON column support exists but is not GIN-indexed (scan, not seek).
  Also, MySQL's async drivers (aiomysql) are less mature than asyncpg.

- MongoDB: document store seems natural for metadata, but joins between
  doc_id and retrieval results require application-side merging. PostgreSQL's
  JOIN is a single query. Also, PostgreSQL JSONB handles flexible metadata
  fields without sacrificing relational integrity.

- asyncpg (not psycopg2): asyncpg is a pure-Python async PostgreSQL driver
  with zero blocking I/O. psycopg2 blocks the event loop. SQLAlchemy 2.0's
  async session uses asyncpg under the hood.
"""

import os
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "raguser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")

# asyncpg DSN — note 'postgresql+asyncpg://' prefix for SQLAlchemy async engine
DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Why pool_size=10, max_overflow=20?
# With 4 uvicorn workers × up to 5 concurrent requests each = 20 peak DB
# connections. pool_size keeps 10 warm; max_overflow allows burst to 30.
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Recycle stale connections silently
    echo=False,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_doc_metadata(doc_ids: list[str]) -> list[dict[str, Any]]:
    """
    Fetch full metadata for a list of doc_ids in a single query.

    Why a single IN query instead of N individual queries?
    N=20 individual queries × ~1ms each = 20ms overhead per request.
    One IN query with 20 IDs = ~2ms. Always batch.
    """
    if not doc_ids:
        return []

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT
                    doc_id,
                    title,
                    source,
                    doc_type,
                    content_preview,
                    metadata,
                    created_at
                FROM documents
                WHERE doc_id = ANY(:ids)
            """),
            {"ids": doc_ids},
        )
        rows = result.mappings().all()
        return [dict(row) for row in rows]


async def insert_document(
    doc_id: str,
    title: str,
    source: str,
    doc_type: str,  # 'text' | 'image' | 'multimodal'
    content_preview: str,
    metadata: dict,
) -> None:
    """
    Insert a single document's metadata during ingestion.
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO documents
                    (doc_id, title, source, doc_type, content_preview, metadata)
                VALUES
                    (:doc_id, :title, :source, :doc_type, :content_preview, :metadata::jsonb)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    content_preview = EXCLUDED.content_preview,
                    metadata = EXCLUDED.metadata
            """),
            {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "doc_type": doc_type,
                "content_preview": content_preview,
                "metadata": str(metadata).replace("'", '"'),
            },
        )
        await session.commit()