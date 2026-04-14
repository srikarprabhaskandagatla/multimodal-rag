import json
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

DATABASE_URL = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  
    echo=False,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_doc_metadata(doc_ids: list[str]) -> list[dict[str, Any]]:
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
    async with AsyncSessionLocal() as session:
        await session.execute(
            text("""
                INSERT INTO documents
                    (doc_id, title, source, doc_type, content_preview, metadata)
                VALUES
                    (:doc_id, :title, :source, :doc_type, :content_preview, cast(:metadata as jsonb))
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
                "metadata": json.dumps(metadata),
            },
        )
        await session.commit()