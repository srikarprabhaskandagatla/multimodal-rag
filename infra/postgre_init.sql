-- infra/postgres_init.sql
-- Executed once by the PostgreSQL container on first startup.
-- ─────────────────────────────────────────────────────────────────────────────
-- WHY THIS SCHEMA DESIGN?
--
-- doc_type ENUM enforces data integrity — prevents inserting 'IMG' instead of
-- 'image'. More efficient than a text column (4 bytes vs variable length).
--
-- metadata JSONB: stores flexible per-document fields (e.g., image dimensions,
-- page number, author). JSONB is binary-parsed and supports GIN indexing —
-- unlike TEXT::json which is re-parsed on every query.
--
-- GIN index on metadata: enables fast queries like
--   WHERE metadata @> '{"source": "arxiv"}'
-- without full table scan.
--
-- BRIN index on created_at: documents are inserted in roughly chronological
-- order. BRIN (Block Range INdex) stores min/max per disk block — excellent
-- for time-range scans on append-mostly tables at minimal storage cost.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TYPE doc_type_enum AS ENUM ('text', 'image', 'multimodal');

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       TEXT NOT NULL UNIQUE,         -- Matches FAISS id_map entries
    title        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    doc_type     doc_type_enum NOT NULL DEFAULT 'text',
    content_preview TEXT NOT NULL DEFAULT '',  -- First 200 chars for display
    metadata     JSONB NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Primary lookup: FAISS retrieves doc_ids, we fetch metadata by doc_id
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);

-- Flexible metadata filtering (source, category, date ranges stored as JSONB)
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN(metadata);

-- Time-range queries on large, append-only corpus
CREATE INDEX IF NOT EXISTS idx_documents_created_brin ON documents USING BRIN(created_at);

-- Partial index for image-only document lookups (frequent in multimodal queries)
CREATE INDEX IF NOT EXISTS idx_documents_images
    ON documents(doc_id)
    WHERE doc_type IN ('image', 'multimodal');

-- Auto-update updated_at on row modification
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();