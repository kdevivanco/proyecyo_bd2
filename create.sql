CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title TEXT,
    artist TEXT,
    lyric TEXT
);

-- para full-text search de Postgres (despues)
ALTER TABLE documents
ADD COLUMN IF NOT EXISTS ts tsvector;

CREATE INDEX IF NOT EXISTS idx_documents_ts
ON documents USING GIN (ts);
