-- Runs once, when the Postgres volume is first created.
-- pgvector is enabled now so week 2 (RAG) needs no manual step.
CREATE EXTENSION IF NOT EXISTS vector;
