# Architecture

## Components
- API Gateway (FastAPI): exposes `/api/search_papers`, `/api/chat`, `/api/ingest`, `/papers/{id}`.
- Ingestion Service: harvest IDs (arXiv/OpenAlex), download PDFs, persist to MinIO/PG, enqueue parsing.
- Parsing Service: GROBID → fallback Unstructured; normalize sections/page spans.
- Indexing Service: chunking, embeddings, upsert to ES (`paper_chunks`, `papers_meta`).
- Retrieval Service: hybrid BM25 + dense, fusion; optional LLM re-rank.
- Agent Search (PaperFinder-lite): query analysis, retrieval plan, aggregation, ranking.
- Agent Chat (ScholarQA-lite): retrieval, context building, grounded answer with citations.
- Frontend (Next.js): Find Papers and Chat with streaming.

## Data Stores
- MinIO: `pdf/{paper_id}.pdf`, `parsed/{paper_id}.json`.
- Postgres: papers, authors, identifiers; optional citations.
- Elasticsearch: `paper_chunks` (text, embedding, paper_id, section, year, venue), `papers_meta`.

## Interfaces (SOLID)
- Retriever interface: `retrieve(query, filters, k) -> passages`.
- Ranker interface: `rerank(passages) -> passages`.
- Agent interface: `run(inputs) -> outputs`.

## Modes
- fast: single-pass retrieval and answer.
- diligent: up to 2–3 refinement iterations.
