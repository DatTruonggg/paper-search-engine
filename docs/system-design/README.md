## System Design (v1)

This document outlines the architecture for the academic chatbot built with OpenAI API, Elasticsearch, Postgres, and MinIO. Scope: Find Papers and Ask Question (RAG). No Asta code reuse.

### Architecture Overview
- Web App (Next.js): chat UI with two modes, streaming responses, citations.
- API Gateway (Node/Python): routes to services; auth, rate-limit, tracing.
- Services (separation of concerns, SOLID):
  - Ingestion Service: harvest IDs (arXiv/OpenAlex), download PDFs, persist to MinIO, write metadata to Postgres, enqueue parse jobs.
  - Parsing Service: GROBID TEI extraction; fallback Unstructured; normalize to JSON with sections, headings, page spans.
  - Indexing Service: section-aware chunking (512–1024 tokens, 15–20% overlap), OpenAI embeddings, upsert to Elasticsearch (BM25 + dense HNSW). Stores payloads: `paper_id`, `section`, `headings`, `year`, `venue`.
  - Retrieval Service: hybrid BM25 + dense search with Rank Fusion (RRF or weighted), optional OpenAI rerank; returns passages with provenance.
  - RAG Answering Service: prompt assembly, context window packing (6–12 chunks), OpenAI generation, structured citations and quotes; streaming to client.
  - Academic Aggregator: fetchers for arXiv/OpenAlex/Semantic Scholar, normalization and dedupe, ranking, caching.

### Data Stores
- MinIO (S3-compatible):
  - `pdf/{paper_id}.pdf`
  - `parsed/{paper_id}.json`
- Postgres:
  - `papers` (id, title, abstract, year, venue, primary_ids, urls)
  - `authors`, `paper_authors`
  - `identifiers` (arxiv_id, doi, openalex_id, s2_id)
  - `citations` (optional)
- Elasticsearch:
  - `paper_chunks` index: `text`, `embedding (dense_vector)`, `paper_id`, `section`, `headings`, `year`, `venue`
  - `papers_meta` index for aggregator ranking and filters

### APIs (v1)
- `POST /search_papers { query, filters?, top_k }` → ranked papers with snippets and why-relevant.
- `POST /qa { question, scope?: { paper_ids? } }` → answer + citations (paper_id, section, quote, url).
- `POST /ingest { source, ids? | query? }` → enqueue ingestion; returns job id.
- `POST /search_academic { query, filters?, top_k }` → unified results across sources.
- `GET /papers/{id}` → metadata + links.

### Key Algorithms
- Section-aware chunking: split by headings; avoid references/acks; maintain page spans for precise quotes.
- Hybrid retrieval: normalize BM25 and dense scores; tie-break by year/venue.
- Rerank (optional): OpenAI small pass on top 50 passages.
- Prompting: enforce citations; penalize missing quotes; refusal when grounding insufficient.

### Observability & Ops
- Structured request logs with trace ids; retrieval traces (query → candidates → final context).
- Basic metrics: p50/p95 latency, token usage, retrieval Recall@k.
- Docker Compose for ES, Postgres, MinIO, services; environment-driven config.

### Risks & Mitigations
- PDF parsing variance → GROBID first, retries, fallback parser.
- OA gaps → keep abstracts; flag non-groundable in UI.
- Latency → cache aggregator results; batch embeddings; stream model outputs.

