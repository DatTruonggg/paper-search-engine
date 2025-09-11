# Implementation Plan (2 Weeks)

## Objectives
- Deliver Agent Search (PaperFinder-lite) and Agent Chat (ScholarQA-lite).
- Use Elasticsearch (BM25 + dense), Postgres, MinIO, FastAPI, Next.js.

## Week 1 — Data & Search
- Foundations: Docker Compose (ES, PG, MinIO), env/config, project skeleton.
- Ingestion: harvest 1k IDs (arXiv/OpenAlex), store metadata (PG), PDFs (MinIO).
- Parsing: GROBID first, fallback Unstructured; normalize JSON with sections.
- Indexing: section-aware chunking (512–1024, 15–20% overlap), OpenAI embeddings, upsert to ES `paper_chunks`; build `papers_meta`.
- Agent Search (fast mode): simple query analyzer → hybrid retrieval (BM25 + dense) → aggregation → ranking → snippets.

## Week 2 — Chat & Integration
- Agent Chat: hybrid retrieval → optional small LLM re-rank → context packing (6–12 chunks) → grounded answer with citations and quotes.
- API & UI: FastAPI endpoints, Next.js search and chat with streaming.
- Observability: request tracing, retrieval traces, token usage, p50/p95.
- Eval: sanity Recall@k/NDCG set, E2E tests, error handling.

## Deliverables
- Working `/api/search_papers` and `/api/chat` with streaming and citations.
- ES indices `paper_chunks`, `papers_meta`; Docker Compose; minimal UI.

## Risks & Mitigations
- Parsing variance → retries, fallback parser.
- Latency/cost → cache aggregator, batch embeddings, small models for re-rank.
- Ungroundable → explicit refusal path.

## Milestones
- Day 5: Data indexed, search live.
- Day 9: Chat live with citations.
- Day 10: E2E demo, docs, compose.
