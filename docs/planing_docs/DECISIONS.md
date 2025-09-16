# Architectural Decisions (ADRs)

## ADR-001: Retrieval Strategy
- Decision: Hybrid BM25 + dense (OpenAI embeddings) with score fusion; optional small LLM re-rank on top 50.
- Rationale: Strong baseline recall, controllable cost.
- Status: Accepted.

## ADR-002: Agent Modes
- Decision: fast (single pass) and diligent (≤3 refinements) for both search and chat.
- Rationale: Covers latency vs. quality trade-off.
- Status: Accepted.

## ADR-003: Storage Layout
- Decision: MinIO for artifacts, Postgres for metadata, ES for text + vectors.
- Rationale: Clear separation of concerns; scalable.
- Status: Accepted.

## ADR-004: Parsing
- Decision: GROBID primary; fallback to Unstructured for robustness.
- Rationale: Quality + resilience to parser failures.
- Status: Accepted.

## ADR-005: UI Framework
- Decision: Next.js with streaming; citations UI with hover-to-quote.
- Rationale: Developer velocity and UX.
- Status: Accepted.
