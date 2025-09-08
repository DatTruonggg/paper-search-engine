## Workflow Diagram (v1)

This document shows the end-to-end workflow for the academic chatbot with two capabilities: Find Papers and Ask Question (RAG). Stack: OpenAI API (embeddings + generation), Elasticsearch (BM25 + vector), Postgres (metadata), MinIO (PDFs/parsed JSON). No external Asta code.

```mermaid
flowchart TD
  U[User] --> R[Intent Router]
  R -->|Find Papers| AGG[Academic Aggregator\n(arXiv, OpenAlex, S2)]
  AGG --> SR[Ranked Paper Results\n(snippets + scores)]

  R -->|Ask Question| RET[Hybrid Retriever (Elasticsearch)]
  RET --> RR[Optional Rerank (OpenAI)]
  RR --> CT[Context Builder\n(select 6–12 chunks)]
  CT --> GEN[OpenAI LLM\n(gpt-4o-mini)]
  GEN --> ANS[Answer + Citations\n(grounded quotes)]

  SR --> UI[Web UI]
  ANS --> UI

  subgraph Offline Ingestion & Indexing
    Q[Collect IDs (arXiv/OpenAlex)] --> DL[Download PDFs]
    DL --> PRS[Parse: GROBID → Unstructured]
    PRS --> CH[Section-aware Chunking]
    CH --> EMB[OpenAI Embeddings\n(text-embedding-3-large)]
    EMB --> ES[Elasticsearch Index\nBM25 + HNSW]
    META[Metadata → Postgres] --> ES
    PDF[PDF/Parsed JSON → MinIO]
  end
```

### Components
- Intent Router: decides between Find Papers and Ask Question.
- Academic Aggregator: queries arXiv/OpenAlex/Semantic Scholar, normalizes, dedupes, ranks.
- Hybrid Retriever: BM25 + dense-vector search in Elasticsearch with fusion.
- Reranker (optional): small OpenAI pass for improved ordering.
- Context Builder: assembles top passages with provenance.
- LLM: generates answers with citations and inline quotes.
- Ingestion: harvest, parse, chunk, embed, index; store metadata and artifacts.

### Notes
- Citations include `paper_id`, section/heading, and URL.
- Missing PDFs fall back to abstract-only (marked non-groundable).

