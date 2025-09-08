## Two-week Gantt (v1)

This file captures the two-week delivery plan. Team: Tin, Dat.

```mermaid
gantt
  title Two-week plan
  dateFormat  YYYY-MM-DD
  axisFormat  %m/%d
  section Planning
  Finalize requirements, schemas, filters (Tin, Dat)       :p1, 2025-09-08, 1d
  section Data & Index (Tin, Dat)
  Harvest 1k IDs (arXiv/OpenAlex)                          :d1, 2025-09-09, 1d
  Download PDFs + metadata persistence (MinIO/Postgres)     :d2, 2025-09-10, 1d
  GROBID + fallback parsing pipeline                        :d3, 2025-09-11, 2d
  Chunking + embeddings + ES BM25/dense indexing            :d4, 2025-09-13, 2d
  Retrieval sanity eval (Recall@k, NDCG)                    :d5, 2025-09-15, 0.5d
  section Online & UX (Tin, Dat)
  Academic aggregator (arXiv/OpenAlex/S2) + caching         :o1, 2025-09-09, 2d
  API layer (search_papers, search_academic)                :o2, 2025-09-11, 1d
  Web UI (Next.js): Find Papers + Ask Question              :o3, 2025-09-12, 2d
  section RAG & Integration (Tin, Dat)
  Hybrid retrieval + optional re-rank (OpenAI)              :r1, 2025-09-15, 1d
  RAG answering, citations, streaming                       :r2, 2025-09-16, 1d
  End-to-end wire-up + error handling                       :r3, 2025-09-17, 0.5d
  section Testing & Launch (Tin, Dat)
  E2E tests, eval harness, quality passes                   :t1, 2025-09-17, 1d
  Docker Compose deploy (ES, Postgres, MinIO, services)     :t2, 2025-09-18, 0.5d
  Buffer/Polish                                             :t3, 2025-09-19, 0.5d
```

### Milestones
- Data ready and searchable by end of Day 5.
- Hybrid retrieval + RAG answering by Day 8–9.
- E2E demo and deploy by Day 10.

