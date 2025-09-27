## Changes Summary and How to Run

### Backend

- ASTA QA endpoint wired to ScholarQA pipeline in `backend/api/v1/asta.py`.
  - Uses `FullTextRetriever` and `PaperFinder` with Gemini LLM.
  - Avoids self-call deadlock by keeping local S2 calls in a separate service.

- Local Semantic Scholar compatibility endpoints in `backend/api/v1/semantic_scholar_api.py` are used by ScholarQA retriever.
  - Note: `annotations.refMentions` population is pending; citations may be incomplete if left empty.

### Frontend

- QA page rendering in `frontend/app/qa/page.tsx`:
  - Normalizes inline citations to bracketed form `[n]` and removes malformed cases.
  - Renders References as `[n] Title` with `[n]` clickable to arXiv.
  - Cleans model tags and separator artifacts; bold section titles with spacing.

- Main search interface in `frontend/components/main-search-interface.tsx`:
  - Removed gating that required selected papers to run QA. Users can submit QA anytime.

- Textarea component `frontend/components/ui/textarea.tsx` now uses `React.forwardRef`.

- API client `frontend/lib/apiClient.ts` accepts external `AbortSignal` and allows disabling timeouts when needed.

### How to Run

1) Start Backend API (port 8001)

```bash
cd paper-search-engine
uvicorn backend.api.main:app --host 0.0.0.0 --port 8001 --workers 2
```

2) Start Frontend (Next.js)

```bash
cd paper-search-engine/frontend
npm install
npm run dev
```

3) Open the App

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8001` -> To see the UI please: `http://localhost:8001/docs`

### Notes

- Ensure LLM keys (e.g., GOOGLE_API_KEY for Gemini) are set in environment if required by your ScholarQA config.
- If using local S2 endpoints, confirm that `S2_API_BASE_URL` in ScholarQA utils points to `http://localhost:8001/graph/v1/`.
- For production, run Uvicorn without `--reload` and increase workers appropriately.

---

## What Was Changed (Emphasis) + Brief Theory

### Backend Focus

- Integrated ScholarQA at `backend/api/v1/asta.py` with `FullTextRetriever` + `PaperFinder` (no rerank)
  - Theory: A RAG pipeline = retrieval → (optional) reranking → generation. We focus on high recall to ensure the LLM sees enough evidence before answering. Skipping rerank initially avoids precision-loss from an over‑tight reranker and reduces latency.

- Fixed retriever method to GET for `snippet/search`
  - Theory: Aligning HTTP methods with route definitions avoids 405/404 and ensures idempotent snippet retrieval. Retriever stability is critical for consistent evidence gathering.

- Avoided self-call deadlock by separating services / using `--workers 2`
  - Theory: A single worker doing a blocking HTTP call to itself can deadlock. Multiple workers or splitting endpoints prevents request starvation; each request can be served concurrently.

- Kept `PaperFinder` (no reranker) since it yielded citations reliably
  - Theory: Rerankers (cross-encoders) improve precision but can filter out useful context and add cost. For citation-rich outputs, recall and coverage often matter more; reranking can be added later once upstream citations are robust.

- Local S2 compatibility endpoints: `annotations.refMentions` still pending
  - Theory: ScholarQA uses `refMentions` to map inline citations to paper metadata. Without it, the LLM may produce bracket numbers but links/titles can be incomplete. Populating `refMentions` restores strong citation linking.

### Frontend Focus

- Normalized inline citations to strict `[n]` and References as `[n] Title`
  - Theory: Consistent bracketed citations improve readability and scannability. Clickable `[n]` links in References match scholar UX patterns while keeping the text clean (no raw URLs inline).

- Converted `<Paper ...>` tags to inline `[n]` citations and de-ghosted artifacts
  - Theory: Rendering engines can leak tooling tags; normalizing to markdown preserves semantics and keeps the UI minimal and consistent.

- Removed QA gating (no need to pre-select papers)
  - Theory: Supports topic-level QA over the entire retrieved set; lowers friction and aligns with exploratory workflows. Users can still scope to selections when desired.

- Section title formatting and model tag removal
  - Theory: Clear visual hierarchy (bold titles, spacing) improves comprehension; removing model scaffolding avoids noise.

- `Textarea` uses `forwardRef`, API client supports external Abort/timeout control
  - Theory: Proper ref forwarding resolves React warnings; explicit abort/timeout gives precise control for long‑running QA calls.

### How to Verify the Changes

1) Ask a QA query in the UI without selecting papers. It should submit successfully.
2) Inspect the answer:
   - Inline citations appear as `[1]`, `[2]`, ... (no `], [n]` artifacts).
   - Section titles are bold, on their own line.
   - References list shows lines like `[1] Title` where `[1]` is clickable to arXiv.
