# API Specification

## POST /api/search_papers
Request:
```json
{
  "query": "transformer efficiency",
  "top_k": 10,
  "filters": { "year_min": 2018, "year_max": 2024, "authors": ["Xiao"], "venue": "NeurIPS" }
}
```
Response:
```json
{
  "results": [
    {
      "paper_id": "arxiv:1706.03762",
      "title": "Attention Is All You Need",
      "authors": ["Vaswani et al."],
      "year": 2017,
      "venue": "NIPS",
      "snippet": "We introduce the Transformer...",
      "why_relevant": "Matches transformer efficiency criteria.",
      "score": 0.92
    }
  ]
}
```

## POST /api/chat
Request:
```json
{
  "thread_id": "abc123",
  "message": "How do transformers handle long sequences?",
  "scope": { "paper_ids": ["arxiv:2006.04768"] },
  "top_k": 8
}
```
Streamed Response (event chunks):
```json
{ "type": "token", "text": "Transformers handle long..." }
{ "type": "final", "answer": "...", "citations": [ { "paper_id": "arxiv:2006.04768", "section": "3.2", "quote": "Longformer uses dilated windows.", "url": "https://arxiv.org/abs/2004.05150" } ] }
```

## POST /api/ingest
Request:
```json
{ "source": "arxiv", "ids": ["1706.03762"], "query": null }
```
Response:
```json
{ "job_id": "job_001" }
```

## GET /papers/{id}
Response:
```json
{ "id": "arxiv:1706.03762", "title": "Attention Is All You Need", "year": 2017, "venue": "NIPS", "authors": ["Vaswani et al."], "urls": ["https://arxiv.org/abs/1706.03762"] }
```
