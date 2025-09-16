## How to run the UI 
```
cd frontend
npm install
npm dev run 
```

## API Structure

Please read `lib/research-service.ts` and `lib/apiClinent.ts` for understanding the API structure. 

````markdown
# README  

## How to run the UI
```bash
cd frontend
npm install
npm run dev
```

---

## API Structure

The chatbot integrates several services: **Document Upload**, **Paper Search**, **Summarization**, **Q\&A**, **Chat with Citations**, **Bookmarks**, and **Research Sessions**.
Below is the full API structure that the frontend expects the backend to implement.

---

### 1. Documents API

Manage PDF documents uploaded by users (files or URLs).

* **POST /v1/documents**
  Upload a PDF file or add by URL.

  * Request (file upload): `multipart/form-data`
  * Request (URL):

    ```json
    { "url": "https://arxiv.org/pdf/1234.5678.pdf" }
    ```
  * Response:

    ```json
    {
      "id": "doc_123",
      "name": "1234.5678.pdf",
      "status": "processing"
    }
    ```

* **GET /v1/documents**
  List all uploaded documents.

* **GET /v1/documents/{id}**
  Retrieve metadata + processing status of a specific document.

* **DELETE /v1/documents/{id}**
  Remove a document.

---

### 2. Papers API

Search and retrieve research papers.

* **POST /v1/papers/search**
  Query papers with filters.

  * Request:

    ```json
    {
      "query": "transformer architecture",
      "filters": {
        "minCitations": 100,
        "dateRange": { "from": "2018", "to": "2023" },
        "venue": "NeurIPS"
      },
      "limit": 20,
      "offset": 0
    }
    ```
  * Response:

    ```json
    {
      "papers": [
        {
          "id": "p1",
          "title": "Attention Is All You Need",
          "authors": ["Vaswani, A.", "Shazeer, N."],
          "abstract": "...",
          "citationCount": 85000,
          "publicationDate": "2017-06-12",
          "venue": "NIPS 2017",
          "doi": "10.48550/arXiv.1706.03762",
          "url": "https://arxiv.org/abs/1706.03762",
          "pdfUrl": "https://arxiv.org/pdf/1706.03762.pdf",
          "keywords": ["transformer", "attention"]
        }
      ],
      "totalCount": 1234,
      "query": "transformer architecture"
    }
    ```

* **GET /v1/papers/{id}**
  Get detailed metadata for a specific paper.

---

### 3. Summary API

Generate summaries from selected or all papers.

* **POST /v1/summary**

  * Request:

    ```json
    {
      "paper_ids": ["p1", "p2"],
      "mode": "selected" 
    }
    ```

    or

    ```json
    {
      "mode": "all"
    }
    ```
  * Response:

    ```json
    {
      "summary": "The papers highlight key advances in transformer models...",
      "keyPapers": [
        { "id": "p1", "title": "Attention Is All You Need" },
        { "id": "p2", "title": "BERT: Pre-training..." }
      ],
      "topics": ["Transformers", "Language Models"]
    }
    ```

---

### 4. QA API

Answer questions based on papers (selected or all).

* **POST /v1/qa**

  * Request:

    ```json
    {
      "question": "What are the main contributions of transformer models?",
      "paper_ids": ["p1", "p2"],
      "mode": "selected"
    }
    ```

    or

    ```json
    {
      "question": "What are the main contributions of transformer models?",
      "mode": "all"
    }
    ```
  * Response:

    ```json
    {
      "answer": "Transformers introduced self-attention and parallelization...",
      "sources": [
        { "id": "p1", "title": "Attention Is All You Need" },
        { "id": "p2", "title": "BERT: Pre-training..." }
      ],
      "citations": ["Vaswani et al., 2017", "Devlin et al., 2018"]
    }
    ```

---

### 5. Chat API

Multi-turn chat with citation support.

* **POST /v1/chat**

  * Request:

    ```json
    {
      "session_id": "sess_123",
      "message": "Compare BERT and GPT-3",
      "context": {
        "documents": ["doc_123"],
        "papers": ["p1", "p3"]
      },
      "citationStyle": "apa"
    }
    ```
  * Response:

    ```json
    {
      "content": "BERT is bidirectional [1], while GPT-3 focuses on autoregressive generation [2].",
      "sources": [
        { "id": "p1", "title": "BERT: Pre-training..." },
        { "id": "p3", "title": "GPT-3: Language Models are Few-Shot Learners" }
      ],
      "bibliography": [
        "Devlin et al. (2018). BERT...",
        "Brown et al. (2020). GPT-3..."
      ]
    }
    ```

---

### 6. Bookmarks API

Bookmark/unbookmark research papers.

* **POST /v1/papers/{id}/bookmark** → Add bookmark.
* **DELETE /v1/papers/{id}/bookmark** → Remove bookmark.

---

### 7. Sessions API

Manage research sessions (chat history, mode, etc).

* **GET /v1/sessions** → List sessions.
* **POST /v1/sessions** → Create a new session.
* **PATCH /v1/sessions/{id}** → Update session (title, mode, etc).
* **DELETE /v1/sessions/{id}** → Delete a session.

---

## Notes for Backend

* All APIs are currently mocked in `lib/research-service.ts` and `lib/chat-service.ts`.
* Replace mock logic with real implementations (FastAPI, Flask, or Node/Express).
* Ensure response formats match the structures above, so the frontend works without modification.

```
