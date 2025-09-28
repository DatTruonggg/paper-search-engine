# API Reference

## Overview

The Paper Search Engine provides both REST API endpoints and Python library interfaces for programmatic access to search functionality.

## REST API

### Base URL
```
http://localhost:8002
```

### Authentication
Currently, no authentication is required for local deployments.

### Content Types
- Request: `application/json`
- Response: `application/json`

---

## Endpoints

### 1. Search Papers

Search for papers using various search modes and filters.

**Endpoint:** `POST /search`

**Request Body:**
```json
{
  "query": "string",
  "max_results": 10,
  "search_mode": "hybrid",
  "categories": ["cs.CL", "cs.AI"],
  "date_from": "2020-01-01",
  "date_to": "2024-12-31",
  "author": "John Doe"
}
```

**Parameters:**
- `query` (string, required): Search query text
- `max_results` (integer, optional): Maximum number of results (default: 20, max: 100)
- `search_mode` (string, optional): Search mode (default: "hybrid")
  - `"hybrid"`: Combines BM25 and semantic search (40%/60%)
  - `"semantic"`: Pure embedding-based search
  - `"bm25"`: Traditional full-text search
  - `"title_only"`: Search only in paper titles
- `categories` (array, optional): Filter by paper categories
- `date_from` (string, optional): Filter papers from this date (YYYY-MM-DD)
- `date_to` (string, optional): Filter papers to this date (YYYY-MM-DD)
- `author` (string, optional): Filter by author name

**Response:**
```json
{
  "results": [
    {
      "paper_id": "1706.03762",
      "title": "Attention Is All You Need",
      "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
      "abstract": "The dominant sequence transduction models...",
      "score": 0.95,
      "categories": ["cs.CL"],
      "publish_date": "2017-06-12",
      "word_count": 8547,
      "pdf_size": 1024000,
      "has_images": true,
      "markdown_path": "/data/processed/markdown/1706.03762.md"
    }
  ],
  "total_found": 42,
  "search_time_ms": 156,
  "search_mode": "hybrid"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "transformer attention mechanism",
    "max_results": 5,
    "search_mode": "hybrid",
    "categories": ["cs.CL"]
  }'
```

---

### 2. Get Paper Details

Retrieve detailed information about a specific paper.

**Endpoint:** `GET /papers/{paper_id}`

**Parameters:**
- `paper_id` (string, required): Unique paper identifier (e.g., ArXiv ID)

**Response:**
```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
  "content": "# Attention Is All You Need\n\n## Abstract\n\nThe dominant sequence...",
  "categories": ["cs.CL"],
  "publish_date": "2017-06-12",
  "word_count": 8547,
  "chunk_count": 42,
  "has_images": true,
  "pdf_size": 1024000,
  "downloaded_at": "2025-09-12T20:18:28.510962",
  "indexed_at": "2025-09-13T13:25:42.123456",
  "markdown_path": "/data/processed/markdown/1706.03762.md",
  "pdf_path": "/data/pdfs/1706.03762.pdf"
}
```

**Example:**
```bash
curl "http://localhost:8002/papers/1706.03762"
```

---

### 3. Get Similar Papers

Find papers similar to a given paper using semantic similarity.

**Endpoint:** `GET /papers/{paper_id}/similar`

**Parameters:**
- `paper_id` (string, required): Reference paper ID
- `max_results` (integer, optional): Maximum number of results (default: 10)

**Response:**
```json
{
  "reference_paper": {
    "paper_id": "1706.03762",
    "title": "Attention Is All You Need"
  },
  "similar_papers": [
    {
      "paper_id": "1810.04805",
      "title": "BERT: Pre-training of Deep Bidirectional Transformers",
      "authors": ["Jacob Devlin", "Ming-Wei Chang"],
      "similarity_score": 0.87,
      "categories": ["cs.CL"],
      "publish_date": "2018-10-11"
    }
  ],
  "total_found": 15
}
```

**Example:**
```bash
curl "http://localhost:8002/papers/1706.03762/similar?max_results=5"
```

---

### 4. Search Statistics

Get statistics about the search index and system performance.

**Endpoint:** `GET /stats`

**Response:**
```json
{
  "index_stats": {
    "total_papers": 996,
    "total_chunks": 45678,
    "index_size_mb": 245.7,
    "last_updated": "2025-09-13T13:25:42.123456"
  },
  "search_stats": {
    "avg_search_time_ms": 156,
    "total_searches": 1247,
    "popular_categories": [
      {"category": "cs.CL", "count": 387},
      {"category": "cs.AI", "count": 298},
      {"category": "cs.LG", "count": 245}
    ]
  },
  "system_stats": {
    "elasticsearch_health": "green",
    "bge_model_loaded": true,
    "memory_usage_mb": 2048
  }
}
```

**Example:**
```bash
curl "http://localhost:8002/stats"
```

---

### 5. Health Check

Check the health status of all system components.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "elasticsearch": {
      "status": "healthy",
      "cluster_health": "green",
      "response_time_ms": 12
    },
    "bge_embedder": {
      "status": "healthy",
      "model_loaded": true,
      "device": "cpu"
    },
    "search_index": {
      "status": "healthy",
      "document_count": 996,
      "index_exists": true
    }
  },
  "version": "1.0.0",
  "timestamp": "2025-09-13T13:25:42.123456Z"
}
```

**Example:**
```bash
curl "http://localhost:8002/health"
```

---

### 6. Auto-Complete Suggestions

Get search query suggestions based on indexed papers.

**Endpoint:** `GET /suggest`

**Parameters:**
- `q` (string, required): Partial query text
- `max_suggestions` (integer, optional): Maximum suggestions (default: 10)

**Response:**
```json
{
  "suggestions": [
    {
      "text": "transformer attention mechanism",
      "type": "title",
      "count": 15
    },
    {
      "text": "attention is all you need",
      "type": "paper",
      "paper_id": "1706.03762"
    }
  ],
  "query": "transfor"
}
```

**Example:**
```bash
curl "http://localhost:8002/suggest?q=transfor&max_suggestions=5"
```

---

### 7. Bulk Operations

#### Bulk Paper Details
Get details for multiple papers in a single request.

**Endpoint:** `POST /papers/bulk`

**Request Body:**
```json
{
  "paper_ids": ["1706.03762", "1810.04805", "2103.00020"]
}
```

**Response:**
```json
{
  "papers": [
    {
      "paper_id": "1706.03762",
      "title": "Attention Is All You Need",
      "authors": ["Ashish Vaswani"],
      "status": "found"
    },
    {
      "paper_id": "nonexistent",
      "status": "not_found",
      "error": "Paper not found in index"
    }
  ]
}
```

---

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "INVALID_SEARCH_MODE",
    "message": "Invalid search mode: 'invalid'. Supported modes: hybrid, semantic, bm25, title_only",
    "details": {
      "provided": "invalid",
      "supported": ["hybrid", "semantic", "bm25", "title_only"]
    }
  },
  "timestamp": "2025-09-13T13:25:42.123456Z"
}
```

### HTTP Status Codes
- `200 OK`: Success
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Paper not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: System error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Errors

**Invalid Search Mode (400)**
```json
{
  "error": {
    "code": "INVALID_SEARCH_MODE",
    "message": "Invalid search mode: 'invalid'. Supported modes: hybrid, semantic, bm25, title_only"
  }
}
```

**Paper Not Found (404)**
```json
{
  "error": {
    "code": "PAPER_NOT_FOUND",
    "message": "Paper with ID '999999' not found in index"
  }
}
```

**Elasticsearch Connection Error (503)**
```json
{
  "error": {
    "code": "ELASTICSEARCH_UNAVAILABLE",
    "message": "Cannot connect to Elasticsearch cluster"
  }
}
```

---

## Python Library Interface

### Installation
```bash
# Install the package in development mode
pip install -e .
```

### Basic Usage

#### Initialize Components
```python
from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.es_indexer import ESIndexer
from backend.services.search_service import SearchService

# Initialize search service
search_service = SearchService(
    es_host="localhost:9202",
    bge_model="BAAI/bge-large-en-v1.5"
)
```

#### Search Papers
```python
# Basic search
results = search_service.search(
    query="transformer attention mechanism",
    max_results=10,
    search_mode="hybrid"
)

# Advanced search with filters
results = search_service.search(
    query="neural machine translation",
    max_results=20,
    search_mode="semantic",
    categories=["cs.CL"],
    date_from="2020-01-01",
    date_to="2024-12-31"
)

# Process results
for result in results:
    print(f"Title: {result.title}")
    print(f"Score: {result.score:.3f}")
    print(f"Authors: {', '.join(result.authors)}")
    print("---")
```

#### Get Paper Details
```python
# Get single paper
paper = search_service.get_paper_details("1706.03762")
if paper:
    print(f"Title: {paper.title}")
    print(f"Abstract: {paper.abstract[:200]}...")
    print(f"Categories: {paper.categories}")
```

#### Find Similar Papers
```python
# Get papers similar to a reference paper
similar = search_service.suggest_papers(
    paper_id="1706.03762",
    max_results=5
)

for paper in similar:
    print(f"{paper.title} (Similarity: {paper.similarity_score:.3f})")
```

### Advanced Usage

#### Custom Embedder
```python
from data_pipeline.bge_embedder import BGEEmbedder

# Initialize with custom settings
embedder = BGEEmbedder(
    model_name="BAAI/bge-large-en-v1.5",
    device="cuda",  # Use GPU if available
    cache_dir="./models"
)

# Generate embeddings
texts = ["This is a paper about transformers"]
embeddings = embedder.encode(texts)

# Compute similarity
similarity = embedder.compute_similarity(embeddings, embeddings)
```

#### Custom Indexer
```python
from data_pipeline.es_indexer import ESIndexer

# Initialize indexer
indexer = ESIndexer(
    es_host="localhost:9202",
    index_name="my_papers",
    embedding_dim=1024
)

# Create index
indexer.create_index(force=True)

# Index documents
document = {
    "paper_id": "custom_001",
    "title": "Custom Paper",
    "content": "Paper content...",
    "title_embedding": embeddings[0].tolist()
}
indexer.index_document(document)

# Search
results = indexer.search(
    query="custom query",
    query_embedding=query_embeddings,
    size=10
)
```

#### Batch Processing
```python
from data_pipeline.ingest_papers import PaperProcessor

# Initialize processor
processor = PaperProcessor(
    es_host="localhost:9202",
    bge_model="BAAI/bge-large-en-v1.5",
    json_metadata_dir="/path/to/json/files"
)

# Ingest directory
processor.ingest_directory(
    markdown_dir=Path("./data/processed/markdown"),
    batch_size=10,
    max_files=100
)
```

### Data Models

#### SearchResult
```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    score: float
    categories: List[str]
    publish_date: Optional[str]
    word_count: int
    has_images: bool
    markdown_path: Optional[str]
    pdf_path: Optional[str]
```

#### Paper
```python
@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    categories: List[str]
    publish_date: Optional[str]
    word_count: int
    chunk_count: int
    has_images: bool
    pdf_size: int
    downloaded_at: Optional[str]
    indexed_at: Optional[str]
    markdown_path: Optional[str]
    pdf_path: Optional[str]
```

### Configuration

#### Environment Variables
```python
import os

# Configure via environment
os.environ["ES_HOST"] = "localhost:9202"
os.environ["BGE_MODEL_NAME"] = "BAAI/bge-large-en-v1.5"
os.environ["BGE_CACHE_DIR"] = "./models"

# Use in application
from backend.config import settings
search_service = SearchService(es_host=settings.es_host)
```

#### Programmatic Configuration
```python
from backend.services.search_service import SearchService

# Custom configuration
config = {
    "es_host": "localhost:9202",
    "bge_model": "BAAI/bge-large-en-v1.5",
    "chunk_size": 512,
    "chunk_overlap": 100,
    "hybrid_weights": {"bm25": 0.4, "semantic": 0.6}
}

search_service = SearchService(**config)
```

---

## Rate Limiting

### Default Limits
- Search requests: 100 per minute per IP
- Paper details: 200 per minute per IP
- Health checks: 1000 per minute per IP

### Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

---

## WebSocket API (Future)

### Real-time Search Updates
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8002/ws/search');

// Send search query
ws.send(JSON.stringify({
  type: 'search',
  query: 'transformer attention',
  max_results: 10
}));

// Receive results
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Search results:', data.results);
};
```

---

## SDKs

### JavaScript/TypeScript
```typescript
import { PaperSearchClient } from 'paper-search-client';

const client = new PaperSearchClient({
  baseURL: 'http://localhost:8002'
});

// Search papers
const results = await client.search({
  query: 'transformer attention',
  maxResults: 10,
  searchMode: 'hybrid'
});

// Get paper details
const paper = await client.getPaper('1706.03762');
```

### Python
```python
from paper_search_client import PaperSearchClient

client = PaperSearchClient(base_url="http://localhost:8002")

# Search papers
results = client.search(
    query="transformer attention",
    max_results=10,
    search_mode="hybrid"
)

# Get paper details
paper = client.get_paper("1706.03762")
```

---

## Testing

### API Testing
```bash
# Run API tests
python -m pytest tests/api/ -v

# Test specific endpoint
python -m pytest tests/api/test_search_endpoint.py -v
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8002
```

---

## Versioning

The API follows semantic versioning (SemVer):
- **Major version**: Breaking changes
- **Minor version**: New features, backward compatible
- **Patch version**: Bug fixes

Current version: `1.0.0`

### Version Header
All responses include a version header:
```
X-API-Version: 1.0.0
```

### Deprecation Notice
Deprecated endpoints include a warning header:
```
X-Deprecation-Warning: This endpoint will be removed in v2.0.0. Use /v2/search instead.
```