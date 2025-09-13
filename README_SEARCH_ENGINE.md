# Paper Search Engine with BGE Embeddings

A high-accuracy paper search engine using Elasticsearch, BGE embeddings, and MinIO for optimal search results.

## Architecture

- **BGE Embeddings**: BAAI/bge-large-en-v1.5 for semantic search
- **Elasticsearch**: Full-text and vector search
- **MinIO**: Object storage for PDFs and images
- **Docker**: Containerized services
- **Hybrid Search**: Combines BM25 and semantic search for accuracy

## Quick Start

### 1. Start Services

```bash
# Copy environment config
cp .env.example .env

# Start Elasticsearch and MinIO
docker-compose up -d elasticsearch minio

# Wait for services to be ready (check with docker-compose ps)
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Test Implementation

```bash
python test_implementation.py
```

### 4. Ingest Papers

```bash
# Ingest all markdown files with JSON metadata
python data_pipeline/ingest_papers.py --markdown-dir ./data/processed/markdown --json-metadata-dir /Users/admin/code/cazoodle/data/pdfs

# Or ingest first 10 files for testing
python data_pipeline/ingest_papers.py --max-files 10

# Resume from specific paper
python data_pipeline/ingest_papers.py --resume-from 2210.14275

# Test JSON metadata loading first
python test_json_metadata.py
```

### 5. Test Search

```bash
# Test search service
python backend/services/search_service.py
```

## Key Components

### BGE Embedder (`data_pipeline/bge_embedder.py`)
- Loads BAAI/bge-large-en-v1.5 model
- Generates 1024-dimensional embeddings
- Optimized for document retrieval

### Document Chunker (`data_pipeline/document_chunker.py`)
- Smart chunking respecting markdown sections
- 512 tokens per chunk with 100 token overlap
- Preserves document structure

### ES Indexer (`data_pipeline/es_indexer.py`)
- Creates optimized Elasticsearch mappings
- Stores multiple embedding types (title, abstract, content)
- Supports hybrid search queries

### Search Service (`backend/services/search_service.py`)
- Hybrid search combining BM25 + semantic search
- Multiple search modes: hybrid, semantic, bm25, title_only
- Result ranking and filtering

## Search Modes

### Hybrid Search (Default)
Combines BM25 text search with BGE semantic search:
- BM25 weight: 40%
- Semantic weight: 60%

### Semantic Search
Pure embedding-based similarity search using BGE vectors.

### BM25 Search
Traditional full-text search with fuzzy matching.

### Title Only Search
Search only in paper titles for precise matches.

## Usage Examples

```python
from backend.services.search_service import SearchService

# Initialize service
search_service = SearchService()

# Search with different modes
results = search_service.search(
    query="transformer attention mechanism",
    max_results=10,
    search_mode="hybrid"
)

# Get paper details
paper = search_service.get_paper_details("2210.14275")

# Get similar papers
suggestions = search_service.suggest_papers("2210.14275")
```

## Configuration

Key environment variables in `.env`:

```bash
# Elasticsearch
ES_HOST=localhost:9200

# BGE Model
BGE_MODEL_NAME=BAAI/bge-large-en-v1.5
BGE_CACHE_DIR=./models

# Search tuning
DEFAULT_SEARCH_MODE=hybrid
MAX_SEARCH_RESULTS=20
CHUNK_SIZE=512
CHUNK_OVERLAP=100
```

## Data Structure

### Elasticsearch Document Schema
```json
{
  "paper_id": "2210.14275",
  "title": "Paper title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Abstract text",
  "content": "Full markdown content",
  "title_embedding": [1024 dimensions],
  "abstract_embedding": [1024 dimensions],
  "content_chunks": [
    {
      "text": "chunk text",
      "embedding": [1024 dimensions],
      "chunk_index": 0
    }
  ],
  "categories": ["cs.CL"],
  "publish_date": "2022-10-25",
  "pdf_path": "/data/pdfs/2210.14275.pdf",
  "pdf_size": 1024000,
  "downloaded_at": "2025-09-12T20:18:28.510962",
  "word_count": 5000,
  "has_images": true
}
```

### JSON Metadata Source
The system now uses rich metadata from JSON files located in `/Users/admin/code/cazoodle/data/pdfs/`:

```json
{
  "paper_id": "1911.13207",
  "title": "A concrete example of inclusive design: deaf-oriented accessibility",
  "authors": "Claudia S. Bianchini, Fabrizio Borgia, Maria de Marsico",
  "abstract": "One of the continuing challenges of Human Computer Interaction...",
  "categories": "cs.HC cs.CL",
  "downloaded_at": "2025-09-12T20:18:28.510962",
  "pdf_size": 701446
}
```

**Metadata Priority:**
1. **JSON metadata** (primary) - Rich, accurate data from API/crawling
2. **Markdown parsing** (fallback) - Text extraction if JSON missing
3. **Default values** - Sensible defaults for missing fields

## Performance Tips

1. **Batch Processing**: Use `--batch-size 20` for faster ingestion
2. **GPU Acceleration**: BGE will automatically use GPU if available
3. **Memory**: Increase ES heap size for large datasets in docker-compose.yml
4. **Chunking**: Adjust `CHUNK_SIZE` based on your average document length

## Troubleshooting

### Elasticsearch Connection Issues
```bash
# Check ES health
curl http://localhost:9200/_cluster/health

# Restart ES
docker-compose restart elasticsearch
```

### BGE Model Download Issues
```bash
# Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')"
```

### Out of Memory
- Reduce batch size: `--batch-size 5`
- Reduce chunk size: `CHUNK_SIZE=256`
- Increase ES heap: `ES_JAVA_OPTS=-Xms4g -Xmx4g` in docker-compose.yml

## Next Steps

1. Add MinIO integration for PDF storage
2. Implement result ranking improvements
3. Add query expansion with synonyms
4. Create web interface for search
5. Add paper recommendation system