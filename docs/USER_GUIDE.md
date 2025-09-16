# Paper Search Engine User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Ingestion Pipeline](#ingestion-pipeline)
5. [Search Interface](#search-interface)
6. [API Usage](#api-usage)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- 8GB+ RAM (4GB for Elasticsearch)
- 10GB+ disk space

### 1. Clone and Setup
```bash
git clone <repository-url>
cd paper-search-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Services
```bash
# Copy environment configuration
cp .env.example .env

# Start Elasticsearch and MinIO
docker-compose up -d paper-search-elasticsearch paper-search-minio

# Wait for services to be ready
curl http://localhost:9202/_cluster/health
```

### 3. Prepare Your Data
Ensure you have:
- **Markdown files**: Papers in `/data/processed/markdown/`
- **JSON metadata**: Rich metadata in `/data/pdfs/` (optional but recommended)

### 4. Ingest Papers
```bash
# Quick test with 10 papers
python data_pipeline/ingest_papers.py --max-files 10 --es-host localhost:9202

# Full ingestion
python data_pipeline/ingest_papers.py --es-host localhost:9202
```

### 5. Start Search Interface
```bash
# Start backend API
docker-compose up -d paper-search-backend

# Start Streamlit UI
docker-compose up -d paper-search-ui

# Access at http://localhost:8503
```

## Installation

### System Requirements

**Minimum:**
- 4 CPU cores
- 8GB RAM
- 20GB storage
- Python 3.9+

**Recommended:**
- 8+ CPU cores
- 16GB+ RAM
- SSD storage
- GPU for faster embedding generation

### Docker Installation

The easiest way to run the system:

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs paper-search-elasticsearch
```

### Manual Installation

For development or customization:

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Elasticsearch manually
docker run -d --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0

# 3. Run components individually
python data_pipeline/ingest_papers.py
python backend/api/main.py
streamlit run streamlit_app/app.py
```

## Data Preparation

### Supported Formats

#### 1. Markdown Files (.md)
Papers converted to markdown format:
```
data/processed/markdown/
├── 2210.14275.md     # ArXiv ID as filename
├── 1706.03762.md
└── 2301.08727.md
```

**Markdown Format Example:**
```markdown
# Paper Title

Authors: John Doe, Jane Smith

## Abstract

This paper presents...

## Introduction

Recent advances in...
```

#### 2. JSON Metadata (Recommended)
Rich metadata files for enhanced search accuracy:
```
data/pdfs/
├── 2210.14275.json
├── 1706.03762.json
└── 2301.08727.json
```

**JSON Format Example:**
```json
{
  "paper_id": "2210.14275",
  "title": "Attention Is All You Need",
  "authors": "John Doe, Jane Smith",
  "abstract": "Complete abstract text...",
  "categories": "cs.CL cs.AI",
  "downloaded_at": "2025-09-12T20:18:28.510962",
  "pdf_size": 1024000
}
```

### Data Quality Guidelines

#### Markdown Files
- Use clear section headers (`#`, `##`, `###`)
- Include complete abstracts and introductions
- Maintain consistent formatting
- Include author information

#### JSON Metadata
- Ensure `paper_id` matches markdown filename
- Provide complete author strings or arrays
- Include accurate abstracts
- Use standard category codes (cs.CL, cs.AI, etc.)

### Data Sources

The system works with papers from:
- **ArXiv**: Direct downloads and metadata
- **Academic databases**: Exported papers
- **Custom collections**: Any markdown + metadata

## Ingestion Pipeline

### Basic Ingestion

#### Simple Ingestion
```bash
# Ingest first 10 files for testing
python data_pipeline/ingest_papers.py --max-files 10

# Full ingestion with default settings
python data_pipeline/ingest_papers.py
```

#### Advanced Options
```bash
python data_pipeline/ingest_papers.py \
  --markdown-dir ./data/processed/markdown \
  --json-metadata-dir ./data/pdfs \
  --es-host localhost:9202 \
  --bge-model BAAI/bge-large-en-v1.5 \
  --batch-size 10 \
  --chunk-size 512 \
  --chunk-overlap 100 \
  --max-files 100
```

### Monitoring Ingestion

#### Progress Tracking
The ingestion process shows:
- Paper processing progress bar
- Chunking statistics per paper
- Indexing batch confirmations
- Final index statistics

#### Example Output
```
INFO:__main__:Found 996 markdown files to process
INFO:data_pipeline.es_indexer:Created index: papers
Processing papers: 100%|██████████| 996/996 [2:15:30<00:00, 8.17s/it]
INFO:data_pipeline.es_indexer:Indexed 996 documents, 0 failed
INFO:__main__:Index statistics: {'document_count': 996, 'index_size_mb': 245.7}
```

### Resuming Ingestion

If ingestion is interrupted:
```bash
# Resume from specific paper ID
python data_pipeline/ingest_papers.py --resume-from 2210.14275

# The system will skip already processed papers
```

### Batch Processing

#### Small Batches (Development)
```bash
python data_pipeline/ingest_papers.py --batch-size 5 --max-files 50
```

#### Large Batches (Production)
```bash
python data_pipeline/ingest_papers.py --batch-size 20
```

## Search Interface

### Web Interface (Streamlit)

Access the search interface at `http://localhost:8503`

#### Main Search Features
1. **Search Box**: Enter queries in natural language
2. **Search Modes**:
   - Hybrid (recommended): BM25 + semantic search
   - Semantic: Pure embedding-based search
   - BM25: Traditional keyword search
   - Title Only: Search only paper titles

3. **Filters**:
   - Categories (cs.CL, cs.AI, etc.)
   - Date ranges
   - Author names

4. **Results Display**:
   - Relevance scores
   - Author information
   - Abstract previews
   - Paper categories
   - Publication dates

#### Search Tips

**For Best Results:**
- Use descriptive, natural language queries
- Include key concepts and terms
- Try different search modes for comparison

**Example Queries:**
- "transformer attention mechanism neural networks"
- "natural language processing BERT"
- "computer vision object detection"
- "machine learning interpretability"

### Direct API Access

#### Search Endpoint
```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "attention mechanism",
    "max_results": 10,
    "search_mode": "hybrid"
  }'
```

#### Response Format
```json
{
  "results": [
    {
      "paper_id": "1706.03762",
      "title": "Attention Is All You Need",
      "authors": ["Ashish Vaswani", "Noam Shazeer"],
      "abstract": "The dominant sequence transduction...",
      "score": 0.95,
      "categories": ["cs.CL"],
      "publish_date": "2017-06-12"
    }
  ],
  "total_found": 42,
  "search_time_ms": 156
}
```

### Command Line Search

#### Quick Search Script
```bash
# Test search functionality
python test_search_simple.py
```

#### Example Usage
```python
from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.es_indexer import ESIndexer

# Initialize components
embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")
indexer = ESIndexer(es_host="localhost:9202")

# Search
query = "machine learning"
query_embedding = embedder.encode(query)
results = indexer.search(
    query=query,
    query_embedding=query_embedding,
    size=10,
    use_semantic=True,
    use_bm25=True
)
```

## API Usage

### Backend API

The FastAPI backend provides RESTful endpoints:

#### Start Backend
```bash
docker-compose up -d paper-search-backend

# Or manually
cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Available Endpoints

**Search Papers**
```http
POST /search
Content-Type: application/json

{
  "query": "string",
  "max_results": 10,
  "search_mode": "hybrid|semantic|bm25|title_only",
  "categories": ["cs.CL", "cs.AI"],
  "date_from": "2020-01-01",
  "date_to": "2024-12-31"
}
```

**Get Paper Details**
```http
GET /papers/{paper_id}
```

**Search Statistics**
```http
GET /stats
```

**Health Check**
```http
GET /health
```

### Python Client Example

```python
import requests

# Search for papers
response = requests.post("http://localhost:8002/search", json={
    "query": "attention mechanism transformer",
    "max_results": 5,
    "search_mode": "hybrid"
})

results = response.json()
for paper in results["results"]:
    print(f"{paper['title']} (Score: {paper['score']:.3f})")
```

## Configuration

### Environment Variables

Create `.env` file with:
```bash
# Elasticsearch
ES_HOST=localhost:9202

# BGE Model
BGE_MODEL_NAME=BAAI/bge-large-en-v1.5
BGE_CACHE_DIR=./models

# Search Configuration
DEFAULT_SEARCH_MODE=hybrid
MAX_SEARCH_RESULTS=20

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=100

# MinIO (Optional)
MINIO_ENDPOINT=localhost:9002
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### Docker Configuration

#### Memory Settings
For large collections, increase Elasticsearch memory:
```yaml
# docker-compose.yml
services:
  paper-search-elasticsearch:
    environment:
      - "ES_JAVA_OPTS=-Xms4g -Xmx4g"  # Increase from 2g
```

#### Port Configuration
Customize ports to avoid conflicts:
```yaml
services:
  paper-search-elasticsearch:
    ports:
      - "9202:9200"  # Change external port

  paper-search-backend:
    ports:
      - "8002:8000"  # Change external port
```

### Search Tuning

#### Hybrid Search Weights
Adjust in `backend/services/search_service.py`:
```python
# Modify score combination weights
combined_score = 0.4 * bm25_score + 0.6 * semantic_score
```

#### Chunk Size Optimization
For different document types:
- **Short papers**: `--chunk-size 256`
- **Long papers**: `--chunk-size 512` (default)
- **Very long papers**: `--chunk-size 1024`

## Troubleshooting

### Common Issues

#### 1. Elasticsearch Connection Errors
```
ConnectionError: Cannot connect to Elasticsearch
```

**Solutions:**
- Check if Elasticsearch is running: `curl http://localhost:9202`
- Verify port configuration in docker-compose.yml
- Wait for Elasticsearch to fully start (can take 30-60 seconds)

#### 2. BGE Model Download Issues
```
OSError: Unable to load model from transformers
```

**Solutions:**
- Ensure internet connection for model download
- Pre-download model: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')"`
- Check available disk space (model is ~1.3GB)

#### 3. Out of Memory Errors
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--batch-size 5`
- Use CPU-only mode (slower but more memory efficient)
- Increase Docker memory limits

#### 4. JSON Metadata Loading Issues
```
WARNING: JSON metadata not found
```

**Solutions:**
- Verify JSON files exist in specified directory
- Check filename matching (paper_id.json)
- Validate JSON format with `python test_json_metadata.py`

### Performance Issues

#### Slow Ingestion
- **Reduce batch size**: `--batch-size 5`
- **Use GPU**: Ensure CUDA available for BGE
- **Increase memory**: More RAM for Elasticsearch

#### Slow Search
- **Check index size**: Large indices need more memory
- **Reduce result size**: Use `max_results` parameter
- **Optimize queries**: Use specific terms rather than very general ones

### Docker Issues

#### Container Health Checks
```bash
# Check all containers
docker-compose ps

# Check specific service logs
docker-compose logs paper-search-elasticsearch

# Restart problematic service
docker-compose restart paper-search-elasticsearch
```

#### Volume Permissions
```bash
# Fix Elasticsearch data permissions
sudo chown -R 1000:1000 ./data/es-data
```

## Performance Optimization

### Hardware Recommendations

#### For Small Collections (<1,000 papers)
- 4 CPU cores
- 8GB RAM
- Standard HDD acceptable

#### For Medium Collections (1,000-10,000 papers)
- 8 CPU cores
- 16GB RAM
- SSD recommended

#### For Large Collections (>10,000 papers)
- 16+ CPU cores
- 32GB+ RAM
- NVMe SSD required
- GPU for BGE acceleration

### Software Optimization

#### Elasticsearch Tuning
```yaml
# docker-compose.yml
environment:
  - "ES_JAVA_OPTS=-Xms8g -Xmx8g"  # Large heap
  - "index.refresh_interval=30s"   # Less frequent refresh
  - "index.number_of_replicas=0"   # No replicas for single node
```

#### BGE Optimization
```python
# Use GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Batch processing
embedder = BGEEmbedder(device=device, batch_size=32)
```

#### Chunking Optimization
```bash
# Larger chunks for faster processing
python data_pipeline/ingest_papers.py --chunk-size 1024 --chunk-overlap 50

# Smaller chunks for better precision
python data_pipeline/ingest_papers.py --chunk-size 256 --chunk-overlap 100
```

### Monitoring

#### System Resources
```bash
# Monitor Elasticsearch
curl -s "http://localhost:9202/_cluster/stats?pretty"

# Monitor container resources
docker stats

# Monitor disk usage
df -h
```

#### Search Performance
```bash
# Check search timing
curl -w "%{time_total}" -s "http://localhost:9202/papers/_search" > /dev/null
```

### Scaling Strategies

#### Vertical Scaling
1. Increase Elasticsearch heap size
2. Add more CPU cores
3. Use faster storage (NVMe SSD)
4. Add GPU for embedding acceleration

#### Horizontal Scaling
1. Multi-node Elasticsearch cluster
2. Multiple backend API instances
3. Load balancer for distribution
4. Separate embedding service

For production deployments, consider:
- Kubernetes orchestration
- Dedicated Elasticsearch cluster
- CDN for static assets
- Monitoring and alerting (Prometheus/Grafana)