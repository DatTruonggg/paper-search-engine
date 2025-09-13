# Paper Search Engine with BGE Embeddings

A high-accuracy academic paper search engine using BGE embeddings, Elasticsearch hybrid search, and rich JSON metadata integration for optimal semantic and full-text search results.

## 🌟 Features

- **🔍 Hybrid Search**: Combines BM25 (40%) + BGE semantic search (60%) for optimal accuracy
- **🧠 BGE Embeddings**: BAAI/bge-large-en-v1.5 model with 1024-dimensional vectors
- **📊 Rich Metadata**: JSON-first metadata with markdown fallback
- **🚀 Multi-Modal Search**: Title, abstract, and content-level semantic search
- **⚡ High Performance**: Elasticsearch with optimized indexing and search
- **🐳 Docker Ready**: Containerized deployment with Docker Compose
- **🎯 Smart Chunking**: Section-aware document segmentation (512 tokens, 100 overlap)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◄───┤   FastAPI       │◄───┤   Elasticsearch │
│   (Port 8503)   │    │   Backend       │    │   (Port 9202)   │
└─────────────────┘    │   (Port 8002)   │    └─────────────────┘
                       └─────────────────┘               ▲
                                │                        │
                                ▼                        │
                       ┌─────────────────┐               │
                       │   BGE Embedder  │               │
                       │   1024-dim      │               │
                       └─────────────────┘               │
                                                         │
┌─────────────────┐    ┌─────────────────┐               │
│ Markdown + JSON │───►│   Ingestion     │───────────────┘
│   Data Sources  │    │   Pipeline      │
└─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+, Docker & Docker Compose
- 8GB+ RAM, 10GB+ disk space

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd paper-search-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
```

### 2. Start Services
```bash
# Start Elasticsearch (note: uses port 9202 to avoid conflicts)
docker-compose up -d paper-search-elasticsearch

# Verify Elasticsearch is running
curl http://localhost:9202/_cluster/health
```

### 3. Ingest Papers
```bash
# Quick test with 10 papers
python data_pipeline/ingest_papers.py --max-files 10 --es-host localhost:9202

# Full ingestion with JSON metadata
python data_pipeline/ingest_papers.py \
  --markdown-dir ./data/processed/markdown \
  --json-metadata-dir /Users/admin/code/cazoodle/data/pdfs \
  --es-host localhost:9202
```

### 4. Test Search
```bash
# Test search functionality
python test_search_simple.py

# Or test via API
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "max_results": 5}'
```

### 5. Launch Web Interface
```bash
# Start all services
docker-compose up -d

# Access web interface at http://localhost:8503
```

## 📚 Documentation

### Core Guides
- **[📖 User Guide](docs/USER_GUIDE.md)** - Complete installation and usage instructions
- **[🏗️ Architecture](docs/ARCHITECTURE.md)** - System design and technical details
- **[🔌 API Reference](docs/API_REFERENCE.md)** - REST API and Python library documentation
- **[📋 Documentation Index](docs/README.md)** - All documentation overview

### Quick Links
- [Installation Guide](docs/USER_GUIDE.md#installation)
- [Search Interface](docs/USER_GUIDE.md#search-interface)
- [API Endpoints](docs/API_REFERENCE.md#endpoints)
- [Performance Tuning](docs/USER_GUIDE.md#performance-optimization)

## 🔍 Search Capabilities

### Search Modes
- **Hybrid** (Default): BM25 + Semantic search combination
- **Semantic**: Pure BGE embedding similarity
- **BM25**: Traditional full-text search
- **Title Only**: Search paper titles only

### Search Examples
```python
from backend.services.search_service import SearchService

# Initialize search service
search_service = SearchService(es_host="localhost:9202")

# Semantic search for concepts
results = search_service.search(
    query="attention mechanism transformers",
    search_mode="hybrid",
    max_results=10
)

# Get paper details
paper = search_service.get_paper_details("1706.03762")

# Find similar papers
similar = search_service.suggest_papers("1706.03762")
```

## 📊 Data Sources & Format

### Supported Input Formats

#### 1. Markdown Files
```
data/processed/markdown/
├── 1706.03762.md    # ArXiv ID as filename
├── 2210.14275.md
└── 2112.08466.md
```

#### 2. JSON Metadata (Recommended)
```json
{
  "paper_id": "1706.03762",
  "title": "Attention Is All You Need",
  "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar",
  "abstract": "The dominant sequence transduction models...",
  "categories": "cs.CL cs.AI",
  "downloaded_at": "2025-09-12T20:18:28.510962",
  "pdf_size": 1024000
}
```

### Metadata Priority System
1. **JSON metadata** (primary) - Rich, structured data
2. **Markdown parsing** (fallback) - Extracted from content
3. **Default values** - Sensible defaults for missing fields

## ⚡ Performance & Scale

### Tested Performance
- **Ingestion**: ~10-15 papers/minute
- **Search Latency**: <500ms hybrid search
- **Index Size**: ~6MB per 1000 papers
- **Memory Usage**: 2-4GB typical deployment

### Optimization Tips
- Use GPU for BGE acceleration
- Increase Elasticsearch heap for large collections
- Adjust batch sizes based on system resources
- Use SSD storage for optimal performance

## 🛠️ Configuration

### Key Environment Variables
```bash
# Elasticsearch
ES_HOST=localhost:9202

# BGE Model
BGE_MODEL_NAME=BAAI/bge-large-en-v1.5
BGE_CACHE_DIR=./models

# Search Configuration
DEFAULT_SEARCH_MODE=hybrid
MAX_SEARCH_RESULTS=20
CHUNK_SIZE=512
CHUNK_OVERLAP=100

# Docker Service Ports (to avoid conflicts)
ES_PORT=9202
BACKEND_PORT=8002
UI_PORT=8503
```

## 🐳 Docker Services

All services use prefixed names to avoid conflicts:
- `paper-search-elasticsearch` (Port 9202)
- `paper-search-backend` (Port 8002)
- `paper-search-ui` (Port 8503)
- `paper-search-minio` (Ports 9002/9003)

## 🧪 Testing

### Unit Tests
```bash
# Run core component tests
source venv/bin/activate
python -m pytest tests/unit/ -v

# Test specific components
python -m pytest tests/unit/test_document_chunker.py -v
```

### Integration Testing
```bash
# Test complete pipeline
python test_json_metadata.py
python test_search_simple.py

# Test ingestion with sample data
python data_pipeline/ingest_papers.py --max-files 5 --es-host localhost:9202
```

## 🚨 Troubleshooting

### Common Issues

#### Elasticsearch Connection
```bash
# Check ES health
curl http://localhost:9202/_cluster/health

# Restart if needed
docker-compose restart paper-search-elasticsearch
```

#### BGE Model Download
```bash
# Pre-download model (1.3GB)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')"
```

#### Memory Issues
- Reduce batch size: `--batch-size 5`
- Increase Docker memory limits
- Use `ES_JAVA_OPTS=-Xms4g -Xmx4g` for large collections

## 🗂️ Project Structure
```
paper-search-engine/
├── data_pipeline/           # Core ingestion pipeline
│   ├── bge_embedder.py     # BGE embedding generation
│   ├── document_chunker.py # Smart text chunking
│   ├── es_indexer.py       # Elasticsearch operations
│   └── ingest_papers.py    # Main ingestion orchestrator
├── backend/                # FastAPI backend
│   ├── api/               # REST API endpoints
│   └── services/          # Search and business logic
├── streamlit_app/         # Web interface
├── tests/                 # Unit and integration tests
├── docs/                  # Comprehensive documentation
└── docker-compose.yml     # Service orchestration
```

## 🎯 Use Cases

- **Academic Research**: Semantic paper discovery and literature review
- **Educational**: Course material and research topic exploration
- **Industry**: Technical document search and knowledge management
- **Libraries**: Digital collection search and discovery

## 🛣️ Roadmap

### Completed ✅
- BGE embedding integration
- Hybrid search implementation
- JSON metadata priority system
- Docker containerization
- Comprehensive documentation
- Unit test coverage

### Short Term 🔄
- MinIO PDF storage integration
- Advanced result ranking
- Query auto-complete
- Performance optimizations

### Long Term 🔮
- Multi-language support
- Real-time indexing
- Citation network analysis
- ML-based relevance tuning

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see:
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Development Setup](docs/USER_GUIDE.md#installation)

---

*Built with ❤️ for the academic and research community. Powered by BGE embeddings and Elasticsearch.*