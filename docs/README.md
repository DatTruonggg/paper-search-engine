# Paper Search Engine Documentation

## Overview

Welcome to the Paper Search Engine documentation! This system provides state-of-the-art semantic and full-text search capabilities for academic papers using BGE embeddings and Elasticsearch.

## Quick Navigation

### 📚 Core Documentation
- **[Architecture Guide](ARCHITECTURE.md)** - Detailed system architecture and design decisions
- **[User Guide](USER_GUIDE.md)** - Complete installation and usage instructions
- **[API Reference](API_REFERENCE.md)** - REST API endpoints and Python library interface

### 🚀 Getting Started
1. Follow the [Quick Start](USER_GUIDE.md#quick-start) section in the User Guide
2. Review the [Architecture Overview](ARCHITECTURE.md#overview) to understand the system
3. Explore the [API Reference](API_REFERENCE.md) for programmatic access

### 🔧 For Developers
- **[Testing Guide](TESTING.md)** - Running unit tests and integration tests
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow and standards
- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment instructions

## Key Features

### 🔍 Advanced Search
- **Hybrid Search**: Combines BM25 and semantic search (40%/60% weighting)
- **Multiple Search Modes**: Semantic, BM25, Title-only, and Hybrid
- **Rich Metadata**: JSON-based metadata with fallback to markdown parsing
- **Smart Filtering**: By categories, date ranges, and authors

### 🧠 Intelligent Processing
- **BGE Embeddings**: State-of-the-art BAAI/bge-large-en-v1.5 model
- **Smart Chunking**: 512 tokens with 100 token overlap, section-aware
- **Multi-level Embeddings**: Title, abstract, and content chunk embeddings
- **Semantic Similarity**: Find papers similar to a given paper

### ⚡ High Performance
- **Elasticsearch**: Full-text and vector search at scale
- **Batch Processing**: Efficient ingestion of large paper collections
- **Docker Deployment**: Easy setup and scaling
- **Production Ready**: Health checks, monitoring, and error handling

## System Requirements

### Minimum (Development)
- 4 CPU cores, 8GB RAM, 20GB storage
- Python 3.9+, Docker & Docker Compose

### Recommended (Production)
- 8+ CPU cores, 16GB+ RAM, SSD storage
- Optional GPU for faster embedding generation

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd paper-search-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start services
cp .env.example .env
docker-compose up -d paper-search-elasticsearch

# 3. Ingest papers (test with 10 papers)
python data_pipeline/ingest_papers.py --max-files 10 --es-host localhost:9202

# 4. Start search interface
docker-compose up -d paper-search-ui
# Access at http://localhost:8503
```

## Architecture Highlights

### Data Flow
```
JSON Metadata + Markdown → Chunking → BGE Embeddings → Elasticsearch → Search API → Web UI
```

### Search Pipeline
- **Query Processing**: Natural language understanding
- **Embedding Generation**: Semantic vector representation
- **Hybrid Search**: BM25 + semantic similarity scoring
- **Result Ranking**: Relevance-based ordering with metadata enrichment

### Core Components
- **BGE Embedder**: High-quality semantic embeddings
- **Document Chunker**: Smart text segmentation
- **ES Indexer**: Elasticsearch management and search
- **Search Service**: Unified search interface
- **FastAPI Backend**: REST API server
- **Streamlit UI**: Interactive web interface

## Use Cases

### Academic Research
- Find papers by semantic similarity
- Explore research topics and trends
- Discover related work and citations
- Search across large paper collections

### Educational Applications
- Course material discovery
- Literature review assistance
- Research topic exploration
- Academic paper recommendations

### Industry Applications
- Technical document search
- Research and development support
- Competitive analysis
- Knowledge management

## Performance Metrics

### Search Performance
- **Latency**: <500ms for hybrid search
- **Throughput**: 10+ queries per second
- **Accuracy**: High relevance with semantic understanding

### Scalability
- **Documents**: Tested with 1000+ papers
- **Index Size**: ~6MB per 1000 papers
- **Memory Usage**: 2-4GB for typical deployments

## Support and Community

### Documentation
- Complete guides for users and developers
- API reference with examples
- Architecture deep-dives
- Performance optimization tips

### Testing
- Comprehensive unit test coverage
- Integration tests for full pipeline
- Performance benchmarks
- Example queries and results

## Contributing

We welcome contributions! Please see:
- [Contributing Guide](CONTRIBUTING.md) for development workflow
- [Architecture Guide](ARCHITECTURE.md) for system understanding
- [Testing Guide](TESTING.md) for running tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

### Short Term
- [ ] MinIO integration for PDF storage
- [ ] Advanced result ranking algorithms
- [ ] Query expansion and auto-complete
- [ ] Performance optimizations

### Long Term
- [ ] Multi-language support
- [ ] Real-time indexing pipeline
- [ ] Machine learning relevance tuning
- [ ] Citation network analysis
- [ ] Collaborative filtering

## Contact

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review existing discussions

---

*Built with ❤️ for the academic and research community*