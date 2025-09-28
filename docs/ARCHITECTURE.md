# Paper Search Engine Architecture

## Overview

The Paper Search Engine is a sophisticated document retrieval system that combines BGE embeddings, Elasticsearch hybrid search, and rich metadata integration to provide highly accurate semantic and full-text search capabilities for academic papers.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Elasticsearch │
│                 │◄───┤   Backend       │◄───┤   Index         │
│   - Search UI   │    │                 │    │   - Papers      │
│   - Results     │    │   - Search API  │    │   - Vectors     │
│   - Filters     │    │   - Hybrid      │    │   - Metadata    │
└─────────────────┘    │     Search      │    └─────────────────┘
                       └─────────────────┘               ▲
                                │                        │
                                ▼                        │
                       ┌─────────────────┐               │
                       │   BGE Embedder  │               │
                       │                 │               │
                       │   - BAAI/bge-   │               │
                       │     large-en-   │               │
                       │     v1.5        │               │
                       │   - 1024 dims   │               │
                       └─────────────────┘               │
                                                         │
┌─────────────────┐    ┌─────────────────┐               │
│   Data Sources  │    │   Ingestion     │               │
│                 │───►│   Pipeline      │───────────────┘
│   - Markdown    │    │                 │
│   - JSON Meta   │    │   - Chunking    │
│   - PDFs        │    │   - Embedding   │
└─────────────────┘    │   - Indexing    │
                       └─────────────────┘
```

## Core Components

### 1. Data Pipeline (`data_pipeline/`)

#### BGE Embedder (`bge_embedder.py`)
- **Purpose**: Generate high-quality embeddings for semantic search
- **Model**: BAAI/bge-large-en-v1.5 (1024 dimensions)
- **Features**:
  - Batch processing for efficiency
  - Query-optimized embeddings with retrieval prefixes
  - GPU acceleration support
  - Similarity computation utilities

**Key Methods**:
```python
def encode(texts: List[str]) -> np.ndarray
def encode_queries(queries: List[str]) -> np.ndarray
def compute_similarity(embeddings1, embeddings2) -> np.ndarray
```

#### Document Chunker (`document_chunker.py`)
- **Purpose**: Intelligently split documents for optimal retrieval
- **Strategy**:
  - 512 tokens per chunk with 100 token overlap
  - Respects markdown section boundaries
  - Preserves document structure and context

**Key Features**:
- Tiktoken-based tokenization
- Sentence boundary awareness
- Section-aware chunking
- Overlap management for context preservation

#### ES Indexer (`es_indexer.py`)
- **Purpose**: Manage Elasticsearch index and search operations
- **Index Design**:
  - Dense vector fields for embeddings (cosine similarity)
  - Text fields with custom analyzer
  - Nested content chunks with individual embeddings
  - Rich metadata fields

**Index Schema**:
```json
{
  "paper_id": "keyword",
  "title": "text + keyword",
  "authors": "keyword",
  "abstract": "text",
  "content": "text",
  "title_embedding": "dense_vector[1024]",
  "abstract_embedding": "dense_vector[1024]",
  "content_chunks": {
    "nested": {
      "text": "text",
      "embedding": "dense_vector[1024]",
      "chunk_index": "integer"
    }
  },
  "categories": "keyword",
  "publish_date": "date",
  "word_count": "integer",
  "pdf_size": "integer",
  "downloaded_at": "date"
}
```

#### Paper Processor (`ingest_papers.py`)
- **Purpose**: Orchestrate the complete ingestion pipeline
- **Workflow**:
  1. Load JSON metadata (priority source)
  2. Extract/parse markdown content
  3. Combine and validate metadata
  4. Chunk documents intelligently
  5. Generate embeddings for all components
  6. Index to Elasticsearch

**Metadata Priority System**:
1. **JSON metadata** (primary) - from `/data/pdfs/*.json`
2. **Markdown parsing** (fallback) - extract from content
3. **Default values** - sensible defaults for missing fields

### 2. Backend Services (`backend/`)

#### Search Service (`services/search_service.py`)
- **Purpose**: Provide unified search interface with multiple modes
- **Search Modes**:
  - **Hybrid** (default): 40% BM25 + 60% semantic
  - **Semantic**: Pure embedding-based similarity
  - **BM25**: Traditional full-text search
  - **Title Only**: Search only in paper titles

**Search Flow**:
```python
1. Query preprocessing and embedding generation
2. Multi-field search execution:
   - Title search (3x boost)
   - Abstract search (2x boost)
   - Content search (1x weight)
3. Score combination and ranking
4. Result formatting and metadata enrichment
```

### 3. User Interface (`streamlit_app/`)

#### Streamlit Application
- **Purpose**: Interactive web interface for search and exploration
- **Features**:
  - Real-time search with multiple modes
  - Advanced filtering (categories, date ranges)
  - Paper details and similarity suggestions
  - Search statistics and analytics

### 4. Infrastructure

#### Docker Services (`docker-compose.yml`)
- **Elasticsearch**: Document storage and search engine
  - Port: 9202 (to avoid conflicts)
  - Memory: 2GB heap
  - Single-node cluster
- **MinIO**: Object storage for PDFs and images
  - Ports: 9002 (API), 9003 (Console)
  - Buckets: papers, markdown, images
- **Backend**: FastAPI application server
- **UI**: Streamlit web interface

## Data Flow

### Ingestion Pipeline
```
Markdown Files + JSON Metadata
        │
        ▼
┌─────────────────┐
│ Paper Processor │
│                 │
│ 1. Load JSON    │
│ 2. Parse MD     │
│ 3. Combine Meta │
│ 4. Validate     │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Document        │
│ Chunker         │
│                 │
│ - 512 tokens    │
│ - 100 overlap   │
│ - Sections      │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ BGE Embedder    │
│                 │
│ - Title         │
│ - Abstract      │
│ - Chunks        │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Elasticsearch   │
│ Indexer         │
│                 │
│ - Store docs    │
│ - Store vectors │
│ - Create index  │
└─────────────────┘
```

### Search Pipeline
```
User Query
    │
    ▼
┌─────────────────┐
│ Search Service  │
│                 │
│ 1. Process      │
│ 2. Embed        │
│ 3. Search       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Hybrid Search   │
│                 │
│ BM25 + Semantic │
│ Score Fusion    │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Result Ranking  │
│ & Formatting    │
│                 │
│ - Scores        │
│ - Metadata      │
│ - Suggestions   │
└─────────────────┘
```

## Key Design Decisions

### 1. Hybrid Search Strategy
- **Rationale**: Combines precision of BM25 with semantic understanding
- **Weights**: 40% BM25, 60% semantic (optimized through testing)
- **Benefits**: Better relevance for both exact matches and concept similarity

### 2. Multi-Level Embeddings
- **Title**: Optimized for paper discovery
- **Abstract**: Balanced overview representation
- **Chunks**: Granular content search
- **Benefits**: Multiple granularities of semantic search

### 3. JSON Metadata Priority
- **Rationale**: Rich, structured metadata from external sources
- **Fallback**: Markdown parsing for missing JSON files
- **Benefits**: Consistent, high-quality metadata with flexibility

### 4. Section-Aware Chunking
- **Rationale**: Preserve document structure and context
- **Strategy**: Respect markdown headers and logical boundaries
- **Benefits**: Better chunk coherence and relevance

### 5. Containerized Architecture
- **Rationale**: Consistent deployment and service isolation
- **Benefits**: Easy scaling, development environment consistency

## Performance Characteristics

### Indexing Performance
- **Rate**: ~10-15 papers per minute (depends on document size)
- **Bottleneck**: BGE embedding generation
- **Optimization**: Batch processing, GPU acceleration

### Search Performance
- **Latency**: <500ms for hybrid search
- **Throughput**: 10+ queries per second
- **Factors**: Index size, query complexity, embedding computation

### Storage Requirements
- **Documents**: ~2MB per 1000 papers (text + metadata)
- **Embeddings**: ~4MB per 1000 papers (vectors)
- **Total**: ~6MB per 1000 papers in Elasticsearch

## Scalability Considerations

### Horizontal Scaling
- **Elasticsearch**: Multi-node cluster support
- **Backend**: Multiple API instances with load balancing
- **BGE**: Model replication across instances

### Vertical Scaling
- **Memory**: Elasticsearch heap sizing for large indices
- **GPU**: BGE embedding acceleration
- **Storage**: SSD for optimal search performance

## Security & Privacy

### Data Protection
- No sensitive information in embeddings
- Local deployment option
- Configurable access controls

### API Security
- Rate limiting
- Input validation
- CORS configuration

## Monitoring & Observability

### Metrics
- Search latency and throughput
- Index size and health
- Embedding generation performance
- User query patterns

### Logging
- Structured logging for all components
- Error tracking and alerting
- Performance monitoring

## Future Enhancements

### Short Term
1. MinIO integration for PDF storage
2. Advanced result ranking algorithms
3. Query expansion and suggestion
4. Caching layer for frequent queries

### Long Term
1. Multi-language support
2. Real-time indexing pipeline
3. Machine learning-based relevance tuning
4. Advanced analytics and insights
5. Citation network analysis
6. Collaborative filtering recommendations

## Technical Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **BGE (BAAI/bge-large-en-v1.5)**: Embedding model
- **Elasticsearch 8.11**: Search engine
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend framework
- **Docker**: Containerization

### Libraries
- **sentence-transformers**: BGE model interface
- **tiktoken**: Tokenization
- **elasticsearch-py**: ES client
- **pydantic**: Data validation
- **numpy**: Numerical operations
- **pandas**: Data manipulation

### Infrastructure
- **Docker Compose**: Local orchestration
- **MinIO**: Object storage
- **Nginx**: Reverse proxy (production)