# QA Agent System

The QA Agent system provides intelligent question answering capabilities for the Paper Search Engine, supporting both single-paper and multi-paper question answering modes.

## Features

### Single-Paper QA
- Answer questions about a specific research paper
- Retrieve relevant context chunks from the paper
- Include image analysis when available
- Provide source citations and confidence scores

### Multi-Paper QA
- Answer questions across multiple papers
- Compare and contrast findings across papers
- Synthesize information from multiple sources
- Identify consensus, disagreements, or gaps in research

### Search Results QA
- Answer questions using papers from search results
- Automatically retrieve relevant papers based on search query
- Provide comprehensive answers from multiple sources

## Architecture

### Core Components

1. **QAAgent** (`agent.py`)
   - Main agent class for handling QA requests
   - Supports both single-paper and multi-paper modes
   - Integrates with LLM services (OpenAI, Google Gemini)
   - Provides health checks and monitoring

2. **QARetrievalTool** (`tools.py`)
   - Retrieves relevant context chunks from Elasticsearch
   - Supports filtering by paper ID or search results
   - Handles chunk ranking and relevance scoring

3. **ImageAnalysisTool** (`tools.py`)
   - Analyzes images from papers
   - Uploads images to MinIO and replaces base64 with URLs
   - Provides image descriptions for context

4. **ContextBuilder** (`tools.py`)
   - Builds formatted context for LLM prompts
   - Handles both single-paper and multi-paper contexts
   - Includes image information when available

5. **Configuration** (`config.py`)
   - Centralized configuration for QA system
   - LLM provider settings (OpenAI, Google Gemini)
   - Retrieval and context parameters
   - System prompts for different QA modes

## API Endpoints

### QA Endpoints (`/api/v1/qa/`)

- `POST /single-paper` - Answer questions about a specific paper
- `POST /multi-paper` - Answer questions across multiple papers
- `POST /search-results` - Answer questions using search results
- `GET /health` - Health check for QA services
- `GET /papers/{paper_id}/context` - Get context chunks for a paper

### Health Check Endpoints (`/api/v1/health/`)

- `GET /comprehensive` - Comprehensive health check for all services
- `GET /elasticsearch` - Elasticsearch health check
- `GET /minio` - MinIO health check
- `GET /qa-agent` - QA Agent health check

### Ingestion Endpoints (`/api/v1/ingestion/`)

- `GET /status` - Get ingestion pipeline status
- `GET /health` - Health check for ingestion services
- `POST /process-arxiv` - Process ArXiv papers through pipeline
- `POST /process-arxiv-async` - Start async ArXiv processing
- `GET /pipeline-steps` - Get pipeline step information
- `GET /configuration` - Get ingestion configuration

## Usage Examples

### Single-Paper QA

```python
import requests

# Ask a question about a specific paper
response = requests.post("http://localhost:8000/api/v1/qa/single-paper", json={
    "paper_id": "2301.00234",
    "question": "What is the main contribution of this paper?",
    "max_chunks": 10
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Sources: {len(result['sources'])} chunks used")
```

### Multi-Paper QA

```python
# Ask a question across multiple papers
response = requests.post("http://localhost:8000/api/v1/qa/multi-paper", json={
    "paper_ids": ["2301.00234", "2301.00345", "2301.00456"],
    "question": "How do these papers compare in their approach to transformer architectures?",
    "max_chunks_per_paper": 3
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Papers involved: {result['papers_involved']}")
```

### Search Results QA

```python
# Ask a question using search results
response = requests.post("http://localhost:8000/api/v1/qa/search-results", json={
    "search_query": "transformer attention mechanism",
    "question": "What are the latest developments in attention mechanisms?",
    "max_papers": 5,
    "max_chunks_per_paper": 2
})

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Processing time: {result['processing_time']}s")
```

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
DEFAULT_LLM_PROVIDER=openai  # or "google"

# Model Settings
OPENAI_MODEL=gpt-4o
GOOGLE_MODEL=gemini-1.5-pro

# QA Agent Settings
QA_MAX_TOKENS=4000
QA_TEMPERATURE=0.1
QA_MAX_CONTEXT_CHUNKS=10
QA_INCLUDE_IMAGES=true

# Service Endpoints
ES_HOST=http://103.3.247.120:9202
MINIO_ENDPOINT=http://103.3.247.120:9002
```

### Configuration Class

The `QAConfig` class in `config.py` provides centralized configuration management:

```python
from backend.qa_agent.config import qa_config

# Access configuration
print(f"LLM Provider: {qa_config.default_llm_provider}")
print(f"Max Context Chunks: {qa_config.max_context_chunks}")
print(f"Include Images: {qa_config.include_images}")
```

## Health Monitoring

### Comprehensive Health Check

```python
import requests

# Get comprehensive health status
response = requests.get("http://localhost:8000/api/v1/health/comprehensive")
health = response.json()

print(f"Overall Status: {health['status']}")
print(f"Services: {list(health['services'].keys())}")

for service, status in health['services'].items():
    print(f"{service}: {status['status']}")
```

### Individual Service Health

```python
# Check specific services
es_health = requests.get("http://localhost:8000/api/v1/health/elasticsearch").json()
minio_health = requests.get("http://localhost:8000/api/v1/health/minio").json()
qa_health = requests.get("http://localhost:8000/api/v1/health/qa-agent").json()
```

## Ingestion Pipeline

### Processing ArXiv Papers

```python
# Process papers through the complete pipeline
response = requests.post("http://localhost:8000/api/v1/ingestion/process-arxiv", json={
    "num_papers": 100,
    "use_keywords": True,
    "min_keyword_matches": 1
})

result = response.json()
print(f"Status: {result['status']}")
print(f"Pipeline Steps: {result['pipeline_steps']}")
```

### Pipeline Steps

1. **ArXiv Metadata Download** - Download and filter ArXiv metadata
2. **PDF Download** - Download PDF files from ArXiv
3. **PDF Parsing** - Parse PDFs with Docling to extract text and images
4. **Image Processing** - Upload images to MinIO and replace base64 with URLs
5. **Document Chunking** - Split documents into overlapping chunks
6. **Embedding Generation** - Generate BGE embeddings for text chunks
7. **Elasticsearch Indexing** - Index all documents and embeddings

## Error Handling

The QA system includes comprehensive error handling:

- **Service Unavailable**: Returns 503 status when services are down
- **Invalid Requests**: Returns 400 status for malformed requests
- **Processing Errors**: Returns 500 status for internal errors
- **Timeout Handling**: Configurable timeouts for all operations
- **Retry Logic**: Automatic retries for transient failures

## Performance Considerations

### Optimization Features

- **Chunk-based Retrieval**: Efficient retrieval of relevant context
- **Hybrid Search**: Combines BM25 and semantic search for better results
- **Batch Processing**: Processes multiple papers efficiently
- **Caching**: Caches embeddings and model responses
- **Async Operations**: Non-blocking API calls

### Monitoring

- **Processing Time**: Tracks time for each operation
- **Confidence Scores**: Provides relevance scores for answers
- **Source Tracking**: Tracks which chunks were used for answers
- **Health Metrics**: Monitors service health and performance

## Integration with Frontend

The QA system is designed to integrate seamlessly with the frontend:

- **Paper Selection**: Supports single-paper and multi-paper selection
- **Search Integration**: Works with existing search results
- **Citation Support**: Provides proper citations and sources
- **Image Support**: Handles images from papers
- **Real-time Updates**: Supports real-time health monitoring

## Development

### Adding New QA Modes

1. Extend the `QAAgent` class with new methods
2. Add corresponding API endpoints in `qa.py`
3. Update the frontend to support new modes
4. Add tests for new functionality

### Customizing Prompts

Modify the system prompts in `config.py`:

```python
# Custom single-paper prompt
SINGLE_PAPER_QA_PROMPT = """
Your custom prompt here...
{context_chunks}
{question}
"""
```

### Adding New Retrieval Strategies

1. Extend `QARetrievalTool` with new methods
2. Implement custom ranking algorithms
3. Add configuration options
4. Update health checks

## Troubleshooting

### Common Issues

1. **LLM Service Unavailable**
   - Check API keys and endpoints
   - Verify network connectivity
   - Check service health endpoints

2. **Elasticsearch Connection Issues**
   - Verify ES_HOST configuration
   - Check cluster health
   - Ensure index exists

3. **MinIO Upload Failures**
   - Check MinIO endpoint and credentials
   - Verify bucket permissions
   - Check network connectivity

4. **Low Confidence Scores**
   - Increase max_context_chunks
   - Adjust search parameters
   - Check paper content quality

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check Failures

Use the comprehensive health check to identify issues:

```python
health = requests.get("http://localhost:8000/api/v1/health/comprehensive").json()
# Check individual service statuses
```
