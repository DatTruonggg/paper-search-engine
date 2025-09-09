# PaperFinder Implementation Guide

## Overview

PaperFinder is an AI-powered research paper discovery agent that helps users find academic papers based on content and metadata criteria. This guide will help you understand and reuse this codebase for your own projects.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components](#key-components)
3. [Setup and Installation](#setup-and-installation)
4. [Core Concepts](#core-concepts)
5. [Agent System](#agent-system)
6. [API Structure](#api-structure)
7. [Configuration System](#configuration-system)
8. [How to Extend](#how-to-extend)
9. [Integration Points](#integration-points)
10. [Common Use Cases](#common-use-cases)

## Architecture Overview

PaperFinder is built as a modular, agent-based system with the following architecture:

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│         (Entry Point)                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Router Layer                    │
│   (round_v2_routes.py)                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      PaperFinderAgent                   │
│   (Main Orchestrator)                   │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                          ▼
┌─────────────┐         ┌─────────────────┐
│Query Analyzer│         │ Specialized     │
│             │         │ Agents          │
└─────────────┘         └─────────────────┘
```

## Key Components

### 1. **FastAPI Application** (`mabool/api/app.py`)
- **Purpose**: Main entry point for the web service
- **Key Features**:
  - CORS middleware for cross-origin requests
  - Error handling for various exception types
  - Health check endpoint for Kubernetes deployments
  - Dependency injection system integration

### 2. **API Routes** (`mabool/api/round_v2_routes.py`)
- **Purpose**: Defines the REST API endpoints
- **Main Endpoint**: `/api/2/rounds` (POST)
- **Features**:
  - Caching system for repeated queries
  - Token usage tracking
  - Priority-based request handling with semaphore

### 3. **PaperFinderAgent** (`mabool/agents/paper_finder/paper_finder_agent.py`)
- **Purpose**: Main orchestrator that coordinates all sub-agents
- **Responsibilities**:
  - Query analysis delegation
  - Agent selection based on query type
  - Result aggregation and formatting
  - Error handling and fallback mechanisms

### 4. **Query Analyzer** (`mabool/agents/query_analyzer/query_analyzer.py`)
- **Purpose**: Analyzes and decomposes user queries
- **Functions**:
  - Extract time ranges, authors, venues
  - Identify query intent (specific paper, broad search, etc.)
  - Extract relevance criteria
  - Detect refusals or invalid queries

### 5. **Specialized Agents**

Each agent handles specific types of paper searches:

- **BroadSearchAgent**: General content-based searches
- **FastBroadSearchAgent**: Quick searches (~30 seconds)
- **SpecificPaperByTitleAgent**: Find papers by exact title
- **SpecificPaperByNameAgent**: Find papers by descriptive name
- **SearchByAuthorsAgent**: Author-based searches
- **MetadataOnlyAgent**: Searches based only on metadata
- **SnowballAgent**: Citation-based discovery
- **DenseAgent**: Dense retrieval using embeddings

## Setup and Installation

### Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.12.8 or higher required
python --version
```

### API Keys Required

Create a `.env.secret` file in `agents/mabool/api/conf/`:

```bash
OPENAI_API_KEY=your_openai_key
S2_API_KEY=your_semantic_scholar_key
COHERE_API_KEY=your_cohere_key
GOOGLE_API_KEY=your_google_key
```

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd asta-paper-finder

# 2. Setup development environment
make sync-dev

# 3. Start the development server
cd agents/mabool/api
make start-dev
```

The API will be available at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

## Core Concepts

### 1. **Operative System**
The codebase uses an "Operative" pattern for agents:

```python
class MyAgent(Operative[InputType, OutputType, StateType]):
    def register(self):
        # Initialize sub-operatives
        pass
    
    async def handle_operation(self, state, inputs):
        # Main agent logic
        return new_state, response
```

### 2. **Document Collection**
Central data structure for managing papers:

```python
from mabool.utils.dc import DC

# Create empty collection
docs = DC.empty()

# Add fields to documents
docs = await docs.with_fields(["title", "abstract", "authors"])
```

### 3. **Response Types**

Three types of responses:
- **CompleteResponse**: Successful completion
- **PartialResponse**: Partial success with errors
- **VoidResponse**: Complete failure

### 4. **Query Analysis Results**

```python
QueryAnalysisSuccess     # Query successfully analyzed
QueryAnalysisRefusal     # Query refused (out of scope)
QueryAnalysisPartialSuccess  # Partial analysis with fallback
QueryAnalysisFailure     # Complete analysis failure
```

## Agent System

### Agent Types and Their Roles

1. **Content-Based Agents**
   - Search by paper content/abstracts
   - Use relevance judgments
   - Support iterative refinement

2. **Metadata Agents**
   - Search by authors, venues, dates
   - No content analysis required
   - Fast execution

3. **Hybrid Agents**
   - Combine content and metadata
   - Most complex searches
   - Multiple data sources

### Agent Communication

Agents communicate through:
- **Input Models**: Pydantic models defining required inputs
- **Output Models**: Standardized output format
- **State Management**: Maintain state across operations

Example:
```python
@dataclass
class BroadSearchInput:
    user_input: str
    content_query: str
    relevance_criteria: RelevanceCriteria
    time_range: ExtractedYearlyTimeRange
    venues: list[str]
    authors: list[str]
```

## API Structure

### Main Endpoint

**POST** `/api/2/rounds`

Request Body:
```json
{
  "paper_description": "papers about transformer architectures",
  "operation_mode": "fast",  // "fast" | "diligent" | "infer"
  "inserted_before": "2024-01-01",  // Optional date filter
  "read_results_from_cache": false
}
```

Response:
```json
{
  "doc_collection": {
    "documents": [
      {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "year": 2017,
        "abstract": "...",
        "relevance_score": 0.95
      }
    ]
  },
  "response_text": "Found 15 highly relevant papers...",
  "analyzed_query": { ... },
  "token_breakdown_by_model": { ... }
}
```

## Configuration System

### Configuration Files

1. **`config.toml`**: Main configuration
   - LLM model settings
   - Agent parameters
   - API settings
   - Timeout values

2. **`config.extra.fast_mode.toml`**: Fast mode overrides
   - Reduced iterations
   - Lower quotas
   - Faster models

### Key Configuration Parameters

```toml
[default.relevance_judgement]
relevance_model_name = "google:gemini2flash-default"
quota = 250  # Max papers to judge
highly_relevant_cap = 50  # Max highly relevant papers

[default.broad_search_agent]
max_iterations = 3  # Search refinement iterations
llm_n_suggestions = 5  # LLM-suggested papers

[default.s2_api]
concurrency = 10  # Parallel API calls
total_papers_limit = 1000  # Max papers to fetch
```

## How to Extend

### Adding a New Agent

1. **Create Agent Class**:
```python
# agents/mabool/api/mabool/agents/my_agent/my_agent.py
from mabool.infra.operatives import Operative

class MyCustomAgent(Operative[MyInput, MyOutput, MyState]):
    def register(self):
        # Initialize dependencies
        pass
    
    async def handle_operation(self, state, inputs):
        # Implement agent logic
        results = await self.search_papers(inputs)
        return state, CompleteResponse(data=results)
```

2. **Register in PaperFinderAgent**:
```python
def register(self):
    self.my_agent = self.init_operative("my_agent", MyCustomAgent)
```

3. **Add to Query Router**:
```python
case "MY_QUERY_TYPE":
    response = await plan_context.run_my_custom_search()
```

### Adding New Search Sources

1. **Create Data Access Layer**:
```python
# mabool/dal/my_source.py
class MySourceDAL:
    async def search(self, query: str) -> list[Document]:
        # Implement API calls
        pass
```

2. **Integrate in Agents**:
```python
from mabool.dal.my_source import MySourceDAL

class MyAgent:
    def __init__(self):
        self.source = MySourceDAL()
    
    async def search(self, query):
        results = await self.source.search(query)
        return self.process_results(results)
```

## Integration Points

### 1. **External APIs**
- Semantic Scholar API (`mabool/dal/s2.py`)
- Cohere Reranking (`mabool/external_api/rerank/cohere.py`)
- OpenAI/Google LLMs (via `ai2i.chain`)

### 2. **Caching System**
- File-based cache for development
- Redis/Memcached for production
- Configurable TTL

### 3. **Monitoring**
- Token usage tracking
- Request timing
- Error logging

## Common Use Cases

### 1. **Simple Paper Search**
```python
from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.agents.paper_finder.paper_finder_agent import run_agent

input_data = PaperFinderInput(
    query="recent papers on quantum computing",
    operation_mode="fast"
)

result = await run_agent(input_data, thread_id)
papers = result.data.doc_collection.documents
```

### 2. **Author-Based Search**
```python
input_data = PaperFinderInput(
    query="papers by Yoshua Bengio on deep learning",
    operation_mode="diligent"
)
```

### 3. **Metadata-Only Search**
```python
input_data = PaperFinderInput(
    query="CVPR papers from 2023",
    operation_mode="fast"
)
```

### 4. **Custom Relevance Criteria**
```python
from ai2i.dcollection import RelevanceCriteria

criteria = RelevanceCriteria(
    description="Papers specifically about transformer efficiency",
    strict=True
)
```

## Best Practices

### 1. **Error Handling**
- Always handle `VoidResponse` cases
- Provide fallback mechanisms
- Log errors appropriately

### 2. **Performance**
- Use "fast" mode for initial searches
- Cache frequently accessed data
- Limit concurrent API calls

### 3. **Testing**
```bash
# Run tests
cd agents/mabool/api
pytest tests/

# Run with coverage
pytest --cov=mabool tests/
```

### 4. **Deployment**
- Use environment variables for secrets
- Configure appropriate timeouts
- Monitor API rate limits

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all keys in `.env.secret` are valid
   - Check API rate limits

2. **Timeout Errors**
   - Increase `operative_timeout` in config
   - Use "fast" mode for testing

3. **Memory Issues**
   - Reduce `total_papers_limit`
   - Lower `relevance_judgement.quota`

## Conclusion

PaperFinder provides a robust foundation for building academic paper discovery systems. Its modular architecture allows for easy extension and customization. The agent-based design enables complex search workflows while maintaining code organization and reusability.

For production deployments, consider:
- Implementing proper authentication
- Setting up monitoring and alerting
- Optimizing cache strategies
- Scaling the semaphore system for concurrent requests

This codebase demonstrates best practices in:
- Asynchronous Python programming
- LLM integration patterns
- Multi-agent system design
- API development with FastAPI