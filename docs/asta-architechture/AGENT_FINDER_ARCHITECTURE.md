# PaperFinder Codebase Architecture Analysis

## 1. **High-Level Architecture Overview**
The PaperFinder is a sophisticated multi-agent paper search system built on a pipeline-based architecture. It combines manual-coded control flow with LLM decision-making at key points to deliver intelligent paper discovery.

## 2. **Main Module Breakdown (Big to Small)**

### **Level 1: Core Application (`agents/mabool/api/`)**

#### **FastAPI Application** (`mabool/api/app.py`)
- **Entry Point**: Creates the main FastAPI application with dependency injection
- **Configuration**: Loads config from `conf/` directory
- **Middleware**: CORS, error handling, logging setup
- **Health Endpoints**: `/health` for Kubernetes readiness

#### **API Routes** (`mabool/api/round_v2_routes.py`)  
- **Main Endpoint**: `/api/2/rounds` (POST) - the core paper search API
- **Caching**: File-based caching with `@cached` decorator
- **Concurrency**: Priority-based semaphore limiting concurrent requests to 3
- **Token Tracking**: Monitors LLM usage across all operations

### **Level 2: Orchestration Layer (`mabool/agents/paper_finder/`)**

#### **Paper Finder Agent** (`paper_finder_agent.py`)
This is the **main orchestrator** that coordinates the entire search pipeline:
- **Query Analysis**: Uses `query_analyzer` to decompose natural language queries
- **Agent Selection**: Routes queries to appropriate specialized agents based on intent
- **Multi-Agent Coordination**: Manages parallel execution of different search strategies
- **Result Fusion**: Combines results from multiple agents
- **Final Ranking**: Applies content relevance + metadata criteria (influence, recency)

### **Level 3: Specialized Search Agents (`mabool/agents/`)**

#### **Query Analyzer** (`query_analyzer/`)
- **Core Function**: Transforms natural language into structured search objects  
- **LLM Integration**: Uses prompts to extract search intent, constraints, preferences
- **Output**: Structured query specifications for downstream agents

#### **Search Strategy Agents**:

**Broad Search Agents** (`broad_search_by_keyword/`, `complex_search/`)
- **Purpose**: General keyword-based paper discovery
- **Strategies**: Fast vs. exhaustive search modes
- **APIs**: Semantic Scholar integration for large-scale retrieval

**Dense Search** (`dense/`)
- **Purpose**: Semantic similarity-based search using embeddings
- **Technology**: Vector similarity for finding conceptually related papers

**Snowball Search** (`snowball/`)  
- **Purpose**: Citation-based expansion from seed papers
- **Method**: Explores citing/cited papers to build comprehensive result sets

**Metadata Search** (`metadata_only/`, `search_by_authors/`)
- **Purpose**: Author, venue, date-based filtering and search
- **Use Cases**: When queries specify specific researchers or publication venues

**Specific Paper Agents** (`specific_paper_by_name/`, `specific_paper_by_title/`)
- **Purpose**: Direct paper lookup when users request specific publications
- **Method**: Exact matching and fuzzy search for paper identification

#### **Support Agents**:
- **LLM Suggestion** (`llm_suggestion/`) - AI-generated paper recommendations
- **Query Refusal** (`query_refusal/`) - Handles out-of-scope or unclear queries

### **Level 4: Shared Libraries (`libs/`)**

#### **Document Collection (`libs/dcollection/`)**
- **Core Models**: Paper, Author, Citation, Journal data structures
- **Collection Interface**: Unified API for working with document sets
- **External APIs**: Semantic Scholar integration and data fetching
- **Field Management**: Dynamic loading of paper metadata fields

#### **Chain (`libs/chain/`)**  
- **LLM Pipeline**: Computation management for AI operations
- **Prompt Management**: Standardized prompt templates and execution
- **Model Integration**: OpenAI, Cohere, and other LLM providers
- **Response Processing**: Structured output parsing from LLM responses

#### **Config (`libs/config/`)**
- **Configuration Schema**: Type-safe config validation
- **Environment Management**: Dev/prod environment handling  
- **API Keys**: Secure credential management for external services

#### **Common (`libs/common/`)**
- **Utilities**: Shared helper functions and base classes
- **Data Structures**: Common data types used across modules
- **Validation**: Input validation and sanitization utilities

#### **Dependency Injection (`libs/di/`)**
- **Service Container**: Manages service instances and lifecycles
- **Scope Management**: Request-scoped vs singleton services
- **Testing Support**: Mock injection for unit tests

## 3. **Data Flow Pipeline**

```
1. Query Reception → FastAPI receives natural language query
2. Query Analysis → LLM decomposes query into structured search intent  
3. Agent Selection → Routes to appropriate search strategies based on intent
4. Multi-Agent Execution → Parallel execution of different search approaches
5. Result Fusion → Combines and ranks results from multiple agents
6. Relevance Scoring → LLM-based relevance judgment on abstracts/snippets  
7. Final Ranking → Sorts by relevance + metadata criteria (influence, recency, etc.)
8. Response Assembly → Packages results with explanations and metadata
```

## 4. **Key Integration Points**

- **Semantic Scholar API**: Primary paper database for retrieval
- **External LLMs**: OpenAI/others for query analysis and relevance scoring
- **Cohere API**: For result reranking and semantic search
- **File-Based Caching**: Performance optimization for repeated queries
- **Token Tracking**: Comprehensive LLM usage monitoring

## 5. **Operational Modes**

- **"infer" Mode**: Fast search (~30 seconds) with streamlined pipeline
- **"diligent" Mode**: Exhaustive search (~3 minutes) using all available agents

## 6. **Module Details**

### **Core Data Models (`mabool/data_model/`)**
- **Agent Models**: Operation modes, agent states, response types
- **Configuration**: Type-safe config schemas and validation
- **Request/Response**: API contract definitions
- **Specifications**: Query structure and search constraints
- **User-Facing Strings**: Internationalization and messaging

### **Infrastructure (`mabool/infra/`)**
- **Operatives**: Agent execution framework and response handling
- **Interaction**: Agent communication protocols

### **Services (`mabool/services/`)**  
- **Service Layer**: Business logic and external API orchestration
- **Priority Tasks**: Concurrent request management
- **Dependencies**: Service injection and configuration

### **Utilities (`mabool/utils/`)**
- **Logging**: Structured logging and debugging
- **Caching**: File-based result caching
- **Data Classes**: Utility data structures
- **Path Management**: File system utilities

## 7. **Testing Structure**

Each module includes comprehensive test suites:
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-module interaction testing  
- **Regression Tests**: End-to-end pipeline validation
- **Mock Services**: External API mocking for isolated testing

## 8. **Development Workflow**

### **Code Quality Tools**
- **Linting**: Ruff with custom configuration
- **Type Checking**: Pyright with strict settings
- **Formatting**: 120 character line length standard
- **Testing**: Pytest with coverage reporting

### **Build System**
- **Package Manager**: UV for dependency management
- **Workspace**: Monorepo with agents and shared libraries
- **Python Version**: 3.12+ required
- **Development Server**: Auto-reload on port 8000

This architecture enables PaperFinder to handle diverse query types - from broad exploratory searches to specific paper lookups - while maintaining high relevance and comprehensive coverage of the academic literature.