# ASTA Agent Search and Agent Chat Implementation Plan

Based on analysis of the AllenAI repositories and existing documentation, this comprehensive plan outlines implementing ASTA agent search and agent chat capabilities.

## Executive Summary

The plan integrates the ASTA paper-finding agent system with a conversational RAG interface, combining the modular agent architecture from `agent-baselines` with the specialized paper search capabilities from `asta-paper-finder`, while building on the existing system design.

## 1. Architecture Overview

### Core Components Integration
```
┌─────────────────────────────────────────┐
│         Next.js Frontend                │
│    (Chat UI + Paper Discovery)          │
└────────────────┬────────────────────────┘
                 │
┌─────────────────────────────────────────┐
│         FastAPI Gateway                 │
│     (Request Routing + Auth)            │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┴────────────┐
    ▼                          ▼
┌─────────────┐         ┌─────────────────┐
│ASTA Agent   │         │ RAG Chat        │
│Search System│         │ Agent           │
└─────────────┘         └─────────────────┘
```

## 2. Agent System Implementation

### 2.1 ASTA Search Agents (from asta-paper-finder)
Implement specialized agents following the Operative pattern:

- **PaperFinderAgent**: Main orchestrator
- **QueryAnalyzerAgent**: Intent classification and query decomposition
- **BroadSearchAgent**: Content-based semantic search
- **FastBroadSearchAgent**: Quick 30-second searches
- **SpecificPaperAgent**: Exact title/name matching
- **AuthorSearchAgent**: Author-centric searches
- **MetadataAgent**: Venue/year filtering
- **HybridAgent**: Combined content + metadata

### 2.2 RAG Chat Agent (new)
- **ConversationAgent**: Multi-turn dialogue management
- **ContextRetrieverAgent**: Retrieves relevant paper chunks
- **AnswerGeneratorAgent**: LLM-powered response generation
- **CitationAgent**: Manages source attribution and quotes

## 3. Technical Implementation

### 3.1 Agent Framework (Based on agent-baselines)
```python
# Core agent structure following InspectAI pattern
class ASTAAgent(Operative[InputType, OutputType, StateType]):
    def register(self):
        # Initialize sub-agents and dependencies
        self.query_analyzer = self.init_operative("query_analyzer", QueryAnalyzerAgent)
        self.search_agents = {
            "broad": self.init_operative("broad_search", BroadSearchAgent),
            "fast": self.init_operative("fast_search", FastBroadSearchAgent),
            "specific": self.init_operative("specific_paper", SpecificPaperAgent),
            "author": self.init_operative("author_search", AuthorSearchAgent)
        }
        self.chat_agent = self.init_operative("chat", ConversationAgent)
    
    async def handle_operation(self, state, inputs):
        # Route to appropriate agent based on query analysis
        analysis = await self.query_analyzer.execute(inputs.query)
        
        if inputs.mode == "search":
            return await self.route_search_request(state, inputs, analysis)
        elif inputs.mode == "chat":
            return await self.route_chat_request(state, inputs, analysis)
```

### 3.2 Data Integration Layer
Integrate existing storage with ASTA's document collection system:

```python
# Elasticsearch + ASTA Document Collection Bridge
from mabool.utils.dc import DC

class ElasticsearchDCAdapter:
    def __init__(self, es_client):
        self.es = es_client
    
    async def search_to_dc(self, query, filters=None):
        # Convert ES results to ASTA DC format
        results = await self.es.search(...)
        dc = DC.empty()
        
        for hit in results['hits']['hits']:
            dc = dc.add_document({
                'id': hit['_source']['paper_id'],
                'title': hit['_source']['title'],
                'abstract': hit['_source']['abstract'],
                'authors': hit['_source']['authors'],
                'relevance_score': hit['_score']
            })
        
        return await dc.with_fields(["title", "abstract", "authors", "year"])
```

### 3.3 API Integration
Extend existing API structure:

```python
# New endpoints
@app.post("/api/v2/agent/search")
async def agent_search(request: AgentSearchRequest):
    """ASTA-powered paper search with multi-agent coordination"""
    
@app.post("/api/v2/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """Multi-turn RAG conversation with paper corpus"""

@app.post("/api/v2/agent/hybrid")  
async def hybrid_agent(request: HybridRequest):
    """Combined search + chat in single interface"""
```

## 4. Implementation Phases

### Phase 1: Core Agent Infrastructure (Days 1-4)
- [ ] Set up ASTA Operative framework
- [ ] Implement QueryAnalyzerAgent with existing data
- [ ] Create ElasticsearchDCAdapter
- [ ] Build basic PaperFinderAgent orchestrator
- [ ] Test agent communication patterns

### Phase 2: Search Agent Implementation (Days 5-8)
- [ ] Implement BroadSearchAgent with hybrid retrieval
- [ ] Add FastBroadSearchAgent for quick searches
- [ ] Create SpecificPaperAgent for exact matches
- [ ] Build AuthorSearchAgent with author filtering
- [ ] Integrate with existing academic aggregators

### Phase 3: Chat Agent System (Days 9-12)
- [ ] Implement ConversationAgent for multi-turn dialogue
- [ ] Create ContextRetrieverAgent using existing RAG pipeline
- [ ] Build AnswerGeneratorAgent with citation support
- [ ] Add conversation memory and context management
- [ ] Integrate streaming responses

### Phase 4: Integration & Testing (Days 13-14)
- [ ] Connect agent system to Next.js frontend
- [ ] Implement agent selection logic
- [ ] Add error handling and fallback mechanisms
- [ ] End-to-end testing with both modes
- [ ] Performance optimization and deployment

## 5. Key Features

### 5.1 ASTA Agent Search
- **Multi-modal Search**: Content, metadata, author, citation-based
- **Intelligent Routing**: Query analysis determines optimal agent
- **Iterative Refinement**: Agents can refine searches based on results
- **Relevance Scoring**: ASTA's sophisticated relevance judgments

### 5.2 Agent Chat
- **Context-Aware**: Maintains conversation history and paper context
- **Grounded Responses**: All answers cite specific paper sources
- **Multi-turn Dialogue**: Handles follow-up questions and clarifications
- **Paper-Specific Chat**: Can focus on specific papers or collections

### 5.3 Hybrid Interface
- **Seamless Switching**: Users can switch between search and chat modes
- **Shared Context**: Search results inform chat context
- **Progressive Disclosure**: Chat can surface additional relevant papers

## 6. Technical Specifications

### 6.1 Models and APIs
- **LLM**: OpenAI GPT-4o-mini (cost-efficient, fast)
- **Embeddings**: OpenAI text-embedding-3-large
- **Reranking**: Cohere rerank-english-v2.0
- **Analysis**: Google Gemini Flash for query analysis

### 6.2 Performance Targets
- **Fast Mode**: <30 seconds for search results
- **Diligent Mode**: <2 minutes for comprehensive search
- **Chat Response**: <3 seconds for RAG answers
- **Streaming**: Real-time response streaming

### 6.3 Data Requirements
- **Paper Corpus**: 1000+ papers with full-text
- **Chunk Size**: 512-1024 tokens with 15-20% overlap
- **Context Window**: 6-12 chunks per RAG response
- **Citation Granularity**: Section-level with page spans

## 7. Configuration Strategy

### 7.1 Agent Configuration
```toml
[asta_agents]
default_mode = "infer"  # auto-select best agent
max_iterations = 3
timeout_seconds = 120

[search_agents.broad]
relevance_quota = 250
highly_relevant_cap = 50
llm_suggestions = 5

[search_agents.fast]  
relevance_quota = 50
max_iterations = 1
timeout_seconds = 30

[chat_agent]
max_context_chunks = 12
conversation_memory_turns = 10
streaming_enabled = true
```

### 7.2 Integration Points
- **Elasticsearch**: Existing index structure maintained
- **MinIO**: PDF and parsed content storage
- **Postgres**: Paper metadata and conversation history
- **Redis**: Agent state caching and session management

## 8. Success Metrics

### 8.1 Search Quality
- **Precision@10**: >0.8 for relevant paper retrieval
- **User Satisfaction**: >4.0/5.0 rating for search results
- **Coverage**: Handle 95%+ of academic search intents

### 8.2 Chat Quality  
- **Grounding Rate**: >90% of answers properly cited
- **Response Quality**: Human evaluation >4.0/5.0
- **Context Coherence**: Multi-turn conversation flow

### 8.3 Performance
- **Availability**: 99.5% uptime
- **Response Times**: Meet targets above
- **Cost Efficiency**: <$0.10 per search/chat session

## 9. Risk Mitigation

### 9.1 Technical Risks
- **Agent Complexity**: Start with simple agents, add sophistication gradually
- **Performance**: Implement aggressive caching and async processing
- **Cost**: Use smaller models for non-critical operations

### 9.2 Data Quality
- **PDF Parsing**: GROBID + fallback parsers
- **Citation Accuracy**: Validate citations against source documents
- **Relevance Tuning**: A/B testing for ranking improvements

## 10. Future Enhancements

### 10.1 Advanced Agents
- **SnowballAgent**: Citation-based paper discovery
- **TrendAnalysisAgent**: Research trend identification  
- **CollaborationAgent**: Author network analysis
- **SummaryAgent**: Multi-paper synthesis

### 10.2 Enhanced Chat
- **Visual Aids**: Chart/graph generation from paper data
- **Multi-modal**: Handle paper figures and equations
- **Collaborative**: Multi-user research sessions
- **Export**: Research note and bibliography generation

## Implementation Resources

### Required Dependencies
```python
# Core agent framework
inspect-ai>=0.3.0
ai2i>=0.2.0

# LLM and embedding APIs
openai>=1.0.0
cohere>=4.0.0
google-generativeai>=0.3.0

# Data processing
elasticsearch>=8.0.0
psycopg2-binary>=2.9.0
minio>=7.0.0

# Web framework
fastapi>=0.100.0
uvicorn>=0.20.0
```

### Key Implementation Files
```
agents/
├── asta_agent_system/
│   ├── core/
│   │   ├── operative.py          # Base agent class
│   │   ├── orchestrator.py       # Main agent coordinator
│   │   └── document_collection.py # DC adapter
│   ├── search_agents/
│   │   ├── paper_finder.py       # Main search orchestrator
│   │   ├── query_analyzer.py     # Intent classification
│   │   ├── broad_search.py       # Content-based search
│   │   ├── fast_search.py        # Quick search
│   │   ├── specific_paper.py     # Exact matching
│   │   └── author_search.py      # Author-focused search
│   ├── chat_agents/
│   │   ├── conversation.py       # Multi-turn dialogue
│   │   ├── context_retriever.py  # RAG context building
│   │   ├── answer_generator.py   # LLM response generation
│   │   └── citation_manager.py   # Source attribution
│   └── api/
│       ├── routes.py             # Agent API endpoints
│       ├── models.py             # Request/response schemas
│       └── middleware.py         # Auth and rate limiting
```

This implementation plan provides a roadmap for building a sophisticated academic research assistant that combines the power of ASTA's multi-agent paper search with conversational AI capabilities.