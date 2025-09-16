# Evidence-Based Paper Search Implementation Plan

## Overview
Implement a sophisticated paper search flow that:
1. **Search** → Get top 20 papers with highest scores
2. **Read** → Retrieve full paper text content
3. **Analyze** → Determine relevance with AI analysis
4. **Extract** → Return relevant papers with supporting evidence chunks
5. **Iterate** → Search more papers if insufficient relevant results found

## Current System Analysis

### ✅ Existing Infrastructure
- **Elasticsearch Service**: `backend/services/es_search_service.py`
  - `search()` method returns top papers by score
  - `get_paper_details()` retrieves full paper content
  - Papers already chunked and indexed with full text
- **LlamaIndex Agent**: `backend/llama_agent/agent.py`
  - ReAct agent with search and detail retrieval tools
  - Gemini Flash 2.5 LLM integration
- **BGE Embeddings**: Semantic similarity already available
- **Paper Data Models**: `SearchResult` and `PaperDetails` with full content

## Implementation Plan

### Phase 1: Evidence Analysis Module
**File**: `backend/llama_agent/evidence_analyzer.py`

#### Components:
1. **Data Models (Pydantic)**
   ```python
   class EvidenceChunk(BaseModel):
       text: str                    # The supporting evidence text
       start_position: int          # Character position in paper
       confidence_score: float      # How confident this supports the query (0-1)
       relevance_reason: str        # Why this chunk is relevant

   class PaperAnalysis(BaseModel):
       paper_id: str
       title: str
       is_relevant: bool           # Does this paper fit the query?
       relevance_score: float      # Overall relevance (0-1)
       evidence_chunks: List[EvidenceChunk]  # Supporting evidence
       analysis_summary: str       # Brief explanation of relevance
   ```

2. **EvidenceAnalyzer Class**
   ```python
   class EvidenceAnalyzer:
       def __init__(self, llm: Gemini)

       async def analyze_paper_relevance(
           self,
           paper: PaperDetails,
           query: str
       ) -> PaperAnalysis

       async def extract_evidence_chunks(
           self,
           paper_content: str,
           query: str,
           relevance_threshold: float = 0.7
       ) -> List[EvidenceChunk]

       async def batch_analyze_papers(
           self,
           papers: List[PaperDetails],
           query: str
       ) -> List[PaperAnalysis]
   ```

3. **Analysis Strategy**
   - **Chunk-based Analysis**: Split paper into 1000-char chunks with 200-char overlap
   - **Relevance Scoring**: Use LLM to score each chunk's relevance to query
   - **Evidence Extraction**: Extract top 3 most relevant chunks as evidence
   - **Overall Assessment**: Combine chunk scores for paper-level relevance

#### Prompts:
```python
CHUNK_RELEVANCE_PROMPT = """
Analyze if this text chunk is relevant to the query: "{query}"

Chunk: {chunk_text}

Rate relevance (0.0-1.0) and explain why:
- 0.0: Not relevant at all
- 0.5: Somewhat related
- 1.0: Directly addresses the query

Return JSON: {{"relevance_score": 0.8, "reason": "explanation"}}
"""

PAPER_RELEVANCE_PROMPT = """
Based on these chunk analyses, determine if this paper is relevant to: "{query}"

Paper: {title}
Chunk scores: {chunk_scores}

Consider:
1. Does the paper directly address the query topic?
2. Are there sufficient relevant sections?
3. Is this a primary focus or just mentioned?

Return JSON: {{"is_relevant": true, "overall_score": 0.85, "summary": "explanation"}}
"""
```

### Phase 2: Enhanced Search Agent
**File**: `backend/llama_agent/evidence_search_agent.py`

#### EvidenceSearchAgent Class:
```python
class EvidenceSearchAgent(PaperSearchAgent):
    def __init__(self, es_service, llm):
        super().__init__(es_service, llm)
        self.evidence_analyzer = EvidenceAnalyzer(llm)

    async def evidence_based_search(
        self,
        query: str,
        min_relevant_papers: int = 5,
        max_iterations: int = 3
    ) -> EvidenceSearchResponse
```

#### Search Flow Logic:
```python
async def evidence_based_search(self, query: str):
    relevant_papers = []
    search_iteration = 1
    total_analyzed = 0

    while len(relevant_papers) < min_relevant_papers and search_iteration <= max_iterations:
        # 1. Search for top 20 papers
        search_results = await self.search_papers(query, max_results=20)

        # 2. Get full content for each paper
        papers_with_content = []
        for result in search_results:
            details = await self.get_paper_details(result.paper_id)
            if details:
                papers_with_content.append(details)

        # 3. Batch analyze relevance
        analyses = await self.evidence_analyzer.batch_analyze_papers(
            papers_with_content, query
        )

        # 4. Filter relevant papers
        for analysis in analyses:
            if analysis.is_relevant and analysis.relevance_score >= 0.7:
                relevant_papers.append(analysis)

        total_analyzed += len(papers_with_content)
        search_iteration += 1

        # 5. If insufficient results, modify search strategy
        if len(relevant_papers) < min_relevant_papers:
            query = self.refine_query_for_next_iteration(query)

    return EvidenceSearchResponse(
        query=original_query,
        relevant_papers=relevant_papers[:10],  # Top 10 most relevant
        search_iterations=search_iteration - 1,
        total_analyzed=total_analyzed
    )
```

### Phase 3: Enhanced Tools & API

#### New Tools (`backend/llama_agent/tools.py`):
```python
def create_evidence_tools(es_service, evidence_analyzer):
    return [
        FunctionTool.from_defaults(
            fn=evidence_analyzer.analyze_paper_relevance,
            name="analyze_paper_relevance",
            description="Analyze if a paper is relevant to query with evidence extraction"
        ),
        FunctionTool.from_defaults(
            fn=evidence_analyzer.batch_analyze_papers,
            name="batch_analyze_papers",
            description="Analyze multiple papers for relevance in batch"
        )
    ]
```

#### New API Endpoint (`backend/llama_agent/api.py`):
```python
@llama_router.post("/evidence-search", response_model=EvidenceSearchResponse)
async def evidence_based_search(request: EvidenceSearchRequest):
    """
    Intelligent evidence-based paper search:
    1. Search top papers by score
    2. Analyze full content for relevance
    3. Extract supporting evidence chunks
    4. Return only truly relevant papers with evidence
    """
    agent = get_evidence_search_agent()
    return await agent.evidence_based_search(
        query=request.query,
        min_relevant_papers=request.min_relevant_papers,
        max_iterations=request.max_iterations
    )
```

### Phase 4: Response Models

#### Request/Response Models:
```python
class EvidenceSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    min_relevant_papers: int = Field(5, description="Minimum relevant papers to find")
    max_iterations: int = Field(3, description="Maximum search iterations")
    relevance_threshold: float = Field(0.7, description="Minimum relevance score")

class EvidenceSearchResponse(BaseModel):
    query: str
    relevant_papers: List[PaperAnalysis]
    search_iterations: int
    total_analyzed: int
    execution_time: float
    summary: str
```

#### Example Response:
```json
{
  "query": "retrieval augmented generation for question answering",
  "relevant_papers": [
    {
      "paper_id": "2005.11401",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "is_relevant": true,
      "relevance_score": 0.95,
      "evidence_chunks": [
        {
          "text": "RAG combines parametric and non-parametric memory for generation. We show that RAG models generate more specific, diverse and factual language than state-of-the-art seq2seq models for question answering.",
          "start_position": 1250,
          "confidence_score": 0.98,
          "relevance_reason": "Directly describes RAG for question answering with specific claims about performance"
        },
        {
          "text": "For knowledge-intensive tasks like question answering, RAG achieves state-of-the-art results on Natural Questions, WebQuestions, and CuratedTrec datasets.",
          "start_position": 3420,
          "confidence_score": 0.92,
          "relevance_reason": "Shows empirical results of RAG on QA benchmarks"
        }
      ],
      "analysis_summary": "This paper introduces the RAG architecture specifically for knowledge-intensive NLP tasks including question answering, with strong empirical results."
    }
  ],
  "search_iterations": 2,
  "total_analyzed": 35,
  "execution_time": 12.4,
  "summary": "Found 1 highly relevant paper with strong evidence for RAG in question answering applications."
}
```

## Configuration Updates

### Enhanced Config (`backend/llama_agent/config.py`):
```python
class EvidenceAnalysisConfig(BaseModel):
    evidence_chunk_size: int = Field(default=1000)
    evidence_overlap: int = Field(default=200)
    min_relevance_threshold: float = Field(default=0.7)
    max_evidence_chunks: int = Field(default=3)
    batch_size: int = Field(default=5)
    max_search_iterations: int = Field(default=3)
    min_relevant_papers: int = Field(default=5)
```

## Implementation Timeline

### Step 1: Evidence Analyzer (2 hours)
- [ ] Create `evidence_analyzer.py` with data models
- [ ] Implement chunk-based relevance analysis
- [ ] Add evidence extraction logic
- [ ] Create batch processing methods

### Step 2: Enhanced Search Agent (2 hours)
- [ ] Create `evidence_search_agent.py`
- [ ] Implement adaptive search flow
- [ ] Add iteration logic for insufficient results
- [ ] Integrate evidence analysis

### Step 3: API Integration (1 hour)
- [ ] Add new tools to agent
- [ ] Create evidence search endpoint
- [ ] Update request/response models
- [ ] Add proper error handling

### Step 4: Testing & Optimization (1 hour)
- [ ] Test with various query types
- [ ] Tune relevance thresholds
- [ ] Optimize batch processing performance
- [ ] Add comprehensive logging

## Expected Benefits

### ✅ Higher Precision
- Only returns papers that are truly relevant to the query
- Evidence-based ranking instead of just search scores
- Reduces noise from tangentially related papers

### ✅ Explainable Results
- Each paper includes supporting evidence chunks
- Clear relevance scores and explanations
- Users can immediately see why papers were selected

### ✅ Adaptive Search
- Continues searching until sufficient relevant papers found
- Refines search strategy across iterations
- Handles edge cases where initial search is insufficient

### ✅ Rich Context
- Evidence chunks provide immediate value
- Users get key insights without reading full papers
- Supporting quotes ready for citations

### ✅ Scalable Architecture
- Batch processing for efficiency
- Configurable thresholds and parameters
- Extensible for different analysis types

## Technical Considerations

### Performance Optimizations
- **Parallel Processing**: Analyze multiple papers concurrently
- **Caching**: Cache paper content and analyses to avoid re-processing
- **Chunk Reuse**: Leverage existing chunking from Elasticsearch index
- **Batch LLM Calls**: Group multiple analyses into single LLM requests

### Error Handling
- **Graceful Degradation**: Return partial results if some analyses fail
- **Timeout Management**: Set reasonable timeouts for LLM calls
- **Fallback Strategy**: Fall back to regular search if evidence analysis fails
- **Rate Limiting**: Respect LLM API rate limits

### Quality Assurance
- **Threshold Tuning**: Calibrate relevance thresholds on test data
- **Evidence Validation**: Ensure extracted chunks actually support claims
- **Consistency Checks**: Verify analysis consistency across similar papers
- **User Feedback Loop**: Collect feedback to improve relevance scoring