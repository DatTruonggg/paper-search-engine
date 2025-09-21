"""QA API endpoints for question answering across single, multiple, search-result, and mixed paper contexts.

Refactors:
- Centralized QA agent acquisition
- Helper for uniform API response building
- Reduced duplication across endpoints
- Structured logging fields for easier analysis
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from backend.services import ElasticsearchSearchService
from backend.config import config
from backend.qa_agent import QAAgent
from logs import log


# Create router for QA endpoints
router = APIRouter(prefix="/api/v1/qa", tags=["Question Answering"])

# Global QA agent instance
qa_agent: Optional[QAAgent] = None


# Pydantic models for API
class SinglePaperQARequest(BaseModel):
    """Request model for single-paper QA"""
    paper_id: str = Field(..., description="ID of the paper to ask about", min_length=1)
    question: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    max_chunks: int = Field(default=10, ge=1, le=20, description="Maximum number of context chunks to use")


class MultiPaperQARequest(BaseModel):
    """Request model for multi-paper QA"""
    paper_ids: List[str] = Field(
        ..., 
        description="List of paper IDs to search across", 
        min_items=1, 
        max_items=100
    )
    question: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    max_chunks_per_paper: int = Field(default=3, ge=1, le=10, description="Maximum chunks per paper")


class SearchResultsQARequest(BaseModel):
    """Request model for search results QA"""
    search_query: str = Field(..., description="Original search query that returned papers", min_length=1)
    question: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    max_papers: int = Field(default=5, ge=1, le=10, description="Maximum number of papers to include")
    max_chunks_per_paper: int = Field(default=2, ge=1, le=5, description="Maximum chunks per paper")


class MixedQARequest(BaseModel):
    """Request model combining selected paper IDs and/or a search query."""
    question: str = Field(..., min_length=1, max_length=1000)
    paper_ids: List[str] | None = Field(default=None, description="Explicit selected (bookmarked) paper IDs")
    search_query: str | None = Field(default=None, description="Optional fresh search query to augment context")
    max_chunks_per_selected: int = Field(default=3, ge=1, le=10)
    max_search_papers: int = Field(default=5, ge=1, le=15)
    max_search_chunks_per_paper: int = Field(default=2, ge=1, le=5)

    def model_post_init(self, *args, **kwargs):  # type: ignore[override]
        if (not self.paper_ids or len(self.paper_ids) == 0) and not self.search_query:
            raise ValueError("Provide at least paper_ids or search_query")


class QAResponse(BaseModel):
    """Response model for QA requests"""
    answer: str = Field(description="Generated answer")
    sources: List[Dict[str, Any]] = Field(description="List of source chunks used")
    confidence_score: float = Field(description="Confidence score (0.0-1.0)")
    processing_time: float = Field(description="Processing time in seconds")
    context_chunks_count: int = Field(description="Number of context chunks used")
    papers_involved: List[str] = Field(description="List of paper IDs involved in the answer")

class QAHealthResponse(BaseModel):
    """Health check response for QA services"""
    status: str = Field(description="Overall status")
    components: Dict[str, Dict[str, Any]] = Field(description="Individual service statuses")


def get_qa_agent() -> QAAgent:
    """Get (and lazily initialize) the singleton QAAgent instance."""
    global qa_agent
    if qa_agent is not None:
        return qa_agent
    try:
        es_service = ElasticsearchSearchService(
            es_host=config.ES_HOST,
            index_name=config.ES_INDEX_NAME,
            bge_model=config.BGE_MODEL_NAME,
            bge_cache_dir=config.BGE_CACHE_DIR,
        )
        qa_agent_obj = QAAgent(es_service=es_service)
        log.info("QA agent initialized", extra={
            "es_host": config.ES_HOST,
            "index": config.ES_INDEX_NAME,
            "bge_model": config.BGE_MODEL_NAME,
        })
        qa_agent = qa_agent_obj
        return qa_agent
    except Exception as e:
        log.error(f"Failed to initialize QA agent: {e}")
        raise HTTPException(status_code=503, detail="QA agent unavailable")


def _build_api_response(agent_response, explicit_paper_ids: Optional[List[str]] = None) -> QAResponse:
    """Internal helper to convert QAAgent.QAResponse -> API QAResponse.

    If explicit_paper_ids is provided (single-paper flow), we use that directly; otherwise
    we derive the involved papers from the context chunks.
    """
    involved = explicit_paper_ids or list({c.paper_id for c in agent_response.context_chunks})
    return QAResponse(
        answer=agent_response.answer,
        sources=agent_response.sources,
        confidence_score=agent_response.confidence_score,
        processing_time=agent_response.processing_time,
        context_chunks_count=len(agent_response.context_chunks),
        papers_involved=involved,
    )


@router.get("/health", response_model=QAHealthResponse)
async def qa_health():
    """Health check for QA services"""
    try:
        agent = get_qa_agent()
        health = await agent.health_check()
        
        return QAHealthResponse(
            status=health["status"],
            components=health["components"]
        )

    except Exception as e:
        log.error(f"QA health check failed: {e}")
        return QAHealthResponse(
            status="unhealthy",
            components={
                "error": {
                    "status": "unhealthy",
                    "message": str(e)
                }
            }
        )


@router.post("/single-paper", response_model=QAResponse)
async def single_paper_qa(request: SinglePaperQARequest):
    """
    Answer a question about a specific paper.
    
    This endpoint allows you to ask questions about a single research paper.
    The system will retrieve relevant context chunks from the paper and generate
    an answer based on the paper's content.
    """
    try:
        log.info(f"Single-paper QA request: paper_id={request.paper_id}, question='{request.question}'")
        
        # Get QA agent
        agent = get_qa_agent()
        
        # Execute single-paper QA
        response = await agent.answer_single_paper_question(
            paper_id=request.paper_id,
            question=request.question,
            max_chunks=request.max_chunks
        )
        
        return _build_api_response(response, explicit_paper_ids=[request.paper_id])

    except Exception as e:
        log.exception("Single-paper QA failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-paper", response_model=QAResponse)
async def multi_paper_qa(request: MultiPaperQARequest):
    """
    Answer a question across multiple papers.
    
    This endpoint allows you to ask questions that require information from
    multiple research papers. The system will retrieve relevant context chunks
    from all specified papers and generate a comprehensive answer.
    """
    try:
        log.info(f"Multi-paper QA request: {len(request.paper_ids)} papers, question='{request.question}'")
        
        # Get QA agent
        agent = get_qa_agent()
        
        # Execute multi-paper QA
        response = await agent.answer_multi_paper_question(
            paper_ids=request.paper_ids,
            question=request.question,
            max_chunks_per_paper=request.max_chunks_per_paper
        )
        
        return _build_api_response(response)

    except Exception as e:
        log.exception("Multi-paper QA failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-results", response_model=QAResponse)
async def search_results_qa(request: SearchResultsQARequest):
    """
    Answer a question using papers from a search query.
    
    This endpoint allows you to ask questions using papers returned by a search query.
    The system will first retrieve papers matching the search query, then find relevant
    context chunks from those papers to answer your question.
    """
    try:
        log.info(f"Search results QA request: query='{request.search_query}', question='{request.question}'")
        
        # Get QA agent
        agent = get_qa_agent()
        
        # Execute search results QA
        response = await agent.answer_search_results_question(
            search_query=request.search_query,
            question=request.question,
            max_papers=request.max_papers,
            max_chunks_per_paper=request.max_chunks_per_paper
        )
        
        return _build_api_response(response)

    except Exception as e:
        log.exception("Search results QA failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed", response_model=QAResponse)
async def mixed_qa(request: MixedQARequest):
    """Answer a question using any combination of selected paper IDs and / or a new search query.

    This supports the UI workflow: user bookmarks some papers (paper_ids), then submits
    a new question possibly with a fresh search query producing additional dynamic context.
    At least one of paper_ids or search_query must be provided.
    """
    if (not request.paper_ids or len(request.paper_ids) == 0) and not request.search_query:
        raise HTTPException(status_code=422, detail="Provide at least paper_ids or search_query")
    try:
        agent = get_qa_agent()
        response = await agent.answer_mixed_question(
            question=request.question,
            paper_ids=request.paper_ids,
            search_query=request.search_query,
            max_chunks_per_selected=request.max_chunks_per_selected,
            max_search_papers=request.max_search_papers,
            max_search_chunks_per_paper=request.max_search_chunks_per_paper,
        )
        return _build_api_response(response)
    except Exception as e:
        log.exception("Mixed QA failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers/{paper_id}/context")
async def get_paper_context(paper_id: str, question: str = None, max_chunks: int = 5):
    """
    Get context chunks for a specific paper.
    
    This endpoint allows you to retrieve relevant context chunks from a paper
    for a given question, useful for understanding what information the QA
    system would use to answer questions about the paper.
    """
    try:
        log.info(f"Getting context for paper {paper_id}")
        
        # Get QA agent
        agent = get_qa_agent()
        
        # Retrieve context chunks
        context_chunks = await agent.retrieval_tool.retrieve_single_paper_context(
            paper_id=paper_id,
            question=question or "general information",
            max_chunks=max_chunks
        )
        
        # Convert to response format
        chunks_data = []
        for chunk in context_chunks:
            chunks_data.append({
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "section_path": chunk.section_path,
                "page_number": chunk.page_number,
                "relevance_score": chunk.relevance_score,
                "image_urls": chunk.image_urls
            })
        
        return {
            "paper_id": paper_id,
            "paper_title": context_chunks[0].paper_title if context_chunks else "Unknown",
            "question": question,
            "chunks": chunks_data,
            "total_chunks": len(chunks_data)
        }

    except Exception as e:
        log.error(f"Error getting paper context: {e}")
        raise HTTPException(status_code=500, detail=str(e))
