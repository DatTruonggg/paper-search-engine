"""
QA API endpoints for single-paper and multi-paper question answering.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Request

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
    paper_ids: List[str] = Field(..., description="List of paper IDs to search across", min_items=1, max_items=10)
    question: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    max_chunks_per_paper: int = Field(default=3, ge=1, le=10, description="Maximum chunks per paper")


class SearchResultsQARequest(BaseModel):
    """Request model for search results QA"""
    search_query: str = Field(..., description="Original search query that returned papers", min_length=1)
    question: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    max_papers: int = Field(default=5, ge=1, le=10, description="Maximum number of papers to include")
    max_chunks_per_paper: int = Field(default=2, ge=1, le=5, description="Maximum chunks per paper")


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
    """Dependency to get QA agent"""
    global qa_agent
    if qa_agent is None:
        try:
            # Initialize Elasticsearch service
            es_service = ElasticsearchSearchService(
                es_host=config.ES_HOST,
                index_name=config.ES_INDEX_NAME,
                bge_model=config.BGE_MODEL_NAME,
                bge_cache_dir=config.BGE_CACHE_DIR
            )

            # Initialize QA agent
            qa_agent = QAAgent(es_service=es_service)
            log.info("QA agent initialized successfully")

        except Exception as e:
            log.error(f"Failed to initialize QA agent: {e}")
            raise HTTPException(status_code=503, detail="QA agent unavailable")

    return qa_agent


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
        
        # Convert to API response
        return QAResponse(
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            context_chunks_count=len(response.context_chunks),
            papers_involved=[request.paper_id]
        )

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
        
        # Convert to API response
        return QAResponse(
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            context_chunks_count=len(response.context_chunks),
            papers_involved=list(set(chunk.paper_id for chunk in response.context_chunks))
        )

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
        
        # Convert to API response
        return QAResponse(
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            context_chunks_count=len(response.context_chunks),
            papers_involved=list(set(chunk.paper_id for chunk in response.context_chunks))
        )

    except Exception as e:
        log.exception("Search results QA failed")
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
