"""
LlamaIndex agent endpoints for intelligent paper search.

Provides AI-powered search capabilities using ReAct agent with Gemini.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from backend.services import ElasticsearchSearchService
from backend.config import config
from backend.llama_agent.agent import PaperSearchAgent
from backend.llama_agent.response_builder import FormattedResponse

logger = logging.getLogger(__name__)

# Create router for LlamaIndex endpoints
router = APIRouter(prefix="/api/v1/llama", tags=["LlamaIndex Agent"])

# Global agent instance
paper_agent: Optional[PaperSearchAgent] = None


# Pydantic models for API
class LlamaSearchRequest(BaseModel):
    """Request model for LlamaIndex search"""
    query: str = Field(..., description="User's paper search query", min_length=1, max_length=1000)
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    enable_iterations: bool = Field(default=True, description="Enable iterative search refinement")
    include_summaries: bool = Field(default=True, description="Include AI-generated summaries")


class LlamaSearchResponse(BaseModel):
    """Response model for LlamaIndex search with evidence chunks"""
    success: bool = Field(description="Whether the request was successful")
    query: str = Field(description="Original query")
    papers: List[Dict[str, Any]] = Field(description="Found papers with evidence chunks")
    total_found: int = Field(description="Total papers found")
    search_iterations: int = Field(description="Number of search iterations performed")
    error: Optional[str] = Field(default=None, description="Error message if any")


class LlamaChatRequest(BaseModel):
    """Request model for chat interaction"""
    message: str = Field(..., description="User message to the agent")


class LlamaChatResponse(BaseModel):
    """Response model for chat interaction"""
    response: str = Field(description="Agent's response")
    success: bool = Field(default=True, description="Whether the request was successful")


class LlamaHealthResponse(BaseModel):
    """Health check response for LlamaIndex services"""
    status: str = Field(description="Overall status")
    services: Dict[str, Dict[str, Any]] = Field(description="Individual service statuses")


def get_paper_agent() -> PaperSearchAgent:
    """Dependency to get paper search agent"""
    global paper_agent
    if paper_agent is None:
        try:
            # Initialize Elasticsearch service
            es_service = ElasticsearchSearchService(
                es_host=config.ES_HOST,
                index_name=config.ES_INDEX_NAME,
                bge_model=config.BGE_MODEL_NAME,
                bge_cache_dir=config.BGE_CACHE_DIR
            )

            # Initialize agent
            paper_agent = PaperSearchAgent(es_service=es_service)
            logger.info("LlamaIndex paper search agent initialized")

        except Exception as e:
            logger.error(f"Failed to initialize paper search agent: {e}")
            raise HTTPException(status_code=503, detail="Paper search agent unavailable")

    return paper_agent


@router.get("/health", response_model=LlamaHealthResponse)
async def llama_health():
    """Health check for LlamaIndex services"""
    try:
        agent = get_paper_agent()

        # Check if agent components are initialized
        agent_health = {
            "status": "healthy",
            "components": {
                "llm": "initialized" if agent.llm else "not initialized",
                "search_tool": "initialized" if agent.search_tool else "not initialized",
                "query_analyzer": "initialized" if agent.query_analyzer else "not initialized",
                "response_builder": "initialized" if agent.response_builder else "not initialized",
                "react_agent": "initialized" if agent.agent else "not initialized"
            }
        }

        # Check Elasticsearch
        es_health = {"status": "unknown"}
        try:
            es_health = agent.search_tool.es_service.health_check()
        except Exception as e:
            es_health = {"status": "unhealthy", "error": str(e)}

        services = {
            "llama_agent": agent_health,
            "elasticsearch": es_health
        }

        # Overall status
        overall_status = "healthy"
        if agent_health["status"] != "healthy" or es_health.get("status") != "healthy":
            overall_status = "unhealthy"

        return LlamaHealthResponse(
            status=overall_status,
            services=services
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return LlamaHealthResponse(
            status="unhealthy",
            services={
                "error": {
                    "status": "unhealthy",
                    "message": str(e)
                }
            }
        )


@router.post("/search", response_model=LlamaSearchResponse)
async def llama_search(request: LlamaSearchRequest):
    """
    Intelligent paper search using LlamaIndex ReAct agent with Gemini.

    This endpoint provides:
    - Query understanding and analysis
    - Iterative search refinement
    - Paper results with relevant text evidence chunks
    - Relevance ranking and deduplication
    """
    try:
        logger.info(
            f"LlamaIndex search request: query='{request.query}', "
            f"max_results={request.max_results}, iterations={request.enable_iterations}"
        )

        # Get agent
        agent = get_paper_agent()

        # Update config based on request
        from backend.llama_agent.config import llama_config
        original_max_iterations = llama_config.max_iterations
        original_include_summaries = llama_config.include_summaries

        if not request.enable_iterations:
            llama_config.max_iterations = 1
        llama_config.include_summaries = request.include_summaries
        llama_config.response_max_papers = request.max_results

        try:
            # Execute search
            response: FormattedResponse = await agent.search(request.query)

            # Convert to API response
            return LlamaSearchResponse(
                success=response.success,
                query=response.query,
                papers=response.papers,
                total_found=response.total_found,
                search_iterations=response.search_iterations,
                error=response.error
            )

        finally:
            # Restore original config
            llama_config.max_iterations = original_max_iterations
            llama_config.include_summaries = original_include_summaries

    except Exception as e:
        logger.exception("LlamaIndex search failed")
        return LlamaSearchResponse(
            success=False,
            query=request.query,
            papers=[],
            total_found=0,
            search_iterations=0,
            error=str(e)
        )


# Chat endpoint removed - ReActAgent doesn't support achat interface
# Use the search endpoint for all queries