"""
Search endpoints and models for paper search API.

Provides search functionality for papers with multiple search modes:
- Title/abstract search
- Full paper content search
- Search suggestions
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Search"])


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    search_mode: str = Field("hybrid", description="Search mode: hybrid, semantic, fulltext")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    date_from: Optional[str] = Field(None, description="Filter papers from this date")
    date_to: Optional[str] = Field(None, description="Filter papers to this date")
    author: Optional[str] = Field(None, description="Filter by author name")

    def model_post_init(self, __context) -> None:
        """Auto-clean filter values - set 'string' literals to None"""
        if self.categories == ["string"]:
            self.categories = None
        if self.date_from == "string":
            self.date_from = None
        if self.date_to == "string":
            self.date_to = None
        if self.author == "string":
            self.author = None


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[Dict[str, Any]]
    total: int
    query: str
    search_mode: str


@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(request: SearchRequest):
    """
    Search for papers in title and abstract only.

    Search modes:
    - hybrid: Combines BM25 (30%) and semantic search (70%)
    - semantic: Pure embedding-based similarity search
    - fulltext: Traditional BM25 full-text search

    Note: This endpoint searches only in title and abstract fields for faster results.
    Use /api/v1/search_full_papers to search within paper content.
    """
    from fastapi import Request
    from backend.services import ElasticsearchSearchService

    # Get search service from app state
    # Note: This is a workaround since we can't inject Request directly in this structure
    # The search_service will be available globally after startup
    search_service = None
    try:
        from backend.api.main import search_service as global_search_service
        search_service = global_search_service
    except ImportError:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        logger.info(f"Search request: query='{request.query}', mode={request.search_mode}")

        results = search_service.search(
            query=request.query,
            max_results=request.max_results,
            search_mode=request.search_mode,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
            author=request.author,
            include_chunks=False  # Title/abstract only search
        )

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "score": result.score,
                "categories": result.categories,
                "publish_date": result.publish_date,
                "word_count": result.word_count,
                "has_images": result.has_images,
                "pdf_size": result.pdf_size
            })

        return SearchResponse(
            results=results_dict,
            total=len(results),
            query=request.query,
            search_mode=request.search_mode
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search_full_papers", response_model=SearchResponse, tags=["Search"])
async def search_full_papers(request: SearchRequest):
    """
    Search for papers in title, abstract, and full paper content (chunks).

    Search modes:
    - hybrid: Combines BM25 (30%) and semantic search (70%)
    - semantic: Pure embedding-based similarity search
    - fulltext: Traditional BM25 full-text search

    Note: This endpoint searches in title, abstract, AND paper content (chunks).
    This provides deeper search but may be slower. Use /api/v1/search for title/abstract only.
    """
    from backend.api.main import search_service as global_search_service
    search_service = global_search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        logger.info(f"Full paper search request: query='{request.query}', mode={request.search_mode}")

        results = search_service.search(
            query=request.query,
            max_results=request.max_results,
            search_mode=request.search_mode,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
            author=request.author,
            include_chunks=True  # Full paper search including chunks
        )

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "score": result.score,
                "categories": result.categories,
                "publish_date": result.publish_date,
                "word_count": result.word_count,
                "has_images": result.has_images,
                "pdf_size": result.pdf_size
            })

        return SearchResponse(
            results=results_dict,
            total=len(results),
            query=request.query,
            search_mode=request.search_mode
        )

    except Exception as e:
        logger.error(f"Full paper search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/suggest", tags=["Search"])
async def search_suggestions(
    query: str = Query(..., description="Query prefix for suggestions"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum number of suggestions")
):
    """
    Get search suggestions based on query prefix.
    This is a simplified version - in production, you'd want to implement
    proper autocomplete with a dedicated suggestion index.
    """
    from backend.api.main import search_service as global_search_service
    search_service = global_search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Search only in titles for suggestions
        results = search_service.search(
            query=query,
            max_results=max_results,
            search_mode="fulltext",
            include_chunks=False  # Title/abstract only for suggestions
        )

        suggestions = [result.title for result in results]

        return {
            "query": query,
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))