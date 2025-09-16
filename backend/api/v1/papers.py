"""
Paper endpoints and models for paper management API.

Provides functionality for:
- Getting paper details
- Finding similar papers
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/papers", tags=["Papers"])


class SimilarPapersRequest(BaseModel):
    """Similar papers request model"""
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")


@router.get("/{paper_id}", tags=["Papers"])
async def get_paper(paper_id: str):
    """
    Get detailed information about a specific paper.
    """
    from backend.api.main import search_service as global_search_service
    search_service = global_search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        paper = search_service.get_paper_details(paper_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "content": paper.content[:1000] + "..." if len(paper.content) > 1000 else paper.content,
            "content_length": len(paper.content),
            "categories": paper.categories,
            "publish_date": paper.publish_date,
            "word_count": paper.word_count,
            "chunk_count": paper.chunk_count,
            "has_images": paper.has_images,
            "pdf_size": paper.pdf_size,
            "downloaded_at": paper.downloaded_at,
            "indexed_at": paper.indexed_at,
            "markdown_path": paper.markdown_path,
            "pdf_path": paper.pdf_path,
            "minio_pdf_url": paper.minio_pdf_url,
            "minio_markdown_url": paper.minio_markdown_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{paper_id}/similar", tags=["Papers"])
async def find_similar_papers(paper_id: str, request: SimilarPapersRequest):
    """
    Find papers similar to a given paper.
    """
    from backend.api.main import search_service as global_search_service
    search_service = global_search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        results = search_service.find_similar_papers(
            paper_id=paper_id,
            max_results=request.max_results
        )

        if not results:
            raise HTTPException(status_code=404, detail=f"Reference paper not found: {paper_id}")

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "similarity_score": result.score,
                "categories": result.categories,
                "publish_date": result.publish_date
            })

        return {
            "reference_paper_id": paper_id,
            "similar_papers": results_dict,
            "total": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar papers for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))