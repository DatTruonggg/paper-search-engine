"""
Paper endpoints and models for paper management API.

Provides functionality for:
- Getting paper details
- Finding similar papers
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from logs import log

router = APIRouter(prefix="/api/v1/papers", tags=["Papers"])


class SimilarPapersRequest(BaseModel):
    """Similar papers request model"""
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")


class PapersBatchRequest(BaseModel):
    """Request model for fetching metadata of multiple papers by ID."""
    paper_ids: List[str] = Field(..., min_items=1, max_items=50, description="List of paper IDs to fetch")


class PaperMetadata(BaseModel):
    """Lightweight paper metadata for selection & QA context building (no full content)."""
    paper_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    categories: Optional[List[str]] = None
    publish_date: Optional[str] = None
    chunk_count: Optional[int] = None
    has_images: Optional[bool] = None
    minio_pdf_url: Optional[str] = None
    minio_markdown_url: Optional[str] = None


class PapersBatchResponse(BaseModel):
    total: int
    found: int
    missing: List[str]
    papers: List[PaperMetadata]


from backend.api.main import search_service


@router.get("/{paper_id}", tags=["Papers"])
async def get_paper(paper_id: str, request: Request):
    """
    Get detailed information about a specific paper.
    """
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
            "content": paper.content[:],
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
        log.error(f"Error getting paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{paper_id}/similar", tags=["Papers"])
async def find_similar_papers(paper_id: str, request: SimilarPapersRequest, req: Request):
    """
    Find papers similar to a given paper.
    """
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
        log.error(f"Error finding similar papers for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=PapersBatchResponse, tags=["Papers"])
async def get_papers_batch(batch_request: PapersBatchRequest, request: Request):
    """Fetch lightweight metadata for multiple papers in one request.

    Designed for the UI selection panel: user ticks papers after search, frontend
    calls this to refresh authoritative metadata (title, abstract snippet, links)
    before invoking QA endpoints. Does not return full content to keep payload small.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    unique_ids = list({pid.strip() for pid in batch_request.paper_ids if pid and pid.strip()})
    if not unique_ids:
        raise HTTPException(status_code=400, detail="No valid paper_ids provided")

    found: List[PaperMetadata] = []
    missing: List[str] = []

    for pid in unique_ids:
        try:
            paper = search_service.get_paper_details(pid)
            if not paper:
                missing.append(pid)
                continue
            found.append(PaperMetadata(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                categories=paper.categories,
                publish_date=paper.publish_date,
                chunk_count=paper.chunk_count,
                has_images=paper.has_images,
                minio_pdf_url=paper.minio_pdf_url,
                minio_markdown_url=paper.minio_markdown_url
            ))
        except Exception:
            missing.append(pid)
            continue

    return PapersBatchResponse(
        total=len(unique_ids),
        found=len(found),
        missing=missing,
        papers=found
    )