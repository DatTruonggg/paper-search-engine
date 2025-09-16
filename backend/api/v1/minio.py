"""
MinIO and storage-related API routes.

Provides endpoints for PDF info, upload, listing, and presigned download URLs.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException, Request, Query

from backend.config import config


router = APIRouter(prefix="/api/v1", tags=["Storage"])


@router.get("/papers/{paper_id}/pdf")
async def get_pdf_url(paper_id: str, request: Request):
    """Return a presigned MinIO URL for a paper's PDF if available."""
    search_service = getattr(request.app.state, "search_service", None)
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        paper = search_service.get_paper_details(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        if not paper.minio_pdf_url:
            raise HTTPException(status_code=404, detail=f"PDF not available in MinIO for paper: {paper_id}")

        from data_pipeline.minio_storage import MinIOStorage

        minio_storage = MinIOStorage(endpoint=config.MINIO_ENDPOINT)
        presigned_url = minio_storage.get_pdf_url(paper_id)

        if not presigned_url:
            raise HTTPException(status_code=404, detail=f"Could not generate PDF download URL for: {paper_id}")

        return {
            "paper_id": paper_id,
            "title": paper.title,
            "download_url": presigned_url,
            "expires_in_seconds": 3600,
            "file_size_bytes": paper.pdf_size,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers/{paper_id}/info")
async def get_paper_with_pdf_info(paper_id: str):
    """Return paper information along with MinIO PDF availability metadata."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()
        paper_info = paper_service.get_paper_info(paper_id)

        if not paper_info:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        return paper_info
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/storage/summary")
async def get_storage_summary():
    """Return a summary comparing indexed papers to available PDFs in storage."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()
        return paper_service.get_summary()
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/storage/upload-pdf")
async def upload_pdf(paper_id: str):
    """Upload a PDF for a specific paper from a configured local directory."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()

        pdf_dir = Path(config.PDF_LOCAL_DIR)
        pdf_path = pdf_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")

        result = paper_service.upload_pdf_for_paper(paper_id, pdf_path)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/storage/pdf/{paper_id}/download")
async def get_pdf_download_url(paper_id: str):
    """Return a presigned MinIO URL to directly download a paper PDF."""
    try:
        from backend.services.pdf_service import PDFService

        pdf_service = PDFService()
        if not pdf_service.pdf_exists(paper_id):
            raise HTTPException(status_code=404, detail=f"PDF not found for: {paper_id}")

        download_url = pdf_service.get_pdf_url(paper_id)
        if not download_url:
            raise HTTPException(status_code=500, detail="Could not generate download URL")

        return {"paper_id": paper_id, "download_url": download_url, "expires_in": "1 hour"}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))