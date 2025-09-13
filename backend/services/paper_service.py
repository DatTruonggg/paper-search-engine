"""
Simple service to combine paper metadata from ES with PDF storage info.
"""

import logging
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services import ElasticsearchSearchService
from backend.services.pdf_service import PDFService

logger = logging.getLogger(__name__)


class PaperService:
    """Service to manage papers with PDF storage info."""

    def __init__(
        self,
        es_host: str = "localhost:9202",
        minio_endpoint: str = "localhost:9002"
    ):
        """Initialize with ES and MinIO connections."""
        self.es_service = ElasticsearchSearchService(es_host=es_host)
        self.pdf_service = PDFService(endpoint=minio_endpoint)
        logger.info("Paper service initialized")

    def get_paper_info(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Get paper info from ES with PDF availability.

        Args:
            paper_id: Paper ID (DOI)

        Returns:
            Paper info with PDF status
        """
        # Get paper from ES
        paper = self.es_service.get_paper_details(paper_id)
        if not paper:
            return None

        # Check PDF availability
        pdf_available = self.pdf_service.pdf_exists(paper_id)
        pdf_url = None

        if pdf_available:
            pdf_url = self.pdf_service.get_pdf_url(paper_id)

        return {
            "paper_id": paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "categories": paper.categories,
            "publish_date": paper.publish_date,
            "pdf_available": pdf_available,
            "pdf_download_url": pdf_url
        }

    def list_papers_with_pdfs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List papers that have PDFs available."""
        # Get PDF IDs from storage
        pdf_ids = self.pdf_service.list_pdfs()[:limit]

        papers = []
        for paper_id in pdf_ids:
            paper_info = self.get_paper_info(paper_id)
            if paper_info:
                papers.append(paper_info)

        return papers

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of papers and PDFs."""
        # Get ES stats
        es_stats = self.es_service.get_index_stats()
        total_papers = es_stats.get("total_papers", 0)

        # Get PDF stats
        pdf_stats = self.pdf_service.get_stats()
        total_pdfs = pdf_stats.get("total_pdfs", 0)

        # Calculate coverage
        coverage = 0
        if total_papers > 0:
            coverage = round((total_pdfs / total_papers) * 100, 2)

        return {
            "total_papers": total_papers,
            "total_pdfs": total_pdfs,
            "pdf_coverage_percent": coverage,
            "index_size_mb": es_stats.get("index_size_mb", 0),
            "pdf_storage_size_mb": pdf_stats.get("total_size_mb", 0)
        }

    def upload_pdf_for_paper(self, paper_id: str, pdf_path: Path) -> Dict[str, Any]:
        """
        Upload PDF for a paper that exists in ES.

        Args:
            paper_id: Paper ID (DOI)
            pdf_path: Path to PDF file

        Returns:
            Upload result
        """
        # Check if paper exists in ES
        paper = self.es_service.get_paper_details(paper_id)
        if not paper:
            return {
                "success": False,
                "error": f"Paper {paper_id} not found in search index"
            }

        # Upload PDF
        result = self.pdf_service.upload_pdf(paper_id, pdf_path)

        if result["success"]:
            result["paper_title"] = paper.title

        return result