"""
Simple PDF storage service using MinIO.
Handles PDF upload/download by paper ID (DOI).
"""

from logs import log
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import timedelta

from minio import Minio
from minio.error import S3Error



class PDFService:
    """Simple service to manage PDFs in MinIO storage."""

    def __init__(
        self,
        endpoint: str = "localhost:9002",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = "papers"
    ):
        """Initialize PDF service with MinIO."""
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        # Ensure bucket exists
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                log.info(f"Created bucket: {bucket_name}")
        except S3Error as e:
            log.error(f"Error creating bucket: {e}")

        log.info(f"PDF service initialized with bucket: {bucket_name}")

    def upload_pdf(self, paper_id: str, pdf_path: Path) -> Dict[str, Any]:
        """
        Upload a PDF file to MinIO.

        Args:
            paper_id: Paper ID (DOI)
            pdf_path: Path to PDF file

        Returns:
            Result dictionary with status and URL
        """
        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}"
            }

        # Use paper_id as object name
        object_name = f"{paper_id.replace('/', '_')}.pdf"

        try:
            # Upload file
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(pdf_path),
                content_type="application/pdf"
            )

            return {
                "success": True,
                "paper_id": paper_id,
                "object_name": object_name,
                "file_size": pdf_path.stat().st_size
            }

        except S3Error as e:
            log.error(f"Failed to upload PDF for {paper_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_pdf_url(self, paper_id: str) -> Optional[str]:
        """
        Get a download URL for a PDF.

        Args:
            paper_id: Paper ID (DOI)

        Returns:
            Presigned download URL or None
        """
        object_name = f"{paper_id.replace('/', '_')}.pdf"

        try:
            # Check if object exists
            self.client.stat_object(self.bucket_name, object_name)

            # Generate presigned URL (1 hour expiry)
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=timedelta(hours=1)
            )
            return url

        except S3Error:
            log.warning(f"PDF not found for paper_id: {paper_id}")
            return None

    def pdf_exists(self, paper_id: str) -> bool:
        """Check if PDF exists for a paper."""
        object_name = f"{paper_id.replace('/', '_')}.pdf"
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error:
            return False

    def list_pdfs(self) -> List[str]:
        """List all PDF paper IDs in storage."""
        try:
            objects = self.client.list_objects(self.bucket_name)
            paper_ids = []

            for obj in objects:
                # Convert object name back to paper_id
                paper_id = obj.object_name.replace('.pdf', '').replace('_', '/')
                paper_ids.append(paper_id)

            return paper_ids

        except S3Error as e:
            log.error(f"Error listing PDFs: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            objects = list(self.client.list_objects(self.bucket_name))

            total_count = len(objects)
            total_size = sum(obj.size for obj in objects)

            return {
                "total_pdfs": total_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "bucket_name": self.bucket_name
            }

        except S3Error as e:
            log.error(f"Error getting stats: {e}")
            return {"error": str(e)}