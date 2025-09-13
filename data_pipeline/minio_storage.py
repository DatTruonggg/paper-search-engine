"""
MinIO storage client for PDF and markdown file management.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import minio
from minio import Minio
from minio.error import S3Error
import os

logger = logging.getLogger(__name__)


class MinIOStorage:
    """MinIO client for storing and retrieving PDF and markdown files."""

    def __init__(
        self,
        endpoint: str = "localhost:9002",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
        pdf_bucket: str = "papers",
        markdown_bucket: str = "markdown"
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Use HTTPS
            pdf_bucket: Bucket name for PDF files
            markdown_bucket: Bucket name for markdown files
        """
        self.endpoint = endpoint
        self.pdf_bucket = pdf_bucket
        self.markdown_bucket = markdown_bucket

        # Initialize MinIO client
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        logger.info(f"MinIO client initialized: {endpoint}")
        self._ensure_buckets_exist()

    def _ensure_buckets_exist(self):
        """Ensure required buckets exist."""
        for bucket in [self.pdf_bucket, self.markdown_bucket]:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    logger.info(f"Created MinIO bucket: {bucket}")
                else:
                    logger.info(f"MinIO bucket exists: {bucket}")
            except S3Error as e:
                logger.error(f"Error with bucket {bucket}: {e}")
                raise

    def upload_pdf(self, paper_id: str, pdf_path: Path) -> Optional[str]:
        """
        Upload PDF file to MinIO.

        Args:
            paper_id: Paper identifier
            pdf_path: Local path to PDF file

        Returns:
            MinIO URL for the uploaded file or None if failed
        """
        if not pdf_path.exists():
            logger.warning(f"PDF file not found: {pdf_path}")
            return None

        object_name = f"{paper_id}.pdf"

        try:
            # Upload file
            self.client.fput_object(
                bucket_name=self.pdf_bucket,
                object_name=object_name,
                file_path=str(pdf_path),
                content_type="application/pdf"
            )

            # Generate URL
            minio_url = f"http://{self.endpoint}/{self.pdf_bucket}/{object_name}"
            logger.info(f"Uploaded PDF: {paper_id} -> {minio_url}")
            return minio_url

        except S3Error as e:
            logger.error(f"Failed to upload PDF {paper_id}: {e}")
            return None

    def upload_markdown(self, paper_id: str, markdown_path: Path) -> Optional[str]:
        """
        Upload markdown file to MinIO.

        Args:
            paper_id: Paper identifier
            markdown_path: Local path to markdown file

        Returns:
            MinIO URL for the uploaded file or None if failed
        """
        if not markdown_path.exists():
            logger.warning(f"Markdown file not found: {markdown_path}")
            return None

        object_name = f"{paper_id}.md"

        try:
            # Upload file
            self.client.fput_object(
                bucket_name=self.markdown_bucket,
                object_name=object_name,
                file_path=str(markdown_path),
                content_type="text/markdown"
            )

            # Generate URL
            minio_url = f"http://{self.endpoint}/{self.markdown_bucket}/{object_name}"
            logger.info(f"Uploaded markdown: {paper_id} -> {minio_url}")
            return minio_url

        except S3Error as e:
            logger.error(f"Failed to upload markdown {paper_id}: {e}")
            return None

    def get_pdf_url(self, paper_id: str) -> Optional[str]:
        """
        Get presigned URL for PDF download.

        Args:
            paper_id: Paper identifier

        Returns:
            Presigned URL for PDF download
        """
        object_name = f"{paper_id}.pdf"

        try:
            # Check if object exists
            self.client.stat_object(self.pdf_bucket, object_name)

            # Generate presigned URL (valid for 1 hour)
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name=self.pdf_bucket,
                object_name=object_name,
                expires=timedelta(seconds=3600)
            )
            return url

        except S3Error as e:
            logger.warning(f"PDF not found in MinIO: {paper_id}")
            return None

    def get_markdown_url(self, paper_id: str) -> Optional[str]:
        """
        Get presigned URL for markdown download.

        Args:
            paper_id: Paper identifier

        Returns:
            Presigned URL for markdown download
        """
        object_name = f"{paper_id}.md"

        try:
            # Check if object exists
            self.client.stat_object(self.markdown_bucket, object_name)

            # Generate presigned URL (valid for 1 hour)
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name=self.markdown_bucket,
                object_name=object_name,
                expires=timedelta(seconds=3600)
            )
            return url

        except S3Error as e:
            logger.warning(f"Markdown not found in MinIO: {paper_id}")
            return None

    def delete_paper_files(self, paper_id: str) -> Dict[str, bool]:
        """
        Delete both PDF and markdown files for a paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Dictionary with deletion status for each file type
        """
        results = {"pdf": False, "markdown": False}

        # Delete PDF
        try:
            self.client.remove_object(self.pdf_bucket, f"{paper_id}.pdf")
            results["pdf"] = True
            logger.info(f"Deleted PDF from MinIO: {paper_id}")
        except S3Error as e:
            logger.warning(f"Could not delete PDF {paper_id}: {e}")

        # Delete markdown
        try:
            self.client.remove_object(self.markdown_bucket, f"{paper_id}.md")
            results["markdown"] = True
            logger.info(f"Deleted markdown from MinIO: {paper_id}")
        except S3Error as e:
            logger.warning(f"Could not delete markdown {paper_id}: {e}")

        return results

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        stats = {
            "endpoint": self.endpoint,
            "buckets": {},
            "total_objects": 0,
            "total_size_bytes": 0
        }

        for bucket in [self.pdf_bucket, self.markdown_bucket]:
            try:
                objects = list(self.client.list_objects(bucket))
                object_count = len(objects)
                total_size = sum(obj.size for obj in objects)

                stats["buckets"][bucket] = {
                    "object_count": object_count,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }

                stats["total_objects"] += object_count
                stats["total_size_bytes"] += total_size

            except S3Error as e:
                logger.error(f"Error getting stats for bucket {bucket}: {e}")
                stats["buckets"][bucket] = {"error": str(e)}

        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Check MinIO service health.

        Returns:
            Health status dictionary
        """
        try:
            # Try to list buckets to test connectivity
            buckets = list(self.client.list_buckets())

            return {
                "status": "healthy",
                "endpoint": self.endpoint,
                "buckets_accessible": len(buckets),
                "required_buckets": {
                    self.pdf_bucket: self.client.bucket_exists(self.pdf_bucket),
                    self.markdown_bucket: self.client.bucket_exists(self.markdown_bucket)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint,
                "error": str(e)
            }


def main():
    """Test MinIO storage functionality."""
    # Initialize storage
    storage = MinIOStorage()

    # Get health check
    health = storage.health_check()
    print(f"MinIO Health: {health}")

    # Get storage stats
    stats = storage.get_storage_stats()
    print(f"Storage Stats: {stats}")


if __name__ == "__main__":
    main()