#!/usr/bin/env python3
"""
Separate PDF uploader service that uploads existing PDFs to MinIO
and updates Elasticsearch with MinIO URLs.
"""

import os
import sys
import argparse
from logs import log
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.minio_storage import MinIOStorage
from data_pipeline.es_indexer import ESIndexer


class PDFUploader:
    """Service to upload PDFs to MinIO and update ES with URLs."""

    def __init__(
        self,
        pdf_dir: str = "/Users/admin/code/cazoodle/data/pdfs",
        minio_endpoint: str = "localhost:9002",
        es_host: str = "localhost:9202",
        es_index: str = "papers"
    ):
        """
        Initialize PDF uploader.

        Args:
            pdf_dir: Directory containing PDF files
            minio_endpoint: MinIO server endpoint
            es_host: Elasticsearch host
            es_index: Elasticsearch index name
        """
        self.pdf_dir = Path(pdf_dir)

        # Initialize MinIO storage
        self.minio_storage = MinIOStorage(endpoint=minio_endpoint)

        # Initialize ES indexer for updates
        self.es_indexer = ESIndexer(es_host=es_host, index_name=es_index)

        log.info(f"PDF Uploader initialized")
        log.info(f"PDF directory: {self.pdf_dir}")
        log.info(f"MinIO endpoint: {minio_endpoint}")
        log.info(f"ES host: {es_host}")

    def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the PDF directory."""
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        log.info(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def get_paper_id_from_pdf_path(self, pdf_path: Path) -> str:
        """Extract paper ID from PDF filename."""
        return pdf_path.stem  # Remove .pdf extension

    def upload_pdf_and_update_es(self, pdf_path: Path) -> bool:
        """
        Upload a PDF to MinIO and update Elasticsearch with the URL.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if successful, False otherwise
        """
        paper_id = self.get_paper_id_from_pdf_path(pdf_path)

        try:
            # Upload PDF to MinIO
            minio_pdf_url = self.minio_storage.upload_pdf(paper_id, pdf_path)

            if not minio_pdf_url:
                log.error(f"Failed to upload PDF to MinIO: {paper_id}")
                return False

            # Check if document exists in ES
            document = self.es_indexer.get_document(paper_id)
            if not document:
                log.warning(f"Document not found in ES, skipping URL update: {paper_id}")
                return False

            # Update document with MinIO URL
            update_body = {
                "doc": {
                    "minio_pdf_url": minio_pdf_url,
                    "pdf_storage_updated_at": self.es_indexer._get_current_timestamp()
                }
            }

            response = self.es_indexer.es.update(
                index=self.es_indexer.index_name,
                id=paper_id,
                body=update_body
            )

            log.info(f"Updated ES document with MinIO URL: {paper_id}")
            return True

        except Exception as e:
            log.error(f"Error processing {paper_id}: {e}")
            return False

    def upload_all_pdfs(self, max_files: Optional[int] = None) -> Dict[str, int]:
        """
        Upload all PDFs to MinIO and update Elasticsearch.

        Args:
            max_files: Maximum number of files to process

        Returns:
            Statistics dictionary
        """
        pdf_files = self.find_pdf_files()

        if max_files:
            pdf_files = pdf_files[:max_files]

        stats = {
            "total_found": len(pdf_files),
            "uploaded": 0,
            "updated": 0,
            "failed": 0,
            "skipped": 0
        }

        log.info(f"Starting upload of {len(pdf_files)} PDF files...")

        for pdf_path in tqdm(pdf_files, desc="Uploading PDFs"):
            paper_id = self.get_paper_id_from_pdf_path(pdf_path)

            # Check if already uploaded
            if self._is_already_uploaded(paper_id):
                log.debug(f"PDF already uploaded, skipping: {paper_id}")
                stats["skipped"] += 1
                continue

            success = self.upload_pdf_and_update_es(pdf_path)

            if success:
                stats["uploaded"] += 1
                stats["updated"] += 1
            else:
                stats["failed"] += 1

        log.info(f"PDF upload completed: {stats}")
        return stats

    def _is_already_uploaded(self, paper_id: str) -> bool:
        """Check if a paper's PDF is already uploaded to MinIO."""
        try:
            document = self.es_indexer.get_document(paper_id)
            if document and document.get("minio_pdf_url"):
                return True
            return False
        except Exception:
            return False

    def get_upload_stats(self) -> Dict:
        """Get statistics about uploaded PDFs."""
        try:
            # Get all documents with MinIO URLs
            search_body = {
                "query": {
                    "exists": {
                        "field": "minio_pdf_url"
                    }
                },
                "size": 0,
                "aggs": {
                    "total_with_minio_urls": {
                        "value_count": {
                            "field": "minio_pdf_url"
                        }
                    }
                }
            }

            response = self.es_indexer.es.search(
                index=self.es_indexer.index_name,
                body=search_body
            )

            # Get MinIO storage stats
            minio_stats = self.minio_storage.get_storage_stats()

            return {
                "es_documents_with_minio_urls": response["aggregations"]["total_with_minio_urls"]["value"],
                "minio_storage": minio_stats,
                "pdf_directory": {
                    "path": str(self.pdf_dir),
                    "total_pdf_files": len(list(self.pdf_dir.glob("*.pdf")))
                }
            }

        except Exception as e:
            log.error(f"Error getting upload stats: {e}")
            return {"error": str(e)}


def main():
    """Main function for PDF uploader."""
    parser = argparse.ArgumentParser(description="Upload PDFs to MinIO and update Elasticsearch")

    parser.add_argument(
        "--pdf-dir",
        default="/Users/admin/code/cazoodle/data/pdfs",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--minio-endpoint",
        default="localhost:9002",
        help="MinIO server endpoint"
    )
    parser.add_argument(
        "--es-host",
        default="localhost:9202",
        help="Elasticsearch host"
    )
    parser.add_argument(
        "--es-index",
        default="papers",
        help="Elasticsearch index name"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't upload"
    )

    args = parser.parse_args()

    # Initialize uploader
    uploader = PDFUploader(
        pdf_dir=args.pdf_dir,
        minio_endpoint=args.minio_endpoint,
        es_host=args.es_host,
        es_index=args.es_index
    )

    if args.stats_only:
        # Show statistics only
        stats = uploader.get_upload_stats()
        log.info("\n=== PDF Upload Statistics ===")
        log.info(json.dumps(stats, indent=2))
    else:
        # Upload PDFs
        stats = uploader.upload_all_pdfs(max_files=args.max_files)
        log.info(f"\n=== Upload Results ===")
        log.info(f"Total PDFs found: {stats['total_found']}")
        log.info(f"Successfully uploaded: {stats['uploaded']}")
        log.info(f"ES documents updated: {stats['updated']}")
        log.info(f"Failed: {stats['failed']}")
        log.info(f"Skipped (already uploaded): {stats['skipped']}")


if __name__ == "__main__":
    main()