import asyncio
import httpx
from typing import List, Optional
from minio import Minio
from minio.error import S3Error
from pathlib import Path
import tempfile

from app.schemas import Paper
from app.settings import settings


class MinIOService:
    def __init__(self):
        if not settings.minio_endpoint:
            self.client = None
            return
            
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure the papers bucket exists"""
        if not self.client:
            return
            
        try:
            if not self.client.bucket_exists(settings.minio_bucket):
                self.client.make_bucket(settings.minio_bucket)
        except S3Error as e:
            print(f"Error creating bucket: {e}")
    
    async def download_and_store_pdf(self, paper: Paper) -> bool:
        """Download PDF from ArXiv and store in MinIO"""
        if not self.client or not paper.url_pdf:
            return False
        
        try:
            # Download PDF
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(paper.url_pdf)
                response.raise_for_status()
                
                pdf_content = response.content
            
            # Store in MinIO
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                object_name = f"pdfs/{paper.id}.pdf"
                self.client.fput_object(
                    settings.minio_bucket,
                    object_name,
                    tmp_file.name,
                    content_type="application/pdf"
                )
            
            return True
            
        except Exception as e:
            print(f"Error downloading/storing PDF for {paper.id}: {e}")
            return False
    
    def get_pdf_url(self, paper_id: str) -> Optional[str]:
        """Get presigned URL for PDF"""
        if not self.client:
            return None
        
        try:
            object_name = f"pdfs/{paper_id}.pdf"
            # Check if object exists
            self.client.stat_object(settings.minio_bucket, object_name)
            
            # Generate presigned URL (valid for 1 hour)
            url = self.client.presigned_get_object(
                settings.minio_bucket,
                object_name,
                expires=3600
            )
            return url
        except S3Error:
            return None
    
    def delete_pdf(self, paper_id: str) -> bool:
        """Delete PDF from storage"""
        if not self.client:
            return False
        
        try:
            object_name = f"pdfs/{paper_id}.pdf"
            self.client.remove_object(settings.minio_bucket, object_name)
            return True
        except S3Error:
            return False
