import json
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

from app.schemas import Paper, IngestResponse
from app.settings import settings
from app.services.elasticsearch_service import ElasticsearchService
from app.services.postgres_service import PostgresService
from app.services.minio_service import MinIOService


class IngestService:
    def __init__(self):
        self.es_service = ElasticsearchService() if settings.data_backend == "es" else None
        self.pg_service = PostgresService() if settings.data_backend == "pg" else None
        self.minio_service = MinIOService() if settings.minio_endpoint else None
        
    def normalize_paper(self, raw_paper: Dict[str, Any]) -> Paper:
        """Convert raw ArXiv JSON to normalized Paper model"""
        # Parse authors
        authors = []
        if "authors" in raw_paper and raw_paper["authors"]:
            # Split by comma and clean
            authors = [author.strip() for author in raw_paper["authors"].split(",")]
            authors = [re.sub(r'\s+', ' ', author) for author in authors if author]
        
        # Parse categories
        categories = []
        if "categories" in raw_paper and raw_paper["categories"]:
            categories = raw_paper["categories"].split()
        
        # Extract year
        year = self._extract_year(raw_paper)
        
        # Generate PDF URL
        url_pdf = None
        if "id" in raw_paper:
            url_pdf = f"https://arxiv.org/pdf/{raw_paper['id']}.pdf"
        
        return Paper(
            id=raw_paper.get("id", ""),
            title=raw_paper.get("title", "").strip(),
            abstract=raw_paper.get("abstract", "").strip(),
            authors=authors,
            categories=categories,
            year=year,
            doi=raw_paper.get("doi"),
            url_pdf=url_pdf,
            journal_ref=raw_paper.get("journal-ref"),
            update_date=raw_paper.get("update_date")
        )
    
    def _extract_year(self, raw_paper: Dict[str, Any]) -> int:
        """Extract year from update_date or journal-ref"""
        # Try update_date first
        if "update_date" in raw_paper and raw_paper["update_date"]:
            try:
                return int(raw_paper["update_date"][:4])
            except (ValueError, TypeError):
                pass
        
        # Try journal-ref
        if "journal-ref" in raw_paper and raw_paper["journal-ref"]:
            # Look for 4-digit year in journal reference
            year_match = re.search(r'\b(19|20)\d{2}\b', raw_paper["journal-ref"])
            if year_match:
                return int(year_match.group())
        
        # Default to current year
        return datetime.now().year
    
    async def ingest_from_file(self, file_path: str, download_pdfs: bool = False, batch_size: int = 1000) -> IngestResponse:
        """Ingest papers from JSON file"""
        start_time = time.time()
        processed = 0
        errors = 0
        
        # Resolve file path
        if not Path(file_path).is_absolute():
            # Try relative paths
            for base_path in ["../data", "data", "../frontend/public/data"]:
                test_path = Path(base_path) / file_path
                if test_path.exists():
                    file_path = str(test_path)
                    break
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        papers_batch = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():  # Skip empty lines
                        raw_paper = json.loads(line)
                        paper = self.normalize_paper(raw_paper)
                        papers_batch.append(paper)
                        
                        # Process batch
                        if len(papers_batch) >= batch_size:
                            await self._process_batch(papers_batch, download_pdfs)
                            processed += len(papers_batch)
                            papers_batch = []
                            
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    errors += 1
                    continue
        
        # Process remaining papers
        if papers_batch:
            await self._process_batch(papers_batch, download_pdfs)
            processed += len(papers_batch)
        
        took_ms = int((time.time() - start_time) * 1000)
        
        return IngestResponse(
            message=f"Ingested {processed} papers with {errors} errors",
            processed=processed,
            errors=errors,
            took_ms=took_ms
        )
    
    async def _process_batch(self, papers: List[Paper], download_pdfs: bool = False):
        """Process a batch of papers"""
        # Store in search backend
        if self.es_service:
            await self.es_service.index_papers(papers)
        elif self.pg_service:
            await self.pg_service.upsert_papers(papers)
        
        # Download PDFs if requested and MinIO is available
        if download_pdfs and self.minio_service:
            pdf_tasks = [self.minio_service.download_and_store_pdf(paper) for paper in papers]
            await asyncio.gather(*pdf_tasks, return_exceptions=True)
