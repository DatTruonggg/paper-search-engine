import asyncio
import aiohttp
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from tqdm import tqdm
from logs import log 
from io import BytesIO
from config import cfg
from minio import Minio

class ArxivPDFDownloader:
    def __init__(self,
                config,
                bucket_name: str = "paper_engine",
                raw_prefix: str = "raw",
                json_metadata_prefix: str = "json_bucket",
                max_concurrent: int = 10,
                retry_attempts: int = 3,
                delay_between_requests: float = 0.5):
        """
        Initialize the PDF downloader.

        Args:
            config: Holds MinIO endpoint, access key, secret key, secure flag
            bucket_name: MinIO bucket name to store artifacts
            raw_prefix: Prefix within bucket for PDF objects
            json_metadata_prefix: Prefix within bucket for JSON metadata
            max_concurrent: Maximum concurrent downloads
            retry_attempts: Number of retry attempts for failed downloads
            delay_between_requests: Delay between requests to avoid rate limiting
        """
        # Paths and concurrency
        self.bucket_name = bucket_name
        self.raw_prefix = raw_prefix
        self.json_metadata_prefix = json_metadata_prefix
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.delay_between_requests = delay_between_requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.config = config

        # Track download statistics
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }

        self.minio_client = Minio(
                            endpoint=self.config.minio_client.endpoint,
                            access_key=self.config.minio_client.access_key,
                            secret_key=self.config.minio_client.secret_key,
                            secure=bool(self.config.minio_client.secure)
                            )
        # Ensure target bucket exists
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
        except Exception as e:
            log.warning(f"Unable to ensure bucket '{self.bucket_name}': {e}")

    def extract_arxiv_id(self, paper_id: str) -> str:
        """Extract a canonical ArXiv ID (drop version, prefix)."""
        # Remove version number if present (e.g., "1234.5678v2" -> "1234.5678")
        if 'v' in paper_id:
            paper_id = paper_id.split('v')[0]
        
        # Remove 'arXiv:' prefix if present
        if paper_id.startswith('arXiv:'):
            paper_id = paper_id[6:]
        
        return paper_id
    
    def get_pdf_url(self, paper_id: str) -> str:
        """Generate the PDF URL from a paper ID."""
        clean_id = self.extract_arxiv_id(paper_id)
        return f"https://arxiv.org/pdf/{clean_id}.pdf"
    
    def _pdf_object_name(self, paper_id: str) -> str:
        """MinIO object name for the PDF under raw prefix."""
        clean_id = self.extract_arxiv_id(paper_id).replace('/', '_')
        prefix = self.raw_prefix.strip('/') if self.raw_prefix else ""
        return f"{prefix}/{clean_id}.pdf" if prefix else f"{clean_id}.pdf"

    def _json_object_name(self, paper_id: str) -> str:
        """MinIO object name for the JSON metadata under json_metadata_prefix."""
        clean_id = self.extract_arxiv_id(paper_id).replace('/', '_')
        prefix = self.json_metadata_prefix.strip('/') if self.json_metadata_prefix else ""
        return f"{prefix}/{clean_id}.json" if prefix else f"{clean_id}.json"

    # No local file paths are used; all data is streamed directly to MinIO

    async def download_pdf(self, 
                          session: aiohttp.ClientSession, 
                          paper_id: str,
                          metadata: Optional[Dict] = None) -> Dict: #TODO: check it for me, download and dump to minio bucket with server IP
        """
        Download a single PDF and optionally upload to MinIO.

        Args:
            session: aiohttp session
            paper_id: ArXiv paper ID
            metadata: Optional metadata about the paper

        Returns:
            Dict describing the result
        """
        async with self.semaphore:
            # Skip if already present in MinIO with reasonable size
            try:
                stat = self.minio_client.stat_object(self.bucket_name, self._pdf_object_name(paper_id))
                if getattr(stat, 'size', 0) and stat.size > 1024:
                    self.stats["skipped"] += 1
                    return {
                        "paper_id": paper_id,
                        "status": "skipped",
                        "minio_object": self._pdf_object_name(paper_id),
                        "message": "Already uploaded"
                    }
            except Exception:
                pass
            
            pdf_url = self.get_pdf_url(paper_id)
            
            for attempt in range(self.retry_attempts):
                try:
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(self.delay_between_requests)
                    
                    async with session.get(pdf_url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.read()
                            pdf_object = self._pdf_object_name(paper_id)
                            self.minio_client.put_object(
                                self.bucket_name,
                                pdf_object,
                                BytesIO(content),
                                len(content),
                                content_type="application/pdf",
                            )
                            log.info(f"[MINIO STORAGE] Uploaded PDF: {self.bucket_name}/{pdf_object}")

                            self.stats["successful"] += 1
                            
                            # Save metadata to MinIO if provided
                            json_object = None
                            if metadata:
                                payload = {
                                    "paper_id": paper_id,
                                    "title": metadata.get("title", ""),
                                    "authors": metadata.get("authors", []),
                                    "abstract": metadata.get("abstract", ""),
                                    "categories": metadata.get("categories", ""),
                                    "downloaded_at": datetime.now().isoformat(),
                                    "pdf_size": len(content),
                                }
                                json_bytes = json.dumps(payload, indent=2).encode("utf-8")
                                json_object = self._json_object_name(paper_id)
                                self.minio_client.put_object(
                                    self.bucket_name,
                                    json_object,
                                    BytesIO(json_bytes),
                                    len(json_bytes),
                                    content_type="application/json",
                                )
                                log.info(f"[MINIO STORAGE] Uploaded JSON: {self.bucket_name}/{json_object}")
                            
                            return {
                                "paper_id": paper_id,
                                "status": "success",
                                "minio_pdf_object": pdf_object,
                                "minio_json_object": json_object,
                                "size": len(content),
                            }
                        
                        elif response.status == 404:
                            self.stats["failed"] += 1
                            return {
                                "paper_id": paper_id,
                                "status": "not_found",
                                "message": f"PDF not found (404)"
                            }
                        
                        else:
                            if attempt < self.retry_attempts - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            
                            self.stats["failed"] += 1
                            return {
                                "paper_id": paper_id,
                                "status": "error",
                                "message": f"HTTP {response.status}"
                            }
                
                except asyncio.TimeoutError:
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    self.stats["failed"] += 1
                    return {
                        "paper_id": paper_id,
                        "status": "timeout",
                        "message": "Download timeout"
                    }
                
                except Exception as e:
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    self.stats["failed"] += 1
                    return {
                        "paper_id": paper_id,
                        "status": "error",
                        "message": str(e)
                    }
    
    async def download_batch(self, papers: List[Dict]) -> List[Dict]:
        """
        Download a batch of papers
        
        Args:
            papers: List of paper dictionaries with at least 'id' field
            
        Returns:
            List of download results
        """
        self.stats["total"] = len(papers)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for paper in papers:
                paper_id = paper.get('id', paper.get('paperId', ''))
                if paper_id:
                    task = self.download_pdf(session, paper_id, paper)
                    tasks.append(task)
            
            # Use tqdm for progress bar
            results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Downloading PDFs"):
                result = await coro
                results.append(result)
                
                # Log progress periodically
                if len(results) % 100 == 0:
                    log.info(f"Progress: {len(results)}/{len(tasks)} - "
                              f"Success: {self.stats['successful']}, "
                              f"Failed: {self.stats['failed']}, "
                              f"Skipped: {self.stats['skipped']}")
            
            return results