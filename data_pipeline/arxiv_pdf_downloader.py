
import os
import asyncio
import aiohttp
import json
import io
from logs import log
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
from tqdm import tqdm
from minio import Minio
from minio.error import S3Error

class ArxivPDFDownloader:
    def __init__(self, 
                 max_concurrent: int = 10,
                 retry_attempts: int = 3,
                 delay_between_requests: float = 0.5):
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.delay_between_requests = delay_between_requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_pdf_bytes = None  # type: int | None

        self.minio_client = Minio(**{"endpoint": str(os.getenv("MINIO_ENDPOINT")),
                                          "access_key": str(os.getenv("MINIO_ACCESS_KEY")),
                                          "secret_key": str(os.getenv("MINIO_SECRET_KEY")),
                                          "secure": False
                                        })
        self.bucket_name = str(os.getenv("MINIO_BUCKET"))


        # Track download statistics
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
    
    def extract_arxiv_id(self, paper_id: str) -> str:
        """Extract clean ArXiv ID from various formats"""
        # Remove version number if present (e.g., "1234.5678v2" -> "1234.5678")
        if 'v' in paper_id:
            paper_id = paper_id.split('v')[0]
        
        # Remove 'arXiv:' prefix if present
        if paper_id.startswith('arXiv:'):
            paper_id = paper_id[6:]
        
        return paper_id
    
    def get_pdf_url(self, paper_id: str) -> str:
        """Generate PDF URL from paper ID"""
        clean_id = self.extract_arxiv_id(paper_id)
        return f"https://arxiv.org/pdf/{clean_id}.pdf"
    
    def get_pdf_path(self, paper_id: str):
        """Get local path for PDF file"""
        clean_id = self.extract_arxiv_id(paper_id).replace('/', '_')
        return clean_id, f"{clean_id}.pdf"

    def _object_exists(self, object_name: str) -> bool:
        try:
            self.minio_client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket", "NotFound"):
                return False
            log.warning(f"[STORAGE] stat_object unexpected error {object_name}: {e}")
            return False
        except Exception as e:
            log.warning(f"[STORAGE] stat_object generic error {object_name}: {e}")
            return False

    async def download_pdf(self,
                           paper_id: str,
                           metadata: dict | None = None) -> dict:
        async with self.semaphore:
            clean_id, _ = self.get_pdf_path(paper_id)
            pdf_obj  = f"papers/{clean_id}/pdf/{clean_id}.pdf"
            meta_obj = f"papers/{clean_id}/metadata/{clean_id}.json"

            pdf_exists = self._object_exists(pdf_obj)
            meta_exists = self._object_exists(meta_obj) if metadata else True  # nếu không cần metadata coi như OK

            # Skip logic
            if pdf_exists and meta_exists:
                self.stats["skipped"] += 1
                return {
                    "paper_id": clean_id,
                    "status": "skipped",
                    "message": "Already exists",
                    "objects": [pdf_obj] + ([meta_obj] if metadata else [])
                }

            # Nếu PDF có rồi nhưng thiếu metadata => chỉ upload metadata (không cần re-download PDF)
            if pdf_exists and metadata and not meta_exists:
                # Direct upload metadata without presigned URL (simpler & avoids clock/DNS issues)
                try:
                    meta_doc = {
                        "paper_id": clean_id,
                        "title": metadata.get("title", ""),
                        "authors": metadata.get("authors", []),
                        "abstract": metadata.get("abstract", ""),
                        "categories": metadata.get("categories", ""),
                        "downloaded_at": datetime.utcnow().isoformat() + "Z",
                    }
                    meta_bytes = json.dumps(meta_doc, ensure_ascii=False).encode("utf-8")
                    self.minio_client.put_object(
                        bucket_name=self.bucket_name,
                        object_name=meta_obj,
                        data=io.BytesIO(meta_bytes),
                        length=len(meta_bytes),
                        content_type="application/json"
                    )
                except Exception as e:
                    self.stats["failed"] += 1
                    return {
                        "paper_id": clean_id,
                        "status": "error",
                        "message": f"Upload metadata failed: {e}"
                    }
                self.stats["successful"] += 1
                return {
                    "paper_id": clean_id,
                    "status": "success",
                    "objects": [pdf_obj, meta_obj],
                    "note": "Reused existing PDF; added metadata"
                }

            # Need to download PDF (pdf not present)
            pdf_url = self.get_pdf_url(clean_id)
            timeout = aiohttp.ClientTimeout(total=300)
            last_error = None
            for attempt in range(self.retry_attempts):
                pdf_buffer = bytearray()
                try:
                    await asyncio.sleep(self.delay_between_requests)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(pdf_url) as resp:
                            if resp.status == 404:
                                self.stats["failed"] += 1
                                return {
                                    "paper_id": clean_id,
                                    "status": "not_found",
                                    "message": "PDF 404"
                                }
                            resp.raise_for_status()
                            ctype = resp.headers.get("Content-Type", "")
                            if "pdf" not in ctype.lower():
                                log.warning(f"[DOWNLOAD] Unexpected content-type={ctype} for {clean_id}")

                            async for chunk in resp.content.iter_chunked(1 << 20):
                                if not chunk:
                                    continue
                                pdf_buffer.extend(chunk)
                                if self.max_pdf_bytes and len(pdf_buffer) > self.max_pdf_bytes:
                                    raise RuntimeError(
                                        f"PDF exceeds max size ({len(pdf_buffer)/1024/1024:.2f} MB)"
                                    )

                    # Upload PDF directly via MinIO client (blocking call) off event loop
                    def _put_pdf():
                        self.minio_client.put_object(
                            bucket_name=self.bucket_name,
                            object_name=pdf_obj,
                            data=io.BytesIO(pdf_buffer),
                            length=len(pdf_buffer),
                            content_type="application/pdf"
                        )
                    await asyncio.to_thread(_put_pdf)
                    log.debug(f"[PDF-UPLOAD] {clean_id} size={len(pdf_buffer)} bytes attempt={attempt}")

                    # Upload metadata if requested
                    if metadata:
                        meta_doc = {
                            "paper_id": clean_id,
                            "title": metadata.get("title", ""),
                            "authors": metadata.get("authors", []),
                            "abstract": metadata.get("abstract", ""),
                            "categories": metadata.get("categories", ""),
                            "downloaded_at": datetime.utcnow().isoformat() + "Z",
                            "size_bytes": len(pdf_buffer)
                        }
                        meta_bytes = json.dumps(meta_doc, ensure_ascii=False).encode("utf-8")
                        def _put_meta():
                            self.minio_client.put_object(
                                bucket_name=self.bucket_name,
                                object_name=meta_obj,
                                data=io.BytesIO(meta_bytes),
                                length=len(meta_bytes),
                                content_type="application/json"
                            )
                        await asyncio.to_thread(_put_meta)
                        log.debug(f"[META-UPLOAD] {clean_id} meta_size={len(meta_bytes)}")

                    self.stats["successful"] += 1
                    return {
                        "paper_id": clean_id,
                        "status": "success",
                        "objects": [pdf_obj] + ([meta_obj] if metadata else []),
                        "bytes": len(pdf_buffer),
                        "retries_used": attempt
                    }
                except asyncio.TimeoutError as e:
                    last_error = f"timeout: {e}"
                    if attempt < self.retry_attempts - 1:
                        continue
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.retry_attempts - 1:
                        continue

            self.stats["failed"] += 1
            return {
                "paper_id": clean_id,
                "status": "error",
                "message": last_error or "Unknown error after retries"
            }


    async def download_batch(self, papers: List[Dict]) -> List[Dict]:
        tasks: List[asyncio.Task] = []
        results: List[Dict] = []
        for paper in papers:
            pid = paper.get("id") or paper.get("paperId") or paper.get("paper_id")
            if not pid:
                self.stats["skipped"] += 1
                results.append({"paper_id": None, "status": "skipped", "message": "Missing id"})
                continue
            tasks.append(asyncio.create_task(self.download_pdf(pid, paper)))

        total = len(tasks)
        self.stats["total"] += total
        if total == 0:
            return results

        for fut in tqdm(asyncio.as_completed(tasks), total=total, unit="paper", desc="Downloading PDFs"):
            try:
                res = await fut
            except Exception as e:
                res = {"paper_id": None, "status": "error", "message": str(e)}
                self.stats["failed"] += 1
            results.append(res)
            done = len(results)
            if done % 100 == 0 or done == total:
                log.info(f"[DOWNLOAD] Progress {done}/{total} "
                         f"success={self.stats['successful']} failed={self.stats['failed']} skipped={self.stats['skipped']}")

        return results