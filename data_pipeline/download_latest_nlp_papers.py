import os
import io 
import asyncio
import json
from logs import log
import pandas as pd 
from datetime import datetime
from .arxiv_nlp_pipeline import ArxivDataPipeline
from .arxiv_pdf_downloader import ArxivPDFDownloader
from dotenv import load_dotenv
from minio.error import S3Error
from typing import List, Dict, Optional


load_dotenv()

class NLPPaperDownloader:
    def __init__(self):
        self.pipeline = ArxivDataPipeline()
        self.pdf_downloader = ArxivPDFDownloader(
            max_concurrent=5,
            delay_between_requests=1.0
        )
        # Prefer MINIO_BUCKET then fallback to BUCKET_NAME
        self.bucket_name = str(os.getenv("MINIO_BUCKET") or os.getenv("BUCKET_NAME") or "papers")

    def dataset_exists_in_minio(self, object_name: str):
        """Load dataset from cache or download if not available"""
        # Check if we already have the dataset
        try:
            # pipeline.minio_client is MinioStorage wrapper -> underlying client at .minio_client
            self.pipeline.minio_client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name
            )
            return True
        except S3Error: 
            return False 
        except Exception as e: 
            log.warning(f"[DOWNLOAD] Unexpected stat error: {e}")
            return False

    def ensure_dataset_available(self, object_name: str): 
        if self.dataset_exists_in_minio(object_name):
            log.info(f"[DOWNLOAD] Found existing datatset object: {object_name}")
            return
        log.info("[DOWNLOAD] Dataset not found in MinIO. Downloading + uploading...")
        self.pipeline.download_dataset()
        if not self.dataset_exists_in_minio(object_name):
            raise RuntimeError("Dataset upload failed or object not found after download.")

    def load_dataset_from_minio(self, object_name: str, limit_lines: Optional[int] = None):
        """Stream a large JSONL (arxiv metadata) from MinIO safely without closing underlying prematurely.

        Uses chunk streaming to avoid ValueError: I/O operation on closed file.
        """
        log.info(f"[DOWNLOAD] Loading data from MinIO object: {object_name}")
        try:
            response = self.pipeline.minio_client.get_object(
                self.bucket_name, object_name
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to open dataset object: {e}")

        papers: List[Dict] = []
        bytes_buffer = b""
        line_count = 0
        chunk_size = 64 * 1024  # 64KB
        try:
            for chunk in response.stream(chunk_size):
                if not chunk:
                    continue
                bytes_buffer += chunk
                while b"\n" in bytes_buffer:
                    raw_line, bytes_buffer = bytes_buffer.split(b"\n", 1)
                    if limit_lines and line_count >= limit_lines:
                        break
                    line_count += 1
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        paper = json.loads(line.decode("utf-8"))
                        papers.append(paper)
                    except json.JSONDecodeError:
                        continue
                    if line_count % 100_000 == 0:
                        log.info(f"[DOWNLOAD] Streamed {line_count:,} lines ...")
                if limit_lines and line_count >= limit_lines:
                    break
            # Process tail (no newline at EOF)
            if (not limit_lines or line_count < limit_lines) and bytes_buffer.strip():
                try:
                    papers.append(json.loads(bytes_buffer.decode("utf-8")))
                except json.JSONDecodeError:
                    pass
        finally:
            response.close()
            response.release_conn()

        log.info(f"[DOWNLOAD] Loaded {len(papers):,} rows into memory (requested limit={limit_lines})")
        if not papers:
            return pd.DataFrame()
        df = pd.DataFrame(papers)
        # Defer expensive datetime conversion until after filtering to save memory.
        log.debug("[DOWNLOAD] DONE load_dataset_from_minio")
        return df
    
    # def save_json(self, data: Dict, name: str):
    #     """Persist a small JSON summary/analysis to local disk (logs dir)."""
    #     from pathlib import Path
    #     out_dir = Path("./data/logs")
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     path = out_dir / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    #     with path.open('w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False, indent=2)
    #     return path

    async def run(self, use_categories: bool = True, limit_dataset_lines: Optional[int] = None):
        self.ensure_dataset_available(object_name="arxiv-metadata-oai-snapshot.json")
        df = self.load_dataset_from_minio(object_name="arxiv-metadata-oai-snapshot.json", limit_lines=limit_dataset_lines)

        if df.empty:
            log.error("[DOWNLOAD] Empty dataset")
            return []

        filtered_df = self.pipeline.filter_nlp_papers_advanced(df, use_categories)
        log.info(f"[DOWNLOAD] Filtered: {len(filtered_df):,} papers")

        if filtered_df.empty:
            return []

        papers_to_download = [{"id": row.get("id"), "title": row.get("title"),
                              "authors": row.get("authors"), "abstract": row.get("abstract"),
                              "categories": row.get("categories")} for _, row in filtered_df.iterrows()]

        log.info(f"[DOWNLOAD] Downloading {len(papers_to_download)} PDFs...")
        results = await self.pdf_downloader.download_batch(papers_to_download)

        summary = {"timestamp": datetime.utcnow().isoformat() + "Z",
                   "requested": len(papers_to_download),
                   "success": sum(1 for r in results if r.get("status") == "success"),
                   "skipped": sum(1 for r in results if r.get("status") == "skipped"),
                   "failed": sum(1 for r in results if r.get("status") not in ("success", "skipped"))}
        log.info(f"[DOWNLOAD] Complete: {summary}")
        return summary


async def main():
    downloader = NLPPaperDownloader()
    results = await downloader.run(use_categories=True, limit_dataset_lines=None)
    log.info(f"[DONE] Total results: {results}")

if __name__ == "__main__":
    asyncio.run(main())