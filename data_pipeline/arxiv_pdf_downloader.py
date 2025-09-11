#!/usr/bin/env python3
"""
ArXiv PDF Downloader with async support
Downloads PDFs from ArXiv based on paper IDs
"""

import os
import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
from tqdm.asyncio import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArxivPDFDownloader:
    def __init__(self, 
                 pdfs_dir: str = "../data/pdfs",
                 max_concurrent: int = 10,
                 retry_attempts: int = 3,
                 delay_between_requests: float = 0.5):
        """
        Initialize PDF downloader
        
        Args:
            pdfs_dir: Directory to save PDFs
            max_concurrent: Maximum concurrent downloads
            retry_attempts: Number of retry attempts for failed downloads
            delay_between_requests: Delay between requests to avoid rate limiting
        """
        self.pdfs_dir = Path(pdfs_dir)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.delay_between_requests = delay_between_requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
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
    
    def get_pdf_path(self, paper_id: str) -> Path:
        """Get local path for PDF file"""
        clean_id = self.extract_arxiv_id(paper_id).replace('/', '_')
        return self.pdfs_dir / f"{clean_id}.pdf"
    
    async def download_pdf(self, 
                          session: aiohttp.ClientSession, 
                          paper_id: str,
                          metadata: Optional[Dict] = None) -> Dict:
        """
        Download a single PDF
        
        Args:
            session: aiohttp session
            paper_id: ArXiv paper ID
            metadata: Optional metadata about the paper
            
        Returns:
            Dict with download result
        """
        async with self.semaphore:
            pdf_path = self.get_pdf_path(paper_id)
            
            # Skip if already downloaded
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                self.stats["skipped"] += 1
                return {
                    "paper_id": paper_id,
                    "status": "skipped",
                    "path": str(pdf_path),
                    "message": "Already downloaded"
                }
            
            pdf_url = self.get_pdf_url(paper_id)
            
            for attempt in range(self.retry_attempts):
                try:
                    # Add delay to avoid rate limiting
                    await asyncio.sleep(self.delay_between_requests)
                    
                    async with session.get(pdf_url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.read()
                            
                            # Save PDF
                            with open(pdf_path, 'wb') as f:
                                f.write(content)
                            
                            self.stats["successful"] += 1
                            
                            # Save metadata if provided
                            if metadata:
                                meta_path = pdf_path.with_suffix('.json')
                                with open(meta_path, 'w') as f:
                                    json.dump({
                                        "paper_id": paper_id,
                                        "title": metadata.get("title", ""),
                                        "authors": metadata.get("authors", []),
                                        "abstract": metadata.get("abstract", ""),
                                        "categories": metadata.get("categories", ""),
                                        "downloaded_at": datetime.now().isoformat(),
                                        "pdf_size": len(content)
                                    }, f, indent=2)
                            
                            return {
                                "paper_id": paper_id,
                                "status": "success",
                                "path": str(pdf_path),
                                "size": len(content)
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
            for coro in tqdm.as_completed(tasks, desc="Downloading PDFs"):
                result = await coro
                results.append(result)
                
                # Log progress periodically
                if len(results) % 100 == 0:
                    logger.info(f"Progress: {len(results)}/{len(tasks)} - "
                              f"Success: {self.stats['successful']}, "
                              f"Failed: {self.stats['failed']}, "
                              f"Skipped: {self.stats['skipped']}")
            
            return results
    
    def print_statistics(self):
        """Print download statistics"""
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD STATISTICS:")
        logger.info("="*60)
        logger.info(f"Total papers: {self.stats['total']}")
        logger.info(f"Successfully downloaded: {self.stats['successful']}")
        logger.info(f"Failed downloads: {self.stats['failed']}")
        logger.info(f"Skipped (already exists): {self.stats['skipped']}")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
    
    def save_download_log(self, results: List[Dict], log_file: str = "../data/logs/download_log.json"):
        """Save download results to log file"""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "statistics": self.stats,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Download log saved to: {log_path}")


async def download_sample_papers():
    """Download a sample of papers for testing"""
    # Sample paper IDs from different categories
    sample_papers = [
        {"id": "2301.00234", "title": "Sample NLP Paper 1", "categories": "cs.CL"},
        {"id": "2312.15678", "title": "Sample ML Paper", "categories": "cs.LG cs.AI"},
        {"id": "2401.12345", "title": "Sample IR Paper", "categories": "cs.IR"},
        {"id": "1706.03762", "title": "Attention Is All You Need", "categories": "cs.CL cs.LG"},  # Transformer paper
        {"id": "1810.04805", "title": "BERT", "categories": "cs.CL"},  # BERT paper
    ]
    
    downloader = ArxivPDFDownloader(max_concurrent=3)
    results = await downloader.download_batch(sample_papers)
    
    downloader.print_statistics()
    downloader.save_download_log(results)
    
    return results


def main():
    """Main execution function"""
    logger.info("Starting ArXiv PDF Downloader...")
    
    # Run sample download
    results = asyncio.run(download_sample_papers())
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD RESULTS:")
    logger.info("="*60)
    for result in results:
        if result['status'] == 'success':
            logger.info(f"✓ {result['paper_id']}: Downloaded to {result['path']}")
        elif result['status'] == 'skipped':
            logger.info(f"⊙ {result['paper_id']}: {result['message']}")
        else:
            logger.info(f"✗ {result['paper_id']}: {result.get('message', 'Failed')}")


if __name__ == "__main__":
    main()