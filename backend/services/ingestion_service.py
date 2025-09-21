"""
Ingestion service for processing papers through the complete pipeline.
Integrates with existing data pipeline components and follows MinIO bucket structure:

papers/
  {paperId}/
    pdf/{paperId}.pdf
    metadata/{paperId}.json
    markdown/index.md
    images/
      fig_p{page}_{idx}_{sha16}.{ext}
    manifest.json
"""

import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re
import base64
from datetime import datetime

from backend.config import config
from backend.services import ElasticsearchSearchService
from data_pipeline.minio_storage import MinIOStorage
from data_pipeline.download_latest_nlp_papers import NLPPaperDownloader
from data_pipeline.docling_pdf_parser import run
from data_pipeline.ingest_papers import PaperProcessor
from logs import log

class IngestionService:
    """
    Service for managing the complete paper ingestion pipeline.
    
    Handles:
    1. ArXiv metadata download and filtering
    2. PDF downloading from ArXiv
    3. PDF parsing with Docling
    4. Image upload to MinIO with structured bucket layout
    5. Document chunking and embedding
    6. Elasticsearch indexing
    
    MinIO Bucket Structure:
    papers/
      {paperId}/
        pdf/{paperId}.pdf
        metadata/{paperId}.json
        markdown/index.md
        images/
          fig_p{page}_{idx}_{sha16}.{ext}
        manifest.json
    """
    
    def __init__(self):
        """Initialize ingestion service"""
        self.es_service = ElasticsearchSearchService(
            es_host=config.ES_HOST,
            index_name=config.ES_INDEX_NAME,
            bge_model=config.BGE_MODEL_NAME,
            bge_cache_dir=config.BGE_CACHE_DIR
        )
        
        log.info("Ingestion service initialized")
    
    async def process_arxiv_papers(
        self,
        num_papers: int = 100,
        categories: List[str] = None,
        use_keywords: bool = True,
        min_keyword_matches: int = 1
    ) -> Dict[str, Any]:
        """
        Process ArXiv papers through the complete pipeline.
        
        Args:
            num_papers: Number of papers to process
            categories: List of ArXiv categories to filter by
            use_keywords: Whether to use keyword filtering
            min_keyword_matches: Minimum keyword matches required
            
        Returns:
            Dictionary with processing results
        """
        try:
            log.info(f"Starting ArXiv paper processing pipeline for {num_papers} papers")
            
            # Step 1: Download and filter ArXiv metadata
            log.info("Step 1: Downloading and filtering ArXiv metadata...")
            
            
            downloader = NLPPaperDownloader()
            
            # Load or download dataset
            if not downloader.load_or_download_dataset():
                raise Exception("Failed to load ArXiv dataset")
            
            # Process papers
            results = await downloader.find_and_download_nlp_papers(
                num_papers=num_papers,
                use_categories=use_keywords,
                use_keywords=use_keywords,
                min_keyword_matches=min_keyword_matches
            )
            
            if not results:
                raise Exception("No papers found or downloaded")
            
            successful_downloads = [r for r in results if r.get('status') == 'success']
            log.info(f"Downloaded {len(successful_downloads)} papers successfully")
            
            # Step 2: Parse PDFs with Docling
            log.info("Step 2: Parsing PDFs with Docling...")
            
            
            pdf_dir = downloader.pipeline.pdfs_dir
            markdown_dir = downloader.pipeline.processed_dir / "markdown"
            
            # Run Docling parser
            parse_stats = run(
                input_dir=pdf_dir,
                output_dir=markdown_dir,
                pattern="*.pdf",
                overwrite=False
            )
            
            log.info(f"Parsed {parse_stats['converted']} PDFs to markdown")
            
            # Step 3: Upload images to MinIO and process markdown
            log.info("Step 3: Processing images and uploading to MinIO...")
            image_urls_by_paper = await self._process_images_and_upload_to_minio(markdown_dir)

            # Step 3b: Create complete bucket structure and collect URLs per paper
            storage_papers: List[Dict[str, Any]] = []
            markdown_files = list(markdown_dir.glob("*.md"))
            for md_file in markdown_files:
                try:
                    paper_id = md_file.stem
                    pdf_path = pdf_dir / f"{paper_id}.pdf"
                    markdown_content = md_file.read_text(encoding='utf-8')
                    # Load metadata if available
                    metadata: Dict[str, Any] = {}
                    json_path = pdf_dir / f"{paper_id}.json"
                    if json_path.exists():
                        try:
                            metadata = json.loads(json_path.read_text(encoding='utf-8'))
                        except Exception:
                            metadata = {}

                    urls = await self._create_paper_bucket_structure(
                        paper_id=paper_id,
                        pdf_path=pdf_path,
                        metadata=metadata,
                        markdown_content=markdown_content,
                        image_urls=image_urls_by_paper.get(paper_id, [])
                    )
                    storage_papers.append({
                        "paper_id": paper_id,
                        "urls": urls
                    })
                except Exception as e:
                    log.warning(f"Failed to create bucket structure for {md_file}: {e}")
            
            # Step 4: Ingest into Elasticsearch
            log.info("Step 4: Ingesting papers into Elasticsearch...")
            
            
            processor = PaperProcessor(
                es_host=config.ES_HOST,
                bge_model=config.BGE_MODEL_NAME,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
                json_metadata_dir=str(pdf_dir),
                minio_endpoint=config.MINIO_ENDPOINT,
                enable_minio=True
            )
            
            # Ingest all markdown files
            processor.ingest_directory(
                markdown_dir=markdown_dir,
                batch_size=10,
                max_files=None
            )
            
            # Get final statistics
            es_stats = self.es_service.get_index_stats()
            
            return {
                "status": "success",
                "pipeline_steps": {
                    "arxiv_download": {
                        "total_requested": num_papers,
                        "successful_downloads": len(successful_downloads),
                        "failed_downloads": len(results) - len(successful_downloads)
                    },
                    "pdf_parsing": {
                        "total_pdfs": parse_stats['total'],
                        "converted": parse_stats['converted'],
                        "skipped": parse_stats['skipped'],
                        "failed": parse_stats['failed']
                    },
                    "elasticsearch_indexing": {
                        "total_documents": es_stats.get('document_count', 0),
                        "index_size_mb": es_stats.get('index_size_mb', 0)
                    }
                },
                "storage": {
                    "papers": storage_papers,
                    "total_papers_with_assets": len(storage_papers)
                },
                "processing_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            log.error(f"Error in ArXiv paper processing pipeline: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": datetime.now().isoformat()
            }
    
    async def _process_images_and_upload_to_minio(self, markdown_dir: Path) -> Dict[str, List[str]]:
        """
        Process images in markdown files and upload to MinIO following bucket structure.
        
        MinIO Structure:
        papers/{paperId}/images/fig_p{page}_{idx}_{sha16}.{ext}
        
        Args:
            markdown_dir: Directory containing markdown files
        """
        try:
            image_urls_by_paper: Dict[str, List[str]] = {}
            minio_storage = MinIOStorage(endpoint=config.MINIO_ENDPOINT)
            
            markdown_files = list(markdown_dir.glob("*.md"))
            processed_count = 0
            
            for md_file in markdown_files:
                try:
                    paper_id = md_file.stem
                    content = md_file.read_text(encoding='utf-8')
                    image_urls_by_paper.setdefault(paper_id, [])
                    
                    # Pattern to match base64 images
                    base64_pattern = r'!\[([^\]]*)\]\(data:image/([^;]+);base64,([^)]+)\)'
                    
                    def replace_base64_image(match):
                        alt_text = match.group(1)
                        image_format = match.group(2)
                        base64_data = match.group(3)
                        
                        try:
                            # Decode base64 image
                            image_data = base64.b64decode(base64_data)
                            
                            # Generate SHA16 hash for unique filename
                            image_hash = hashlib.sha256(image_data).hexdigest()[:16]
                            
                            # Extract page number from alt text if available
                            page_match = re.search(r'page[_\s]*(\d+)', alt_text.lower())
                            page_num = page_match.group(1) if page_match else "1"
                            
                            # Generate structured filename: fig_p{page}_{idx}_{sha16}.{ext}
                            image_filename = f"fig_p{page_num}_{processed_count}_{image_hash}.{image_format}"
                            
                            # MinIO object path: papers/{paperId}/images/{filename}
                            minio_object_path = f"{paper_id}/images/{image_filename}"
                            
                            # Upload to MinIO
                            
                            temp_image_path = Path(f"/tmp/{image_filename}")
                            temp_image_path.write_bytes(image_data)
                            
                            minio_url = minio_storage.upload_file(
                                bucket_name="papers",
                                object_name=minio_object_path,
                                file_path=temp_image_path
                            )
                            
                            # Clean up temp file
                            temp_image_path.unlink()
                            
                            if minio_url:
                                image_urls_by_paper[paper_id].append(minio_url)
                            # Return markdown image link
                            return f"![{alt_text}]({minio_url})"
                            
                        except Exception as e:
                            log.warning(f"Failed to process image in {md_file}: {e}")
                            return match.group(0)  # Return original if processing fails
                    
                    # Replace all base64 images
                    updated_content = re.sub(base64_pattern, replace_base64_image, content)
                    
                    # Write updated content back
                    if updated_content != content:
                        md_file.write_text(updated_content, encoding='utf-8')
                        processed_count += 1
                        log.info(f"Processed images in {md_file.name}")
                
                except Exception as e:
                    log.warning(f"Error processing images in {md_file}: {e}")
                    continue
            
            log.info(f"Processed images in {processed_count} markdown files")
            return image_urls_by_paper
            
        except Exception as e:
            log.error(f"Error in image processing: {e}")
            raise
    
    async def _create_paper_bucket_structure(self, paper_id: str, pdf_path: Path, metadata: Dict, markdown_content: str, image_urls: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Create complete MinIO bucket structure for a paper.
        
        Structure:
        papers/{paperId}/
          pdf/{paperId}.pdf
          metadata/{paperId}.json
          markdown/index.md
          images/ (handled separately)
          manifest.json
        
        Args:
            paper_id: Paper identifier
            pdf_path: Path to PDF file
            metadata: Paper metadata
            markdown_content: Processed markdown content
            
        Returns:
            Dictionary with MinIO URLs for each component
        """
        try:
            
            
            minio_storage = MinIOStorage(endpoint=config.MINIO_ENDPOINT)
            urls = {}
            
            # 1. Upload PDF: papers/{paperId}/pdf/{paperId}.pdf
            pdf_object_path = f"{paper_id}/pdf/{paper_id}.pdf"
            pdf_url = minio_storage.upload_file(
                bucket_name="papers",
                object_name=pdf_object_path,
                file_path=pdf_path
            )
            urls["pdf"] = pdf_url
            
            # 2. Upload metadata: papers/{paperId}/metadata/{paperId}.json
            metadata_object_path = f"{paper_id}/metadata/{paper_id}.json"
            metadata_json = json.dumps(metadata, indent=2)
            
            # Create temp file for metadata
            temp_metadata_path = Path(f"/tmp/{paper_id}_metadata.json")
            temp_metadata_path.write_text(metadata_json, encoding='utf-8')
            
            metadata_url = minio_storage.upload_file(
                bucket_name="papers",
                object_name=metadata_object_path,
                file_path=temp_metadata_path
            )
            urls["metadata"] = metadata_url
            temp_metadata_path.unlink()
            
            # 3. Upload markdown: papers/{paperId}/markdown/index.md
            markdown_object_path = f"{paper_id}/markdown/index.md"
            
            # Create temp file for markdown
            temp_markdown_path = Path(f"/tmp/{paper_id}_index.md")
            temp_markdown_path.write_text(markdown_content, encoding='utf-8')
            
            markdown_url = minio_storage.upload_file(
                bucket_name="papers",
                object_name=markdown_object_path,
                file_path=temp_markdown_path
            )
            urls["markdown"] = markdown_url
            temp_markdown_path.unlink()
            
            # 4. Attach images into urls
            urls["images"] = image_urls or []

            # 5. Create manifest: papers/{paperId}/manifest.json
            manifest = {
                "paper_id": paper_id,
                "created_at": datetime.now().isoformat(),
                "structure": {
                    "pdf": f"{paper_id}/pdf/{paper_id}.pdf",
                    "metadata": f"{paper_id}/metadata/{paper_id}.json",
                    "markdown": f"{paper_id}/markdown/index.md",
                    "images": f"{paper_id}/images/"
                },
                "urls": urls,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", []),
                    "categories": metadata.get("categories", []),
                    "publish_date": metadata.get("publish_date"),
                    "word_count": len(markdown_content.split()),
                    "has_images": "![image]" in markdown_content
                }
            }
            
            manifest_object_path = f"{paper_id}/manifest.json"
            manifest_json = json.dumps(manifest, indent=2)
            
            # Create temp file for manifest
            temp_manifest_path = Path(f"/tmp/{paper_id}_manifest.json")
            temp_manifest_path.write_text(manifest_json, encoding='utf-8')
            
            manifest_url = minio_storage.upload_file(
                bucket_name="papers",
                object_name=manifest_object_path,
                file_path=temp_manifest_path
            )
            urls["manifest"] = manifest_url
            temp_manifest_path.unlink()
            
            log.info(f"Created complete bucket structure for paper {paper_id}")
            return urls
            
        except Exception as e:
            log.error(f"Error creating bucket structure for {paper_id}: {e}")
            raise
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current status of the ingestion pipeline.
        
        Returns:
            Dictionary with ingestion status information
        """
        try:
            # Get Elasticsearch stats
            es_stats = self.es_service.get_index_stats()
            
            # Get MinIO stats
            try:
                
                minio_storage = MinIOStorage(endpoint=config.MINIO_ENDPOINT)
                minio_stats = minio_storage.get_storage_stats()
            except Exception as e:
                minio_stats = {"error": str(e)}
            
            return {
                "status": "healthy",
                "elasticsearch": {
                    "document_count": es_stats.get('document_count', 0),
                    "index_size_mb": es_stats.get('index_size_mb', 0),
                    "cluster_health": "unknown"  # Would need to implement cluster health check
                },
                "minio": minio_stats,
                "configuration": {
                    "es_host": config.ES_HOST,
                    "minio_endpoint": config.MINIO_ENDPOINT,
                    "bge_model": config.BGE_MODEL_NAME,
                    "chunk_size": config.CHUNK_SIZE,
                    "chunk_overlap": config.CHUNK_OVERLAP
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log.error(f"Error getting ingestion status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for ingestion service.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Check Elasticsearch
            es_health = self.es_service.health_check()
            
            # Check MinIO
            minio_health = {"status": "unknown"}
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    minio_url = f"{config.MINIO_ENDPOINT}/minio/health/live"
                    if not minio_url.startswith(('http://', 'https://')):
                        minio_url = f"http://{minio_url}"
                    
                    async with session.get(minio_url, timeout=5) as response:
                        if response.status == 200:
                            minio_health = {"status": "healthy"}
                        else:
                            minio_health = {"status": "unhealthy", "error": f"HTTP {response.status}"}
            except Exception as e:
                minio_health = {"status": "unhealthy", "error": str(e)}
            
            # Overall status
            overall_status = "healthy"
            if es_health["status"] != "healthy" or minio_health["status"] != "healthy":
                overall_status = "degraded"
            
            return {
                "status": overall_status,
                "components": {
                    "elasticsearch": es_health,
                    "minio": minio_health
                }
            }
            
        except Exception as e:
            log.error(f"Error in ingestion service health check: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
