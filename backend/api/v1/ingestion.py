"""
Ingestion API endpoints for managing the paper processing pipeline.
"""

from logs import log
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.services.ingestion_service import IngestionService


# Create router for ingestion endpoints
router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion Pipeline"])

# Global ingestion service instance
ingestion_service: Optional[IngestionService] = None


# Pydantic models for API
class ArxivProcessingRequest(BaseModel):
    """Request model for ArXiv paper processing"""
    num_papers: int = Field(default=100, ge=1, le=2000, description="Number of papers to process")
    categories: Optional[List[str]] = Field(default=None, description="ArXiv categories to filter by")
    use_keywords: bool = Field(default=True, description="Whether to use keyword filtering")
    min_keyword_matches: int = Field(default=1, ge=1, le=5, description="Minimum keyword matches required")


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion status"""
    status: str = Field(description="Overall status")
    elasticsearch: Dict[str, Any] = Field(description="Elasticsearch statistics")
    minio: Dict[str, Any] = Field(description="MinIO statistics")
    configuration: Dict[str, Any] = Field(description="Configuration information")
    timestamp: str = Field(description="Status timestamp")


class ProcessingResultResponse(BaseModel):
    """Response model for processing results"""
    status: str = Field(description="Processing status")
    pipeline_steps: Dict[str, Any] = Field(description="Results from each pipeline step")
    processing_time: str = Field(description="Processing completion time")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")


def get_ingestion_service() -> IngestionService:
    """Dependency to get ingestion service"""
    global ingestion_service
    if ingestion_service is None:
        try:
            ingestion_service = IngestionService()
            log.info("Ingestion service initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize ingestion service: {e}")
            raise HTTPException(status_code=503, detail="Ingestion service unavailable")
    
    return ingestion_service


@router.get("/status", response_model=IngestionStatusResponse)
async def get_ingestion_status():
    """
    Get current status of the ingestion pipeline.
    
    Returns information about:
    - Elasticsearch index statistics
    - MinIO storage statistics
    - Configuration settings
    - Overall system health
    """
    try:
        service = get_ingestion_service()
        status = await service.get_ingestion_status()
        
        return IngestionStatusResponse(
            status=status["status"],
            elasticsearch=status["elasticsearch"],
            minio=status["minio"],
            configuration=status["configuration"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        log.error(f"Error getting ingestion status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def ingestion_health_check():
    """
    Health check for ingestion service components.
    
    Checks the health of:
    - Elasticsearch cluster and index
    - MinIO object storage
    - Overall ingestion pipeline readiness
    """
    try:
        service = get_ingestion_service()
        health = await service.health_check()
        
        if health["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in ingestion health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-arxiv", response_model=ProcessingResultResponse)
async def process_arxiv_papers(
    request: ArxivProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Process ArXiv papers through the complete ingestion pipeline.
    
    This endpoint triggers the full pipeline:
    1. Download and filter ArXiv metadata
    2. Download PDFs from ArXiv
    3. Parse PDFs with Docling to extract text and images
    4. Upload images to MinIO and replace base64 with URLs
    5. Chunk documents and generate embeddings
    6. Index everything into Elasticsearch
    
    Note: This is a long-running operation that may take several minutes.
    """
    try:
        log.info(f"Starting ArXiv paper processing: {request.num_papers} papers")
        
        service = get_ingestion_service()
        
        # Process papers (this will run in the background)
        result = await service.process_arxiv_papers(
            num_papers=request.num_papers,
            categories=request.categories,
            use_keywords=request.use_keywords,
            min_keyword_matches=request.min_keyword_matches
        )
        
        return ProcessingResultResponse(
            status=result["status"],
            pipeline_steps=result.get("pipeline_steps", {}),
            processing_time=result["processing_time"],
            error=result.get("error")
        )
        
    except Exception as e:
        log.exception("ArXiv paper processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-arxiv-async", response_model=Dict[str, str])
async def process_arxiv_papers_async(
    request: ArxivProcessingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start ArXiv paper processing as a background task.
    
    This endpoint starts the processing pipeline in the background and returns immediately.
    Use the status endpoint to monitor progress.
    """
    try:
        log.info(f"Starting async ArXiv paper processing: {request.num_papers} papers")
        
        service = get_ingestion_service()
        
        # Add background task
        background_tasks.add_task(
            service.process_arxiv_papers,
            num_papers=request.num_papers,
            categories=request.categories,
            use_keywords=request.use_keywords,
            min_keyword_matches=request.min_keyword_matches
        )
        
        return {
            "status": "started",
            "message": f"ArXiv paper processing started for {request.num_papers} papers",
            "task_id": f"arxiv_processing_{request.num_papers}_{request.use_keywords}"
        }
        
    except Exception as e:
        log.exception("Failed to start async ArXiv processing")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline-steps")
async def get_pipeline_steps():
    """
    Get information about the ingestion pipeline steps.
    
    Returns detailed information about each step in the pipeline
    and what it does.
    """
    return {
        "pipeline_steps": [
            {
                "step": 1,
                "name": "ArXiv Metadata Download",
                "description": "Download and filter ArXiv metadata from Kaggle dataset",
                "components": ["Kaggle API", "ArXiv metadata filtering", "NLP keyword matching"],
                "output": "Filtered list of relevant papers"
            },
            {
                "step": 2,
                "name": "PDF Download",
                "description": "Download PDF files from ArXiv for selected papers",
                "components": ["ArXiv PDF API", "Async downloader", "Rate limiting"],
                "output": "PDF files stored locally"
            },
            {
                "step": 3,
                "name": "PDF Parsing",
                "description": "Parse PDFs using Docling to extract text and images",
                "components": ["Docling parser", "Markdown conversion", "Image extraction"],
                "output": "Markdown files with base64 images"
            },
            {
                "step": 4,
                "name": "Image Processing",
                "description": "Upload images to MinIO and replace base64 with URLs",
                "components": ["MinIO storage", "Image upload", "URL replacement"],
                "output": "Markdown files with MinIO image URLs"
            },
            {
                "step": 5,
                "name": "Document Chunking",
                "description": "Split documents into overlapping chunks for better retrieval",
                "components": ["Document chunker", "Token counting", "Section-aware splitting"],
                "output": "Document chunks with metadata"
            },
            {
                "step": 6,
                "name": "Embedding Generation",
                "description": "Generate BGE embeddings for text chunks",
                "components": ["BGE model", "Batch processing", "Embedding storage"],
                "output": "Vector embeddings for semantic search"
            },
            {
                "step": 7,
                "name": "Elasticsearch Indexing",
                "description": "Index all documents and embeddings into Elasticsearch",
                "components": ["ES indexer", "Bulk indexing", "Hybrid search setup"],
                "output": "Searchable paper index"
            }
        ],
        "total_steps": 7,
        "estimated_time": "5-15 minutes per 100 papers (depending on paper size and system performance)"
    }


@router.get("/configuration")
async def get_ingestion_configuration():
    """
    Get current ingestion pipeline configuration.
    
    Returns the current configuration settings for the ingestion pipeline.
    """
    try:
        from backend.config import config
        
        return {
            "elasticsearch": {
                "host": config.ES_HOST,
                "index_name": config.ES_INDEX_NAME
            },
            "minio": {
                "endpoint": config.MINIO_ENDPOINT,
                "bucket": "papers"
            },
            "bge_embedder": {
                "model_name": config.BGE_MODEL_NAME,
                "cache_dir": config.BGE_CACHE_DIR
            },
            "chunking": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP
            },
            "llm": {
                "default_provider": config.DEFAULT_LLM_PROVIDER,
                "openai_model": config.OPENAI_MODEL,
                "google_model": config.GOOGLE_MODEL
            }
        }
        
    except Exception as e:
        log.error(f"Error getting ingestion configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))
