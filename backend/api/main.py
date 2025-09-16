"""
FastAPI backend for Paper Search Engine with Elasticsearch.
"""

<<<<<<< Updated upstream
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
=======
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
>>>>>>> Stashed changes
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

<<<<<<< Updated upstream
from backend.services import ElasticsearchSearchService, SearchResult, PaperDetails

# Configure logging
logging.basicConfig(level=logging.INFO)
=======
from backend.services import ElasticsearchSearchService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG_MODE", "true").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
>>>>>>> Stashed changes
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Paper Search Engine API",
    description="Elasticsearch-based academic paper search with BGE embeddings",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

<<<<<<< Updated upstream
# Initialize search service
search_service = None


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query text")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    search_mode: str = Field("hybrid", description="Search mode: hybrid, semantic, bm25, title_only")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    date_from: Optional[str] = Field(None, description="Filter papers from this date")
    date_to: Optional[str] = Field(None, description="Filter papers to this date")
    author: Optional[str] = Field(None, description="Filter by author name")

    def model_post_init(self, __context) -> None:
        """Auto-clean filter values - set 'string' literals to None"""
        if self.categories == ["string"]:
            self.categories = None
        if self.date_from == "string":
            self.date_from = None
        if self.date_to == "string":
            self.date_to = None
        if self.author == "string":
            self.author = None


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[Dict[str, Any]]
    total: int
    query: str
    search_mode: str


class SimilarPapersRequest(BaseModel):
    """Similar papers request model"""
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
=======
# Global search service for access from routers
search_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup and attach to app.state."""
>>>>>>> Stashed changes
    global search_service

    from backend.config import config

    logger.info(f"Initializing search service with ES host: {config.ES_HOST}")

    try:
        search_service = ElasticsearchSearchService(
            es_host=config.ES_HOST,
            index_name=config.ES_INDEX_NAME,
            bge_model=config.BGE_MODEL_NAME,
            bge_cache_dir=config.BGE_CACHE_DIR
        )
        logger.info("Search service initialized successfully")
<<<<<<< Updated upstream
=======
        # Attach to app state for access within routers
        app.state.search_service = search_service
>>>>>>> Stashed changes
    except Exception as e:
        logger.error(f"Failed to initialize search service: {e}")
        raise


<<<<<<< Updated upstream
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Paper Search Engine API",
        "version": "1.0.0",
        "status": "online"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    health = search_service.health_check()

    if health["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health)

    return health


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(request: SearchRequest):
    """
    Search for papers using various search modes.

    Search modes:
    - hybrid: Combines BM25 (40%) and semantic search (60%)
    - semantic: Pure embedding-based similarity search
    - bm25: Traditional full-text search
    - title_only: Search only in paper titles
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        logger.info(f"Search request: query='{request.query}', mode={request.search_mode}")

        results = search_service.search(
            query=request.query,
            max_results=request.max_results,
            search_mode=request.search_mode,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
            author=request.author
        )

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "score": result.score,
                "categories": result.categories,
                "publish_date": result.publish_date,
                "word_count": result.word_count,
                "has_images": result.has_images,
                "pdf_size": result.pdf_size
            })

        return SearchResponse(
            results=results_dict,
            total=len(results),
            query=request.query,
            search_mode=request.search_mode
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/papers/{paper_id}", tags=["Papers"])
async def get_paper(paper_id: str):
    """
    Get detailed information about a specific paper.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        paper = search_service.get_paper_details(paper_id)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "content": paper.content[:1000] + "..." if len(paper.content) > 1000 else paper.content,
            "content_length": len(paper.content),
            "categories": paper.categories,
            "publish_date": paper.publish_date,
            "word_count": paper.word_count,
            "chunk_count": paper.chunk_count,
            "has_images": paper.has_images,
            "pdf_size": paper.pdf_size,
            "downloaded_at": paper.downloaded_at,
            "indexed_at": paper.indexed_at,
            "markdown_path": paper.markdown_path,
            "pdf_path": paper.pdf_path,
            "minio_pdf_url": paper.minio_pdf_url,
            "minio_markdown_url": paper.minio_markdown_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/papers/{paper_id}/similar", tags=["Papers"])
async def find_similar_papers(paper_id: str, request: SimilarPapersRequest):
    """
    Find papers similar to a given paper.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        results = search_service.find_similar_papers(
            paper_id=paper_id,
            max_results=request.max_results
        )

        if not results:
            raise HTTPException(status_code=404, detail=f"Reference paper not found: {paper_id}")

        # Convert results to dict format
        results_dict = []
        for result in results:
            results_dict.append({
                "paper_id": result.paper_id,
                "title": result.title,
                "authors": result.authors,
                "abstract": result.abstract,
                "similarity_score": result.score,
                "categories": result.categories,
                "publish_date": result.publish_date
            })

        return {
            "reference_paper_id": paper_id,
            "similar_papers": results_dict,
            "total": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar papers for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/es/stats", tags=["Elasticsearch"])
async def get_index_stats():
    """
    Get statistics about the search index.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        stats = search_service.get_index_stats()
        return stats

    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/es/stats/detailed", tags=["Elasticsearch"])
async def get_detailed_stats():
    """
    Get detailed statistics about papers and chunks in Elasticsearch.
    Shows paper count, chunk count, and breakdown by categories.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Get basic index stats
        basic_stats = search_service.get_index_stats()

        # Get detailed paper and chunk information
        search_body = {
            "size": 10000,  # Get all documents
            "_source": ["paper_id", "title", "categories", "chunk_count", "content_chunks"],
            "query": {"match_all": {}}
        }

        response = search_service.indexer.es.search(
            index=search_service.indexer.index_name,
            body=search_body
        )

        papers = response['hits']['hits']

        # Calculate detailed statistics
        total_papers = len(papers)
        total_chunks = 0
        category_paper_count = {}
        category_chunk_count = {}
        paper_details = []

        for hit in papers:
            source = hit['_source']
            paper_id = source.get('paper_id', '')
            title = source.get('title', '')
            categories = source.get('categories', [])
            chunk_count = source.get('chunk_count', 0)
            content_chunks = source.get('content_chunks', [])

            # Count actual chunks from content_chunks array
            actual_chunks = len(content_chunks) if content_chunks else chunk_count
            total_chunks += actual_chunks

            paper_details.append({
                "paper_id": paper_id,
                "title": title,
                "categories": categories,
                "chunk_count": actual_chunks
            })

            # Count by categories
            for category in categories:
                category_paper_count[category] = category_paper_count.get(category, 0) + 1
                category_chunk_count[category] = category_chunk_count.get(category, 0) + actual_chunks

        # Calculate averages
        avg_chunks_per_paper = total_chunks / total_papers if total_papers > 0 else 0

        detailed_stats = {
            "summary": {
                "total_papers": total_papers,
                "total_chunks": total_chunks,
                "avg_chunks_per_paper": round(avg_chunks_per_paper, 2),
                "index_size_mb": basic_stats.get("index_size_mb", 0)
            },
            "by_category": {
                "paper_count": category_paper_count,
                "chunk_count": category_chunk_count
            },
            "papers": paper_details,
            "elasticsearch_info": {
                "cluster_name": basic_stats.get("cluster_name", "unknown"),
                "document_count": basic_stats.get("document_count", 0),
                "index_size": basic_stats.get("index_size", 0)
            }
        }

        return detailed_stats

    except Exception as e:
        logger.error(f"Error getting detailed stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/search/suggest", tags=["Search"])
async def search_suggestions(
    query: str = Query(..., description="Query prefix for suggestions"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum number of suggestions")
):
    """
    Get search suggestions based on query prefix.
    This is a simplified version - in production, you'd want to implement
    proper autocomplete with a dedicated suggestion index.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # For now, just search titles
        results = search_service.search(
            query=query,
            max_results=max_results,
            search_mode="title_only"
        )

        suggestions = [result.title for result in results]

        return {
            "query": query,
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/papers/{paper_id}/pdf", tags=["Papers"])
async def get_pdf_url(paper_id: str):
    """
    Get presigned URL for PDF download.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Check if paper exists
        paper = search_service.get_paper_details(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        # Check if MinIO PDF URL exists
        if not paper.minio_pdf_url:
            raise HTTPException(status_code=404, detail=f"PDF not available in MinIO for paper: {paper_id}")

        # Get presigned URL from MinIO storage
        from data_pipeline.minio_storage import MinIOStorage
        minio_storage = MinIOStorage(endpoint="localhost:9002")

        presigned_url = minio_storage.get_pdf_url(paper_id)

        if not presigned_url:
            raise HTTPException(status_code=404, detail=f"Could not generate PDF download URL for: {paper_id}")

        return {
            "paper_id": paper_id,
            "title": paper.title,
            "download_url": presigned_url,
            "expires_in_seconds": 3600,
            "file_size_bytes": paper.pdf_size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting PDF URL for {paper_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simple PDF management endpoints
@app.get("/api/v1/papers/{paper_id}/info", tags=["Papers"])
async def get_paper_with_pdf_info(paper_id: str):
    """Get paper information with PDF availability status."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()
        paper_info = paper_service.get_paper_info(paper_id)

        if not paper_info:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        return paper_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/storage/summary", tags=["Storage"])
async def get_storage_summary():
    """Get summary of papers vs PDFs."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()
        return paper_service.get_summary()

    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/papers/with-pdfs", tags=["Papers"])
async def list_papers_with_pdfs(limit: int = 20):
    """List papers that have PDFs available."""
    try:
        from backend.services.paper_service import PaperService

        paper_service = PaperService()
        papers = paper_service.list_papers_with_pdfs(limit)

        return {
            "papers": papers,
            "count": len(papers)
        }

    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/storage/upload-pdf", tags=["Storage"])
async def upload_pdf(paper_id: str):
    """Upload PDF for a paper by ID."""
    try:
        from backend.services.paper_service import PaperService
        from pathlib import Path

        paper_service = PaperService()

        # Look for PDF file
        pdf_dir = Path("/Users/admin/code/cazoodle/data/pdfs")
        pdf_path = pdf_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")

        result = paper_service.upload_pdf_for_paper(paper_id, pdf_path)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/storage/pdf/{paper_id}/download", tags=["Storage"])
async def get_pdf_download_url(paper_id: str):
    """Get download URL for a PDF."""
    try:
        from backend.services.pdf_service import PDFService

        pdf_service = PDFService()

        if not pdf_service.pdf_exists(paper_id):
            raise HTTPException(status_code=404, detail=f"PDF not found for: {paper_id}")

        download_url = pdf_service.get_pdf_url(paper_id)

        if not download_url:
            raise HTTPException(status_code=500, detail="Could not generate download URL")

        return {
            "paper_id": paper_id,
            "download_url": download_url,
            "expires_in": "1 hour"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting download URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

=======
# Include all v1 routers
from backend.api.v1 import health, search, papers, es, minio, agent

app.include_router(health.router)
app.include_router(search.router)
app.include_router(papers.router)
app.include_router(es.router)
app.include_router(minio.router)
app.include_router(agent.router)

logger.info("All API routes configured")
>>>>>>> Stashed changes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)