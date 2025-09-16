"""
FastAPI backend for Paper Search Engine with Elasticsearch.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services import ElasticsearchSearchService
from backend.config import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG_MODE", "true").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

"""Eagerly initialize core services at import time."""
logger.info(f"Initializing search service with ES host: {config.ES_HOST}")

try:
    # Global search service for access from routers
    search_service = ElasticsearchSearchService(
        es_host=config.ES_HOST,
        index_name=config.ES_INDEX_NAME,
        bge_model=config.BGE_MODEL_NAME,
        bge_cache_dir=config.BGE_CACHE_DIR
    )
    # Attach to app state for routes that prefer accessing via request.app.state
    app.state.search_service = search_service
    logger.info("Search service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize search service: {e}")
    raise


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
    global search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    health = search_service.health_check()

    if health["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health)

    return health


# Include all v1 routers
from backend.api.v1 import search, papers, es, minio, agent

app.include_router(search.router)
app.include_router(papers.router)
app.include_router(es.router)
app.include_router(minio.router)
app.include_router(agent.router)

logger.info("All API routes configured")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)