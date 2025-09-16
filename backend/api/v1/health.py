"""
Health and basic status endpoints.

Provides:
- Root endpoint with service information
- Health check endpoint
"""

from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Paper Search Engine API",
        "version": "1.0.0",
        "status": "online"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from backend.api.main import search_service as global_search_service
    search_service = global_search_service

    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    health = search_service.health_check()

    if health["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health)

    return health