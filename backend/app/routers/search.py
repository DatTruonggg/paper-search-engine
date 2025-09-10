from fastapi import APIRouter, HTTPException
from app.schemas import SearchRequest, SearchResponse
from app.services.retrieval import RetrievalService
from app.settings import settings

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """Search papers with filters and sorting"""
    try:
        retrieval_service = RetrievalService()
        
        if settings.data_backend == "es":
            from app.services.elasticsearch_service import ElasticsearchService
            search_service = ElasticsearchService()
        else:
            from app.services.postgres_service import PostgresService
            search_service = PostgresService()
        
        response = await search_service.search_papers(request)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
