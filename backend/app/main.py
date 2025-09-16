from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

from app.settings import settings
from app.routers import search, chat, ingest
from app.routers import agent as agent_router #TODO: New


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("Starting Paper Search Engine API...")
    
    # Initialize search backend
    if settings.data_backend == "es":
        try:
            from app.services.elasticsearch_service import ElasticsearchService
            es_service = ElasticsearchService()
            await es_service.create_index()
            print("✅ Elasticsearch initialized")
        except Exception as e:
            print(f"⚠️  Elasticsearch initialization failed: {e}")
    elif settings.data_backend == "pg":
        try:
            from app.services.postgres_service import PostgresService
            pg_service = PostgresService()
            print("✅ PostgreSQL initialized")
        except Exception as e:
            print(f"⚠️  PostgreSQL initialization failed: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down Paper Search Engine API...")


# Create FastAPI application
app = FastAPI(
    title="Paper Search Engine API",
    description="AI-powered search and chat interface for academic papers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router)
app.include_router(chat.router)  
app.include_router(ingest.router)
app.include_router(agent_router.router) #TODO: New

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Paper Search Engine API",
        "version": "1.0.0",
        "backend": settings.data_backend,
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "backend": settings.data_backend,
        "services": {}
    }
    
    # Check search backend
    try:
        if settings.data_backend == "es":
            from app.services.elasticsearch_service import ElasticsearchService
            es_service = ElasticsearchService()
            # Simple ping test would go here
            status["services"]["elasticsearch"] = "available"
        else:
            from app.services.postgres_service import PostgresService
            pg_service = PostgresService()
            # Simple connection test would go here
            status["services"]["postgresql"] = "available"
    except Exception as e:
        status["services"][settings.data_backend] = f"error: {str(e)}"
    
    # Check Redis
    try:
        if settings.redis_url:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.redis_url)
            await redis_client.ping()
            status["services"]["redis"] = "available"
            await redis_client.close()
    except Exception:
        status["services"]["redis"] = "unavailable"
    
    # Check MinIO
    try:
        if settings.minio_endpoint:
            from app.services.minio_service import MinIOService
            minio_service = MinIOService()
            if minio_service.client:
                status["services"]["minio"] = "available"
            else:
                status["services"]["minio"] = "not configured"
    except Exception:
        status["services"]["minio"] = "unavailable"
    
    # Check OpenAI
    status["services"]["openai"] = "configured" if settings.openai_api_key else "not configured"
    
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
