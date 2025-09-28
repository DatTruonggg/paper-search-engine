"""
Comprehensive health check endpoints for all services.
"""

from logs import log
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
import aiohttp
import asyncio

from backend.config import config


router = APIRouter(prefix="/api/v1/health", tags=["Health Checks"])


@router.get("/comprehensive")
async def comprehensive_health_check(request: Request):
    """
    Comprehensive health check for all services including Elasticsearch, MinIO, and QA Agent.
    
    Returns detailed status information for:
    - Elasticsearch cluster and index
    - MinIO object storage
    - QA Agent and LLM services
    - Search service components
    """
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "services": {}
    }
    
    import datetime
    health_status["timestamp"] = datetime.datetime.now().isoformat()
    
    # Check Elasticsearch
    try:
        search_service = getattr(request.app.state, "search_service", None)
        if search_service:
            es_health = search_service.health_check()
            health_status["services"]["elasticsearch"] = es_health
            if es_health["status"] != "healthy":
                health_status["status"] = "degraded"
        else:
            health_status["services"]["elasticsearch"] = {
                "status": "unhealthy",
                "error": "Search service not initialized"
            }
            health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["services"]["elasticsearch"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check MinIO
    try:
        async with aiohttp.ClientSession() as session:
            # Try to connect to MinIO health endpoint
            minio_health_url = f"{config.MINIO_ENDPOINT}/minio/health/live"
            if not minio_health_url.startswith(('http://', 'https://')):
                minio_health_url = f"http://{minio_health_url}"
            
            async with session.get(minio_health_url, timeout=5) as response:
                if response.status == 200:
                    health_status["services"]["minio"] = {
                        "status": "healthy",
                        "endpoint": config.MINIO_ENDPOINT,
                        "bucket": "papers"
                    }
                else:
                    health_status["services"]["minio"] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
                        "endpoint": config.MINIO_ENDPOINT
                    }
                    health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["minio"] = {
            "status": "unhealthy",
            "error": str(e),
            "endpoint": config.MINIO_ENDPOINT
        }
        health_status["status"] = "degraded"
    
    # Check QA Agent
    try:
        from backend.qa_agent import QAAgent
        from backend.services import ElasticsearchSearchService
        
        # Initialize QA agent for health check
        es_service = ElasticsearchSearchService(
            es_host=config.ES_HOST,
            index_name=config.ES_INDEX_NAME,
            bge_model=config.BGE_MODEL_NAME,
            bge_cache_dir=config.BGE_CACHE_DIR
        )
        
        qa_agent = QAAgent(es_service=es_service)
        qa_health = await qa_agent.health_check()
        
        health_status["services"]["qa_agent"] = qa_health
        
        if qa_health["status"] != "healthy":
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["qa_agent"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check BGE Embedder
    try:
        from data_pipeline.bge_embedder import BGEEmbedder
        
        embedder = BGEEmbedder(
            model_name=config.BGE_MODEL_NAME,
            cache_dir=config.BGE_CACHE_DIR
        )
        
        # Test embedding generation
        test_embedding = embedder.encode("test")
        
        health_status["services"]["bge_embedder"] = {
            "status": "healthy",
            "model": config.BGE_MODEL_NAME,
            "embedding_dim": embedder.embedding_dim,
            "test_embedding_shape": test_embedding.shape if hasattr(test_embedding, 'shape') else "unknown"
        }
        
    except Exception as e:
        health_status["services"]["bge_embedder"] = {
            "status": "unhealthy",
            "error": str(e),
            "model": config.BGE_MODEL_NAME
        }
        health_status["status"] = "degraded"
    
    # Check LLM Services
    try:
        llm_health = {
            "status": "healthy",
            "providers": {}
        }
        
        # Check OpenAI
        if config.OPENAI_API_KEY:
            try:
                from llama_index.llms.openai import OpenAI
                llm = OpenAI(
                    model=config.OPENAI_MODEL,
                    api_key=config.OPENAI_API_KEY,
                    timeout=10
                )
                test_response = asyncio.run(llm.acomplete("test"))
                llm_health["providers"]["openai"] = {
                    "status": "healthy",
                    "model": config.OPENAI_MODEL
                }
            except Exception as e:
                llm_health["providers"]["openai"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "model": config.OPENAI_MODEL
                }
                llm_health["status"] = "degraded"
        
        # Check Google Gemini
        if config.GOOGLE_API_KEY:
            try:
                from llama_index.llms.gemini import Gemini
                llm = Gemini(
                    model=config.GOOGLE_MODEL,
                    api_key=config.GOOGLE_API_KEY,
                    timeout=10
                )
                test_response = asyncio.run(llm.acomplete("test"))
                llm_health["providers"]["gemini"] = {
                    "status": "healthy",
                    "model": config.GOOGLE_MODEL
                }
            except Exception as e:
                llm_health["providers"]["gemini"] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "model": config.GOOGLE_MODEL
                }
                llm_health["status"] = "degraded"
        
        health_status["services"]["llm_services"] = llm_health
        
        if llm_health["status"] != "healthy":
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["services"]["llm_services"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Overall status determination
    unhealthy_services = []
    degraded_services = []
    
    for service_name, service_health in health_status["services"].items():
        if service_health.get("status") == "unhealthy":
            unhealthy_services.append(service_name)
        elif service_health.get("status") == "degraded":
            degraded_services.append(service_name)
    
    if unhealthy_services:
        health_status["status"] = "unhealthy"
        health_status["unhealthy_services"] = unhealthy_services
    
    if degraded_services:
        health_status["degraded_services"] = degraded_services
    
    # Add summary
    health_status["summary"] = {
        "total_services": len(health_status["services"]),
        "healthy_services": len([s for s in health_status["services"].values() if s.get("status") == "healthy"]),
        "degraded_services": len(degraded_services),
        "unhealthy_services": len(unhealthy_services)
    }
    
    return health_status


@router.get("/elasticsearch")
async def elasticsearch_health_check(request: Request):
    """Health check specifically for Elasticsearch"""
    try:
        search_service = getattr(request.app.state, "search_service", None)
        if not search_service:
            raise HTTPException(status_code=503, detail="Search service not initialized")
        
        health = search_service.health_check()
        
        if health["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/minio")
async def minio_health_check():
    """Health check specifically for MinIO"""
    try:
        async with aiohttp.ClientSession() as session:
            minio_health_url = f"{config.MINIO_ENDPOINT}/minio/health/live"
            if not minio_health_url.startswith(('http://', 'https://')):
                minio_health_url = f"http://{minio_health_url}"
            
            async with session.get(minio_health_url, timeout=5) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "endpoint": config.MINIO_ENDPOINT,
                        "bucket": "papers",
                        "response_time_ms": response.headers.get("X-Response-Time", "unknown")
                    }
                else:
                    raise HTTPException(
                        status_code=503, 
                        detail={
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "endpoint": config.MINIO_ENDPOINT
                        }
                    )
                    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "endpoint": config.MINIO_ENDPOINT
            }
        )


@router.get("/qa-agent")
async def qa_agent_health_check():
    """Health check specifically for QA Agent"""
    try:
        from backend.qa_agent import QAAgent
        from backend.services import ElasticsearchSearchService
        
        # Initialize QA agent for health check
        es_service = ElasticsearchSearchService(
            es_host=config.ES_HOST,
            index_name=config.ES_INDEX_NAME,
            bge_model=config.BGE_MODEL_NAME,
            bge_cache_dir=config.BGE_CACHE_DIR
        )
        
        qa_agent = QAAgent(es_service=es_service)
        health = await qa_agent.health_check()
        
        if health["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health)
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
