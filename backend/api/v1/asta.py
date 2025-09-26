"""
ASTA testing routes for retrieval and end-to-end QA.

Endpoints:
- POST /api/v1/asta/retrieve: returns ES-backed ASTA-formatted snippets (no LLM)
- POST /api/v1/asta/qa: runs ASTA ScholarQA pipeline (LLM required per ASTA config)
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from backend.agent.asta_retriever import ElasticsearchAstaRetriever
from backend.agent.asta_agent import AstaQAPipelineAgent
from backend.services import ElasticsearchSearchService
from backend.config import config
from logs import log


router = APIRouter(prefix="/api/v1/asta", tags=["ASTA QA"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    n_retrieval: int = Field(default=20, ge=1, le=256)
    year: Optional[str] = Field(default=None, description="YYYY-YYYY or YYYY or 'start-end'")


class RetrieveResponse(BaseModel):
    count: int
    snippets: List[Dict[str, Any]]


class QARequest(BaseModel):
    query: str = Field(..., min_length=1)


class QAResult(BaseModel):
    result: Dict[str, Any]


def _get_es_service() -> ElasticsearchSearchService:
    return ElasticsearchSearchService(
        es_host=config.ES_HOST,
        index_name=config.ES_INDEX_NAME,
        bge_model=config.BGE_MODEL_NAME,
        bge_cache_dir=config.BGE_CACHE_DIR,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def asta_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Run retrieval only (no LLM), return ASTA-shaped snippets from ES."""
    try:
        log.info(f"[ASTA][API] /asta/retrieve start | query='{req.query}' n_retrieval={req.n_retrieval} year={req.year}")
        es = _get_es_service()
        retriever = ElasticsearchAstaRetriever(es_service=es, n_retrieval=req.n_retrieval)
        kwargs = {}
        if req.year:
            kwargs["year"] = req.year
        snippets = retriever.retrieve_passages(req.query, **kwargs)
        resp = RetrieveResponse(count=len(snippets), snippets=snippets)
        log.info(f"[ASTA][API] /asta/retrieve done | count={resp.count}")
        return resp
    except Exception as e:
        log.exception("ASTA retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa", response_model=QAResult)
async def asta_qa(req: QARequest) -> QAResult:
    """Run full ASTA pipeline (preprocess, retrieval, rerank, generation)."""
    try:
        log.info(f"[ASTA][API] /asta/qa start | query='{req.query}'")
        agent = AstaQAPipelineAgent()
        result = agent.answer_query(req.query)
        log.info("[ASTA][API] /asta/qa done | sections=%s", len(result.get("sections", [])) if isinstance(result, dict) else 'N/A')
        return QAResult(result=result)
    except Exception as e:
        log.exception("ASTA QA failed")
        raise HTTPException(status_code=500, detail=str(e))


