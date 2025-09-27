from typing import Any, Dict
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from asta.api.scholarqa import ScholarQA, PaperFinderWithReranker, PaperFinder
from asta.api.scholarqa.rag.retriever_base import FullTextRetriever
from asta.api.scholarqa.rag.reranker.reranker_base import CrossEncoderScores
from asta.api.scholarqa.llms.constants import GEMINI_25_FLASH
from logs import log

from asta.api.scholarqa.rag.reranker.modal_engine import ModalReranker


router = APIRouter(prefix="/api/v1/asta", tags=["ASTA QA"])


class QARequest(BaseModel):
    """Request schema for ScholarQA endpoint."""
    query: str = Field(..., min_length=1)


class QAResult(BaseModel):
    """Response schema wrapping ScholarQA result payload."""
    result: Dict[str, Any]


@router.post("/qa", response_model=QAResult)
async def asta_qa(req: QARequest) -> QAResult:
    """Run the ScholarQA pipeline end-to-end for a user query.

    Flow:
    - Retrieve passages via Semantic Scholar full-text and keyword search
    - Rerank passages with a cross-encoder
    - Generate structured answer with citations via LLM
    """
    try:
        log.info(f"[ASTA][API] /asta/qa start | query='{req.query}'")

        # Build retrieval + reranking stack
        retriever = FullTextRetriever(n_retrieval=1000, n_keyword_srch=30)
        reranker = ModalReranker(app_name='<modal_app_name>', api_name='<modal_api_name>', batch_size=256, gen_options=dict())

        # Wrap into PaperFinder; n_rerank=-1 keeps all candidates post-rerank
        paper_finder = PaperFinder(retriever, n_rerank=-1, context_threshold=0.0)

        # ScholarQA orchestrates the multi-step QA with the chosen LLM
        scholar_qa = ScholarQA(paper_finder=paper_finder, llm_model=GEMINI_25_FLASH)
        result = scholar_qa.answer_query(req.query)

        log.info("[ASTA][API] /asta/qa done | sections=%s", len(result.get("sections", [])) if isinstance(result, dict) else 'N/A')
        return QAResult(result=result)
    except Exception as e:
        log.exception("ASTA ScholarQA failed")
        raise HTTPException(status_code=500, detail=str(e))