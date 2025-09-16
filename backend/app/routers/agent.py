#TODO: new
from fastapi import APIRouter, HTTPException
from app.schemas import SummarizeRequest, SummarizeResponse, QARequest, QAResponse
from app.services.agent import AgentService


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    try:
        agent = AgentService()
        if request.paper_id:
            result = await agent.summarize_paper(request.paper_id, max_tokens=request.max_tokens)
        elif request.query:
            result = await agent.summarize_query(request.query, top_k=request.top_k, max_tokens=request.max_tokens)
        else:
            raise HTTPException(status_code=400, detail="Provide either paperId or query")
        return SummarizeResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/qa", response_model=QAResponse)
async def qa(request: QARequest):
    try:
        agent = AgentService()
        if request.paper_id:
            result = await agent.qa_single_paper(request.question, request.paper_id, max_tokens=request.max_tokens)
        else:
            result = await agent.qa_multi_paper(request.question, top_k=request.top_k, max_tokens=request.max_tokens)
        return QAResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")


