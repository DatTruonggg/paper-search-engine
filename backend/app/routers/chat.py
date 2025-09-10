from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.schemas import ChatRequest, ChatResponse
from app.services.llm import LLMService

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_papers(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Chat with grounded responses from papers"""
    try:
        llm_service = LLMService()
        session_id = x_session_id or "default"
        
        response = await llm_service.chat(request, session_id)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
