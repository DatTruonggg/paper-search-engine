import time
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import redis.asyncio as redis

from app.schemas import Paper, ChatRequest, ChatResponse, Citation, ChatMessage
from app.settings import settings
from app.services.retrieval import RetrievalService


class LLMService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.redis_client = redis.from_url(settings.redis_url) if settings.redis_url else None
        self.retrieval_service = RetrievalService()
    
    async def chat(self, request: ChatRequest, session_id: str = "default") -> ChatResponse:
        """Process chat request with paper retrieval and grounding"""
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        # Retrieve relevant papers
        papers = await self.retrieval_service.retrieve_papers(request.message, request.top_k)
        
        # Build grounded prompt
        prompt = self._build_grounded_prompt(request.message, papers, request.history or [])
        
        # Generate response
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Add conversation history
        if request.history:
            for msg in request.history[-5:]:  # Limit history to last 5 messages
                messages.insert(-1, {"role": msg.role, "content": msg.content})
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Extract citations and summary
            citations, summary = self._extract_citations_and_summary(answer, papers)
            
            # Clean answer (remove citation markers)
            clean_answer = self._clean_answer(answer)
            
            # Cache interaction
            await self._cache_interaction(session_id, request, papers)
            
            return ChatResponse(
                answer=clean_answer,
                citations=citations,
                usedPapers=papers,
                summary=summary
            )
            
        except Exception as e:
            # Fallback response
            return ChatResponse(
                answer=f"I apologize, but I encountered an error processing your request: {str(e)}",
                citations=[],
                usedPapers=papers,
                summary=None
            )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the AI assistant"""
        return """You are Asta, a scholarly research assistant with access to a corpus of academic papers. 
        
Your role is to:
1. Answer scientific questions based on the provided papers
2. Always include citations using [paper_id] format when referencing papers
3. Provide accurate, well-reasoned responses grounded in the literature
4. Mention limitations when the available papers don't fully address the question
5. Suggest related research directions when appropriate

When citing papers, use the format: [paper_id] where paper_id is the ArXiv ID or DOI.

If asked to summarize a paper, provide a concise summary covering:
- Main research question/problem
- Key methodology or approach  
- Primary findings/results
- Significance/implications"""
    
    def _build_grounded_prompt(self, question: str, papers: List[Paper], history: List[ChatMessage]) -> str:
        """Build a grounded prompt with paper context"""
        prompt_parts = [f"Question: {question}\n"]
        
        if papers:
            prompt_parts.append("Relevant Papers:")
            for i, paper in enumerate(papers[:5], 1):  # Limit to top 5 papers
                snippet = paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                prompt_parts.append(
                    f"\n[{paper.id}] {paper.title}\n"
                    f"Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}\n"
                    f"Year: {paper.year}\n"
                    f"Abstract: {snippet}\n"
                    f"Categories: {', '.join(paper.categories)}\n"
                )
        
        prompt_parts.append(
            "\nPlease provide a comprehensive answer based on the papers above. "
            "Include citations using [paper_id] format. If the question asks for a summary, "
            "provide a structured summary of the relevant paper(s)."
        )
        
        return "\n".join(prompt_parts)
    
    def _extract_citations_and_summary(self, answer: str, papers: List[Paper]) -> tuple[List[Citation], Optional[str]]:
        """Extract citations from the answer and generate summary if applicable"""
        import re
        
        citations = []
        paper_dict = {paper.id: paper for paper in papers}
        
        # Find all citation patterns [paper_id]
        citation_pattern = r'\[([^\]]+)\]'
        found_citations = re.findall(citation_pattern, answer)
        
        for cite_id in set(found_citations):  # Remove duplicates
            if cite_id in paper_dict:
                paper = paper_dict[cite_id]
                citations.append(Citation(
                    id=paper.id,
                    doi=paper.doi,
                    where="referenced in answer"
                ))
        
        # Check if this is a summary request
        summary = None
        if any(word in answer.lower() for word in ["summary", "summarize", "summarizes", "in summary"]):
            # Extract potential summary section
            summary_patterns = [
                r"(?:summary|summarize|summarizes|in summary):?\s*(.+?)(?:\n\n|\[|$)",
                r"(?:main findings|key findings|results):?\s*(.+?)(?:\n\n|\[|$)",
            ]
            
            for pattern in summary_patterns:
                match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
                if match:
                    summary = match.group(1).strip()[:200] + "..."
                    break
        
        return citations, summary
    
    def _clean_answer(self, answer: str) -> str:
        """Clean answer by removing internal formatting"""
        # Keep citation markers as they're useful for users
        return answer.strip()
    
    async def _cache_interaction(self, session_id: str, request: ChatRequest, papers: List[Paper]):
        """Cache the interaction in Redis"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"chat_session:{session_id}"
            
            # Store interaction data
            interaction = {
                "timestamp": time.time(),
                "message": request.message,
                "paper_ids": [paper.id for paper in papers],
                "top_k": request.top_k
            }
            
            # Add to session (keep last 10 interactions)
            await self.redis_client.lpush(cache_key, json.dumps(interaction))
            await self.redis_client.ltrim(cache_key, 0, 9)  # Keep only last 10
            await self.redis_client.expire(cache_key, settings.redis_session_ttl)
            
        except Exception as e:
            print(f"Error caching interaction: {e}")
    
    async def get_session_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get cached session context"""
        if not self.redis_client:
            return []
        
        try:
            cache_key = f"chat_session:{session_id}"
            interactions = await self.redis_client.lrange(cache_key, 0, -1)
            
            return [json.loads(interaction) for interaction in interactions]
        except Exception:
            return []
