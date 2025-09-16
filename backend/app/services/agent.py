#TODO: new 
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

import numpy as np

from app.schemas import Paper
from app.services.retrieval import RetrievalService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.llm import LLMService
from app.settings import settings


class AgentService:
    """Agent providing LLM summarization and QA over retrieved papers.

    Modes:
    - Summarize single paper (by id)
    - Summarize across top-k retrieved papers (by query)
    - QA for a chosen paper (single-paper QA)
    - QA across top-k retrieved papers (multi-paper QA)
    """

    def __init__(self):
        self.retrieval = RetrievalService()
        self.search = ElasticsearchService()
        self.llm = LLMService()

    async def summarize_paper(self, paper_id: str, max_tokens: int = 600) -> Dict[str, Any]:
        papers = await self.search.get_papers_by_ids([paper_id])
        if not papers:
            return {"success": False, "error": f"Paper not found: {paper_id}"}

        paper = papers[0]
        prompt = self._build_summary_prompt_single(paper)

        answer = await self._llm_complete(prompt, max_tokens=max_tokens)
        return {
            "success": True,
            "paperId": paper.id,
            "title": paper.title,
            "summary": answer,
        }

    async def summarize_query(self, query: str, top_k: int = 5, max_tokens: int = 700) -> Dict[str, Any]:
        papers = await self.retrieval.retrieve_papers(query, top_k)
        if not papers:
            return {"success": False, "error": "No relevant papers found"}

        prompt = self._build_summary_prompt_multi(query, papers)
        answer = await self._llm_complete(prompt, max_tokens=max_tokens)
        return {
            "success": True,
            "query": query,
            "summary": answer,
            "sources": [{"id": p.id, "title": p.title} for p in papers],
        }

    async def qa_single_paper(self, question: str, paper_id: str, max_tokens: int = 800) -> Dict[str, Any]:
        papers = await self.search.get_papers_by_ids([paper_id])
        if not papers:
            return {"success": False, "error": f"Paper not found: {paper_id}"}

        paper = papers[0]
        prompt = self._build_qa_prompt(question, [paper])
        answer = await self._llm_complete(prompt, max_tokens=max_tokens)
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": [{"id": paper.id, "title": paper.title}],
        }

    async def qa_multi_paper(self, question: str, top_k: int = 5, max_tokens: int = 900) -> Dict[str, Any]:
        papers = await self.retrieval.retrieve_papers(question, top_k)
        if not papers:
            return {"success": False, "error": "No relevant papers found"}

        prompt = self._build_qa_prompt(question, papers)
        answer = await self._llm_complete(prompt, max_tokens=max_tokens)
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": [{"id": p.id, "title": p.title} for p in papers],
        }

    # ---------------------------- Prompt builders ----------------------------
    def _build_summary_prompt_single(self, paper: Paper) -> str:
        abstract_snippet = paper.abstract[:1500]
        return (
            "You are Asta, a scholarly assistant. Summarize the following paper clearly and concisely.\n"
            "Provide: Problem, Method, Results, and Significance. Keep it factual and grounded.\n\n"
            f"[{paper.id}] {paper.title} ({paper.year})\n"
            f"Authors: {', '.join(paper.authors)}\n"
            f"Categories: {', '.join(paper.categories)}\n\n"
            f"Abstract:\n{abstract_snippet}\n\n"
            "Answer:" 
        )

    def _build_summary_prompt_multi(self, query: str, papers: List[Paper]) -> str:
        header = (
            "You are Asta, a scholarly assistant. Summarize the state of the art for the user query,"
            " integrating key papers. Provide a structured synthesis (themes, methods, findings, gaps).\n\n"
            f"Query: {query}\n\nRelevant Papers (abstract snippets):\n"
        )
        body_parts = []
        for p in papers[:5]:
            snippet = p.abstract[:600]
            body_parts.append(
                f"[{p.id}] {p.title} ({p.year})\n"
                f"Abstract: {snippet}\n"
            )
        return header + "\n".join(body_parts) + "\nAnswer:"

    def _build_qa_prompt(self, question: str, papers: List[Paper]) -> str:
        parts = [
            "You are Asta, a scholarly assistant. Answer the user question ONLY using the papers provided.",
            "Cite papers with [paper_id]. If unknown or not supported by papers, say so.",
            f"\nQuestion: {question}\n\nPapers:\n",
        ]
        for p in papers[:5]:
            snippet = p.abstract[:800]
            parts.append(
                f"[{p.id}] {p.title} ({p.year})\n"
                f"Abstract: {snippet}\n"
            )
        parts.append("\nAnswer (with citations):")
        return "\n".join(parts)

    # ------------------------------ LLM helper -------------------------------
    async def _llm_complete(self, prompt: str, max_tokens: int) -> str:
        # Reuse LLMService's configured OpenAI client and system prompt
        if not self.llm.client:
            raise ValueError("OpenAI API key not configured")
        messages = [
            {"role": "system", "content": self.llm._get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        resp = await self.llm.client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content


