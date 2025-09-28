"""
Evidence extraction service for LlamaIndex agent.
Finds relevant text chunks from papers to support search queries.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from backend.services import ElasticsearchSearchService
from data_pipeline.bge_embedder import BGEEmbedder
from logs import log
from pydantic import BaseModel, Field
from llama_index.llms.gemini import Gemini
from llama_index.core.program import LLMTextCompletionProgram
from .config import llama_config


@dataclass
class EvidenceChunk:
    """A relevant text chunk from a paper"""
    chunk_index: int
    chunk_text: str
    relevance_score: float
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None


@dataclass
class PaperWithEvidence:
    """Paper with extracted evidence sentences"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    publish_date: Optional[str]
    elasticsearch_score: float
    evidence_sentences: List[str]


class EvidenceExtractor:
    """
    Extracts relevant text chunks from papers as evidence for search queries.
    """

    def __init__(
        self,
        es_service: ElasticsearchSearchService,
        embedder: Optional[BGEEmbedder] = None,
        max_chunks_per_paper: int = 5,
        min_relevance_score: float = 0.6,
        llm: Optional[Gemini] = None
    ):
        """
        Initialize evidence extractor.

        Args:
            es_service: Elasticsearch search service
            embedder: Optional BGE embedder (will use es_service.embedder if None)
            max_chunks_per_paper: Maximum evidence chunks per paper
            min_relevance_score: Minimum relevance score for chunks
            llm: Optional Gemini LLM for quick evidence extraction
        """
        self.es_service = es_service
        self.embedder = embedder or es_service.embedder
        self.max_chunks_per_paper = max_chunks_per_paper
        self.min_relevance_score = min_relevance_score
        # Initialize LLM for quick evidence extraction
        self.llm = llm or Gemini(
            model=llama_config.gemini_model,
            api_key=llama_config.gemini_api_key,
            temperature=0.1,
            timeout=llama_config.llm_timeout
        )
        # Program for extracting relevant sentences from title/abstract
        class QuickEvidenceOutput(BaseModel):
            """Structured output for quick evidence extraction"""
            sentences: List[str] = Field(description="Relevant sentences most related to the query")

        self._quick_evidence_output_cls = QuickEvidenceOutput
        self.quick_evidence_program = LLMTextCompletionProgram.from_defaults(
            output_cls=QuickEvidenceOutput,
            prompt_template_str=(
                "Given a research query, a paper title, and an abstract, "
                "extract up to {max_sentences} distinct sentences from the title/abstract that are most relevant to the query.\n"
                "If no clearly relevant sentences can be found, return no evidence found.\n"
                "Never return an empty list. Always return at least one sentence.\n"
                "Return ONLY a JSON object with a 'sentences' array.\n\n"
                "Query: {query}\n"
                "Title: {title}\n"
                "Abstract: {abstract}\n"
                "Max sentences: {max_sentences}"
            ),
            llm=self.llm
        )

        # Program for extracting relevant sentences from full paper content
        class FullEvidenceOutput(BaseModel):
            """Structured output for full-paper evidence extraction"""
            sentences: List[str] = Field(description="Relevant sentences most related to the query from the full paper")

        self._full_evidence_output_cls = FullEvidenceOutput
        self.full_evidence_program = LLMTextCompletionProgram.from_defaults(
            output_cls=FullEvidenceOutput,
            prompt_template_str=(
                "You are given a research query and the full text content of a paper.\n"
                "Select up to {max_sentences} distinct sentences from the paper that most directly support or relate to the query.\n"
                "Prefer sentences with concrete findings, methods, or results. Avoid redundant sentences.\n"
                "Never return an empty list. If nothing is strongly relevant, return the most informative sentences about the topic.\n"
                "Return ONLY a JSON object with a 'sentences' array.\n\n"
                "Query: {query}\n"
                "Paper Content: {content}\n"
                "Max sentences: {max_sentences}"
            ),
            llm=self.llm
        )

    async def extract_evidence(
        self,
        papers: List[Dict[str, Any]],
        query: str
    ) -> List[PaperWithEvidence]:
        """
        Extract relevant sentences for a list of papers using title and abstract only.

        Args:
            papers: List of paper dictionaries from search results
            query: Original search query

        Returns:
            List of PaperWithEvidence objects
        """
        log.info(f"Extracting quick evidences for {len(papers)} papers with query: '{query}'")

        results = []
        for paper in papers:
            try:
                paper_with_evidence = await self._extract_paper_evidence(paper, query)
                if paper_with_evidence:
                    results.append(paper_with_evidence)
            except Exception as e:
                log.warning(f"Failed to extract quick evidences for paper {paper.get('paper_id', 'unknown')}: {e}")
                # Still include paper without evidence sentences
                results.append(self._create_paper_without_evidence(paper))
        log.info(f"Quick evidence extraction results: {results[:5]}")
        log.info(f"Quick evidence extraction completed for {len(results)} papers")
        return results

    async def _extract_paper_evidence(
        self,
        paper: Dict[str, Any],
        query: str
    ) -> Optional[PaperWithEvidence]:
        """
        Extract relevant sentences for a single paper (title + abstract).

        Args:
            paper: Paper dictionary from search results
            query: Search query
            query_embedding: Query embedding vector

        Returns:
            PaperWithEvidence or None if paper not found
        """
        paper_id = paper.get('paper_id')
        if not paper_id:
            log.warning("Paper missing paper_id, skipping evidence extraction")
            return self._create_paper_without_evidence(paper)

        # Extract sentences directly from title and abstract
        sentences = await self.quick_evidences_for_paper(
            query=query,
            paper=paper,
            max_sentences=self.max_chunks_per_paper
        )

        return PaperWithEvidence(
            paper_id=paper_id,
            title=paper.get('title', 'Untitled'),
            authors=paper.get('authors', []),
            abstract=paper.get('abstract', ''),
            categories=paper.get('categories', []),
            publish_date=paper.get('publish_date'),
            elasticsearch_score=paper.get('score', 0.0),
            evidence_sentences=sentences
        )

    async def quick_evidences(
        self,
        query: str,
        title: str,
        abstract: str,
        max_sentences: int = 3
    ) -> List[str]:
        """
        Quickly extract relevant sentences from a paper's title and abstract using the LLM.

        Args:
            query: Original user query guiding relevance.
            title: Paper title.
            abstract: Paper abstract.
            max_sentences: Upper bound on the number of sentences to return.

        Returns:
            List of relevant sentences, ordered by estimated relevance.
        """
        try:
            output: BaseModel = self.quick_evidence_program(
                query=query or "",
                title=title or "",
                abstract=abstract or "",
                max_sentences=max(1, min(max_sentences, 10))
            )
            sentences: List[str] = getattr(output, "sentences", [])  # type: ignore[attr-defined]
            cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
            if not cleaned:
                # Heuristic fallback if LLM returns no sentences
                fallback = self._fallback_sentences(query=query, title=title, abstract=abstract, max_sentences=max_sentences)
                if fallback:
                    log.info("Quick evidences: using heuristic fallback sentences")
                return fallback[:max_sentences] if fallback else []
            return cleaned[:max_sentences]
        except Exception as e:
            log.warning(f"Quick evidences extraction failed: {e}")
            # Final fallback on exception
            return self._fallback_sentences(query=query, title=title, abstract=abstract, max_sentences=max_sentences)

    async def quick_evidences_for_paper(
        self,
        query: str,
        paper: Dict[str, Any],
        max_sentences: int = 5
    ) -> List[str]:
        """
        Convenience wrapper to extract relevant sentences for a paper dict.

        Args:
            query: Original user query.
            paper: Paper dict with 'title' and 'abstract'.
            max_sentences: Maximum sentences to extract.

        Returns:
            List of relevant sentences.
        """
        return await self.quick_evidences(
            query=query,
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            max_sentences=max_sentences
        )

    async def _get_paper_chunks(self, paper_id: str, query: Optional[str] = None, max_chunks: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for a specific paper from Elasticsearch.
        Uses the search service's optimized chunk retrieval.

        Args:
            paper_id: Paper identifier
            query: Optional query to rank chunks by relevance
            max_chunks: Maximum chunks to retrieve

        Returns:
            List of chunk documents
        """
        try:
            # Use the search service's chunk retrieval method
            chunks_by_paper = self.es_service.get_chunks_for_papers(
                paper_ids=[paper_id],
                query=query,
                max_chunks_per_paper=max_chunks
            )

            chunks = chunks_by_paper.get(paper_id, [])
            log.debug(f"Retrieved {len(chunks)} chunks for paper {paper_id}")
            return chunks

        except Exception as e:
            log.error(f"Failed to retrieve chunks for paper {paper_id}: {e}")
            return []

    async def get_full_paper_text(self, paper_id: str, max_chars: int = 20000) -> str:
        """
        Assemble the full paper text by concatenating chunk_text in order.

        Args:
            paper_id: Paper identifier
            max_chars: Maximum number of characters to return from the assembled text

        Returns:
            Concatenated paper content (possibly truncated to max_chars)
        """
        try:
            # Get all chunks for the paper (no query, so they come in order)
            chunks = await self._get_paper_chunks(paper_id, query=None, max_chunks=1000)
            if not chunks:
                return ""
            # Ensure chunks are ordered by chunk_index
            chunks_sorted = sorted(chunks, key=lambda c: c.get('chunk_index', 0))
            texts = [c.get('chunk_text', '') for c in chunks_sorted if c.get('chunk_text')]
            full_text = "\n".join(texts)
            if max_chars and len(full_text) > max_chars:
                return full_text[:max_chars]
            return full_text
        except Exception as e:
            log.warning(f"Failed to assemble full paper text for {paper_id}: {e}")
            return ""

    async def extract_evidence_from_chunks(
        self,
        paper_id: str,
        query: str,
        max_sentences: int = 5
    ) -> List[str]:
        """
        Extract evidence sentences directly from the most relevant chunks.
        This is more efficient than processing the full paper text.

        Args:
            paper_id: Paper identifier
            query: User query to find relevant evidence
            max_sentences: Maximum evidence sentences to return

        Returns:
            List of relevant sentences from chunks
        """
        try:
            # Get the most relevant chunks for this query
            chunks = await self._get_paper_chunks(paper_id, query=query, max_chunks=5)
            if not chunks:
                return []

            # Combine chunk texts
            combined_text = "\n\n".join([c.get('chunk_text', '') for c in chunks if c.get('chunk_text')])

            if not combined_text:
                return []

            # Extract sentences using the LLM
            output: BaseModel = self.full_evidence_program(
                query=query or "",
                content=combined_text[:10000],  # Limit to 10k chars for efficiency
                max_sentences=max(1, min(max_sentences, 10))
            )
            sentences: List[str] = getattr(output, "sentences", [])  # type: ignore[attr-defined]
            cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
            return cleaned[:max_sentences]

        except Exception as e:
            log.warning(f"Failed to extract evidence from chunks for {paper_id}: {e}")
            return []

    async def full_evidence(
        self,
        query: str,
        paper_id: str,
        max_sentences: int = 5,
        max_chars: int = 20000
    ) -> List[str]:
        """
        Extract relevant evidence sentences from the full paper content using the LLM.

        Args:
            query: User query guiding relevance
            paper_id: Paper identifier to retrieve full content
            max_sentences: Maximum number of evidence sentences to return
            max_chars: Character limit for the paper content sent to the LLM

        Returns:
            List of relevant sentences from the full paper
        """
        try:
            content = await self.get_full_paper_text(paper_id, max_chars=max_chars)
            if not content:
                return []
            output: BaseModel = self.full_evidence_program(
                query=query or "",
                content=content,
                max_sentences=max(1, min(max_sentences, 15))
            )
            sentences: List[str] = getattr(output, "sentences", [])  # type: ignore[attr-defined]
            cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
            return cleaned[:max_sentences]
        except Exception as e:
            log.warning(f"Full evidence extraction failed for {paper_id}: {e}")
            return []

    def _fallback_sentences(
        self,
        query: str,
        title: str,
        abstract: str,
        max_sentences: int = 3
    ) -> List[str]:
        """
        Heuristic fallback: pick sentences from title/abstract containing query keywords.

        Args:
            query: user query
            title: paper title
            abstract: paper abstract
            max_sentences: cap on sentences returned

        Returns:
            List of sentences
        """
        try:
            import re
            text = (title or "").strip()
            if abstract:
                text = (text + ". " + abstract.strip()).strip()
            # Split into sentences (simple heuristic)
            parts = re.split(r"(?<=[.!?])\s+", text)
            keywords = [w.lower() for w in (query or "").split() if len(w) > 2]
            scored: List[tuple[float, str]] = []
            for s in parts:
                lower = s.lower()
                score = sum(1 for k in keywords if k in lower)
                if score > 0:
                    scored.append((float(score), s.strip()))
            # If none matched, take first sentences from abstract
            if not scored:
                fallback = [p.strip() for p in parts if p.strip()]
                return fallback[:max_sentences]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored[:max_sentences]]
        except Exception:
            # As last resort, return first non-empty sentence from abstract or title
            base = abstract or title or ""
            return [base.strip()][:1] if base.strip() else []

    def _create_paper_without_evidence(self, paper: Dict[str, Any]) -> PaperWithEvidence:
        """
        Create a PaperWithEvidence object without evidence sentences.

        Args:
            paper: Paper dictionary

        Returns:
            PaperWithEvidence with empty evidence_sentences
        """
        return PaperWithEvidence(
            paper_id=paper.get('paper_id', ''),
            title=paper.get('title', 'Untitled'),
            authors=paper.get('authors', []),
            abstract=paper.get('abstract', ''),
            categories=paper.get('categories', []),
            publish_date=paper.get('publish_date'),
            elasticsearch_score=paper.get('score', 0.0),
            evidence_sentences=[]
        )
