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
    """Paper with extracted evidence chunks"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    publish_date: Optional[str]
    elasticsearch_score: float
    evidence_chunks: List[EvidenceChunk]


class EvidenceExtractor:
    """
    Extracts relevant text chunks from papers as evidence for search queries.
    """

    def __init__(
        self,
        es_service: ElasticsearchSearchService,
        embedder: Optional[BGEEmbedder] = None,
        max_chunks_per_paper: int = 3,
        min_relevance_score: float = 0.6
    ):
        """
        Initialize evidence extractor.

        Args:
            es_service: Elasticsearch search service
            embedder: Optional BGE embedder (will use es_service.embedder if None)
            max_chunks_per_paper: Maximum evidence chunks per paper
            min_relevance_score: Minimum relevance score for chunks
        """
        self.es_service = es_service
        self.embedder = embedder or es_service.embedder
        self.max_chunks_per_paper = max_chunks_per_paper
        self.min_relevance_score = min_relevance_score

    async def extract_evidence(
        self,
        papers: List[Dict[str, Any]],
        query: str
    ) -> List[PaperWithEvidence]:
        """
        Extract evidence chunks for a list of papers.

        Args:
            papers: List of paper dictionaries from search results
            query: Original search query

        Returns:
            List of PaperWithEvidence objects
        """
        log.info(f"Extracting evidence for {len(papers)} papers with query: '{query}'")

        # Generate query embedding once
        query_embedding = self.embedder.encode(query)

        results = []
        for paper in papers:
            try:
                paper_with_evidence = await self._extract_paper_evidence(
                    paper, query, query_embedding
                )
                if paper_with_evidence:
                    results.append(paper_with_evidence)
            except Exception as e:
                log.warning(f"Failed to extract evidence for paper {paper.get('paper_id', 'unknown')}: {e}")
                # Still include paper without evidence chunks
                results.append(self._create_paper_without_evidence(paper))

        log.info(f"Evidence extraction completed for {len(results)} papers")
        return results

    async def _extract_paper_evidence(
        self,
        paper: Dict[str, Any],
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[PaperWithEvidence]:
        """
        Extract evidence chunks for a single paper.

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

        # Get paper details with all chunks
        paper_details = self.es_service.get_paper_details(paper_id)
        if not paper_details:
            log.warning(f"Could not retrieve details for paper {paper_id}")
            return self._create_paper_without_evidence(paper)

        # Get individual chunks for this paper
        chunks = await self._get_paper_chunks(paper_id)
        if not chunks:
            log.warning(f"No chunks found for paper {paper_id}")
            return self._create_paper_without_evidence(paper)

        # Score chunks against query
        evidence_chunks = self._score_and_select_chunks(chunks, query_embedding)

        return PaperWithEvidence(
            paper_id=paper_id,
            title=paper.get('title', 'Untitled'),
            authors=paper.get('authors', []),
            abstract=paper.get('abstract', ''),
            categories=paper.get('categories', []),
            publish_date=paper.get('publish_date'),
            elasticsearch_score=paper.get('score', 0.0),
            evidence_chunks=evidence_chunks
        )

    async def _get_paper_chunks(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific paper from Elasticsearch.

        Args:
            paper_id: Paper identifier

        Returns:
            List of chunk documents
        """
        try:
            search_body = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"paper_id": paper_id}},
                            {"term": {"doc_type": "chunk"}}
                        ]
                    }
                },
                "sort": [{"chunk_index": {"order": "asc"}}],
                "size": 1000,  # Get all chunks
                "_source": {
                    "includes": [
                        "chunk_index", "chunk_text", "chunk_start", "chunk_end",
                        "chunk_embedding"
                    ]
                }
            }

            response = self.es_service.indexer.es.search(
                index=self.es_service.indexer.index_name,
                body=search_body
            )

            chunks = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                chunks.append({
                    'chunk_index': source.get('chunk_index', 0),
                    'chunk_text': source.get('chunk_text', ''),
                    'chunk_start': source.get('chunk_start'),
                    'chunk_end': source.get('chunk_end'),
                    'chunk_embedding': source.get('chunk_embedding', [])
                })

            log.debug(f"Retrieved {len(chunks)} chunks for paper {paper_id}")
            return chunks

        except Exception as e:
            log.error(f"Failed to retrieve chunks for paper {paper_id}: {e}")
            return []

    def _score_and_select_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: np.ndarray
    ) -> List[EvidenceChunk]:
        """
        Score chunks against query and select the most relevant ones.

        Args:
            chunks: List of chunk documents
            query_embedding: Query embedding vector

        Returns:
            List of selected evidence chunks
        """
        scored_chunks = []

        for chunk in chunks:
            chunk_embedding = chunk.get('chunk_embedding', [])
            if not chunk_embedding or len(chunk_embedding) == 0:
                log.debug(f"Chunk {chunk.get('chunk_index')} missing embedding, skipping")
                continue

            try:
                # Convert to numpy array if needed
                if isinstance(chunk_embedding, list):
                    chunk_embedding = np.array(chunk_embedding)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)

                # Only include chunks above minimum threshold
                if similarity >= self.min_relevance_score:
                    evidence_chunk = EvidenceChunk(
                        chunk_index=chunk.get('chunk_index', 0),
                        chunk_text=chunk.get('chunk_text', ''),
                        relevance_score=float(similarity),
                        chunk_start=chunk.get('chunk_start'),
                        chunk_end=chunk.get('chunk_end')
                    )
                    scored_chunks.append(evidence_chunk)

            except Exception as e:
                log.warning(f"Failed to score chunk {chunk.get('chunk_index')}: {e}")
                continue

        # Sort by relevance score (highest first) and take top N
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        selected_chunks = scored_chunks[:self.max_chunks_per_paper]

        log.debug(f"Selected {len(selected_chunks)} evidence chunks from {len(chunks)} total chunks")
        return selected_chunks

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            log.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _create_paper_without_evidence(self, paper: Dict[str, Any]) -> PaperWithEvidence:
        """
        Create a PaperWithEvidence object without evidence chunks.

        Args:
            paper: Paper dictionary

        Returns:
            PaperWithEvidence with empty evidence_chunks
        """
        return PaperWithEvidence(
            paper_id=paper.get('paper_id', ''),
            title=paper.get('title', 'Untitled'),
            authors=paper.get('authors', []),
            abstract=paper.get('abstract', ''),
            categories=paper.get('categories', []),
            publish_date=paper.get('publish_date'),
            elasticsearch_score=paper.get('score', 0.0),
            evidence_chunks=[]
        )

    def set_extraction_params(
        self,
        max_chunks_per_paper: Optional[int] = None,
        min_relevance_score: Optional[float] = None
    ):
        """
        Update extraction parameters.

        Args:
            max_chunks_per_paper: Maximum chunks per paper
            min_relevance_score: Minimum relevance threshold
        """
        if max_chunks_per_paper is not None:
            self.max_chunks_per_paper = max_chunks_per_paper

        if min_relevance_score is not None:
            self.min_relevance_score = min_relevance_score

        log.info(f"Updated extraction params: max_chunks={self.max_chunks_per_paper}, min_score={self.min_relevance_score}")