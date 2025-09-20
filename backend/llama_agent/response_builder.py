"""
Response builder module for formatting search results with evidence chunks.
"""

from logs import log
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import llama_config



@dataclass
class FormattedResponse:
    """Structured response for paper search with evidence chunks"""
    success: bool
    query: str
    papers: List[Dict[str, Any]]
    total_found: int
    search_iterations: int
    error: Optional[str] = None


class ResponseBuilder:
    """
    Builds formatted responses from search results with evidence extraction.
    """

    def __init__(self, evidence_extractor=None):
        """
        Initialize the response builder.

        Args:
            evidence_extractor: Optional evidence extractor instance
        """
        self.evidence_extractor = evidence_extractor

    async def build_response(
        self,
        query: str,
        all_results: List[Dict[str, Any]],
        search_iterations: int = 1
    ) -> FormattedResponse:
        """
        Build a formatted response from search results with evidence extraction.

        Args:
            query: Original user query
            all_results: All search results from iterations
            search_iterations: Number of search iterations performed

        Returns:
            FormattedResponse with formatted results and evidence chunks
        """
        try:
            log.info(f"Building response for {len(all_results)} results")

            # Deduplicate and rank results
            unique_papers = self._deduplicate_papers(all_results)
            ranked_papers = self._rank_papers(unique_papers)

            # Select top papers for response
            top_papers = ranked_papers[:llama_config.response_max_papers]

            # Extract evidence chunks if extractor is available
            formatted_papers = []
            if self.evidence_extractor and top_papers:
                log.info(f"Extracting evidence for {len(top_papers)} papers")
                papers_with_evidence = await self.evidence_extractor.extract_evidence(
                    top_papers, query
                )

                # Convert to response format with sentences
                for paper_evidence in papers_with_evidence:
                    formatted_paper = {
                        "paper_id": paper_evidence.paper_id,
                        "title": paper_evidence.title,
                        "authors": paper_evidence.authors,
                        "abstract": paper_evidence.abstract,
                        "categories": paper_evidence.categories,
                        "publish_date": paper_evidence.publish_date,
                        "elasticsearch_score": paper_evidence.elasticsearch_score,
                        "evidence_sentences": paper_evidence.evidence_sentences,
                    }
                    formatted_papers.append(formatted_paper)
            else:
                # No evidence extraction - return papers as-is but add empty evidence_sentences
                for paper in top_papers:
                    paper["evidence_sentences"] = []
                    formatted_papers.append(paper)

            return FormattedResponse(
                success=True,
                query=query,
                papers=formatted_papers,
                total_found=len(unique_papers),
                search_iterations=search_iterations
            )

        except Exception as e:
            log.error(f"Failed to build response: {e}")
            return FormattedResponse(
                success=False,
                query=query,
                papers=[],
                total_found=0,
                search_iterations=search_iterations,
                error=str(e)
            )


    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on paper_id"""
        seen = set()
        unique = []

        for paper in papers:
            paper_id = paper.get("paper_id")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                unique.append(paper)

        return unique

    def _rank_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank papers by score and other factors"""
        # Sort by score (descending)
        return sorted(papers, key=lambda p: p.get("score", 0), reverse=True)

