"""
Query analysis module using Gemini Flash 2.5 for understanding user intent.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage
from llama_index.core.program import LLMTextCompletionProgram

from .config import llama_config

logger = logging.getLogger(__name__)


# Pydantic models for LlamaIndex structured output
class QueryAnalysisOutput(BaseModel):
    """Structured output for query analysis"""
    keywords: List[str] = Field(description="Main keywords extracted from the query")
    paper_titles: List[str] = Field(default_factory=list, description="Specific paper titles mentioned")
    authors: List[str] = Field(default_factory=list, description="Author names mentioned")
    date_from: Optional[str] = Field(None, description="Start date if mentioned")
    date_to: Optional[str] = Field(None, description="End date if mentioned")
    categories: List[str] = Field(default_factory=list, description="Research categories")
    query_type: str = Field(default="broad_search", description="Type of query: broad_search, specific_paper, author_search")
    search_strategy: str = Field(default="Standard keyword search", description="Strategy for searching")

class ResultEvaluationOutput(BaseModel):
    """Structured output for result evaluation"""
    quality_score: float = Field(description="Quality score from 0.0 to 1.0")
    has_sufficient_results: bool = Field(description="Whether we have enough results")
    needs_refinement: bool = Field(description="Whether search needs refinement")
    refinement_strategy: Optional[str] = Field(None, description="Strategy for refinement if needed")

# Dataclass versions for internal use
@dataclass
class AnalyzedQuery:
    """Structured representation of an analyzed query"""
    original_query: str
    keywords: List[str]
    paper_titles: List[str]
    authors: List[str]
    date_range: Dict[str, Optional[str]]
    categories: List[str]
    query_type: str  # broad_search, specific_paper, author_search
    search_strategy: str


@dataclass
class ResultEvaluation:
    """Evaluation of search results quality"""
    quality_score: float
    has_sufficient_results: bool
    needs_refinement: bool
    refinement_strategy: Optional[str]


class QueryAnalyzer:
    """
    Analyzes user queries using Gemini to extract intent and parameters.
    """

    def __init__(self, llm: Optional[Gemini] = None):
        """
        Initialize the query analyzer.

        Args:
            llm: Optional Gemini LLM instance
        """
        if llm is None:
            self.llm = Gemini(
                model=llama_config.gemini_model,
                api_key=llama_config.gemini_api_key,
                temperature=0.1,  # Low temperature for consistent analysis
                timeout=llama_config.llm_timeout
            )
        else:
            self.llm = llm

        # Create structured output programs
        self.query_program = LLMTextCompletionProgram.from_defaults(
            output_cls=QueryAnalysisOutput,
            prompt_template_str=(
                "Analyze the following research paper search query and extract key information:\n"
                "Query: {query}\n\n"
                "Extract keywords, authors, paper titles, date ranges, categories, query type, and search strategy."
            ),
            llm=self.llm
        )

        self.evaluation_program = LLMTextCompletionProgram.from_defaults(
            output_cls=ResultEvaluationOutput,
            prompt_template_str=(
                "Evaluate the quality of search results for this query: {query}\n\n"
                "Results summary: {results_summary}\n\n"
                "Assess quality score (0.0-1.0), whether we have sufficient results, "
                "and if refinement is needed with strategy."
            ),
            llm=self.llm
        )

    async def analyze_query(self, query: str) -> AnalyzedQuery:
        """
        Analyze a user query to extract search parameters and intent.

        Args:
            query: User's search query

        Returns:
            AnalyzedQuery with extracted information
        """
        try:
            logger.info(f"Analyzing query: {query}")

            # Use structured output program
            analysis: QueryAnalysisOutput = self.query_program(query=query)

            # Create structured result
            result = AnalyzedQuery(
                original_query=query,
                keywords=analysis.keywords,
                paper_titles=analysis.paper_titles,
                authors=analysis.authors,
                date_range={"from": analysis.date_from, "to": analysis.date_to},
                categories=analysis.categories,
                query_type=analysis.query_type,
                search_strategy=analysis.search_strategy
            )

            logger.info(f"Query analysis complete: type={result.query_type}")
            return result

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return basic analysis on error
            safe_keywords = query.split()[:5] if query else ["search"]
            return AnalyzedQuery(
                original_query=query or "search",
                keywords=safe_keywords,
                paper_titles=[],
                authors=[],
                date_range={"from": None, "to": None},
                categories=[],
                query_type="broad_search",
                search_strategy="Basic keyword search"
            )

    async def evaluate_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> ResultEvaluation:
        """
        Evaluate the quality of search results.

        Args:
            query: Original search query
            results: List of search results

        Returns:
            ResultEvaluation with quality assessment
        """
        try:
            logger.info(f"Evaluating {len(results)} results for query: {query}")

            # Create simple summary of results
            results_summary = f"Found {len(results)} results. "
            if results:
                top_titles = [r.get('title', 'Untitled')[:50] for r in results[:3]]
                results_summary += f"Top results: {'; '.join(top_titles)}"

            # Use structured output program
            evaluation: ResultEvaluationOutput = self.evaluation_program(
                query=query,
                results_summary=results_summary
            )

            result = ResultEvaluation(
                quality_score=evaluation.quality_score,
                has_sufficient_results=evaluation.has_sufficient_results,
                needs_refinement=evaluation.needs_refinement,
                refinement_strategy=evaluation.refinement_strategy
            )

            logger.info(f"Evaluation complete: score={result.quality_score}, needs_refinement={result.needs_refinement}")
            return result

        except Exception as e:
            logger.error(f"Result evaluation failed: {e}")
            # Return basic evaluation on error - conservative approach
            avg_score = sum(r.get("score", 0) for r in results[:5]) / min(5, len(results)) if results else 0
            return ResultEvaluation(
                quality_score=min(avg_score, 1.0),
                has_sufficient_results=len(results) >= 5,
                needs_refinement=len(results) < 3,
                refinement_strategy="Try broader keywords" if len(results) < 3 else None
            )

    # Helper methods removed - now using LlamaIndex structured output