"""
Main LlamaIndex ReAct agent for intelligent paper search.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool

from .config import llama_config
from .tools import PaperSearchTool
from .query_analyzer import QueryAnalyzer, AnalyzedQuery
from .response_builder import ResponseBuilder, FormattedResponse
from .evidence_extractor import EvidenceExtractor

logger = logging.getLogger(__name__)


class PaperSearchAgent:
    """
    Intelligent paper search agent using LlamaIndex ReAct framework.

    This agent:
    1. Analyzes user queries to understand intent
    2. Executes iterative searches with refinement
    3. Evaluates result quality
    4. Generates comprehensive responses
    """

    def __init__(
        self,
        es_service=None,
        llm: Optional[Gemini] = None
    ):
        """
        Initialize the paper search agent.

        Args:
            es_service: Optional ElasticsearchSearchService instance
            llm: Optional Gemini LLM instance
        """
        # Initialize LLM
        if llm is None:
            self.llm = Gemini(
                model=llama_config.gemini_model,
                api_key=llama_config.gemini_api_key,
                temperature=0.2,
                timeout=llama_config.llm_timeout
            )
        else:
            self.llm = llm

        # Initialize components
        self.search_tool = PaperSearchTool(es_service)
        self.query_analyzer = QueryAnalyzer(self.llm)

        # Initialize evidence extractor
        self.evidence_extractor = EvidenceExtractor(es_service)

        # Initialize response builder with evidence extractor
        self.response_builder = ResponseBuilder(evidence_extractor=self.evidence_extractor)

        # Create tools for the agent
        self.tools = self._create_agent_tools()

        # Initialize ReAct agent with enhanced system prompt
        from .config import AGENT_SYSTEM_PROMPT

        self.agent = ReActAgent(
            tools=self.tools,
            llm=self.llm,
            verbose=llama_config.verbose,
            system_prompt=AGENT_SYSTEM_PROMPT
        )

    def _create_agent_tools(self) -> List[FunctionTool]:
        """Create tools for the ReAct agent"""
        tools = []

        # Paper search tool
        search_tool = FunctionTool.from_defaults(
            async_fn=self._search_papers_with_analysis,
            name="search_papers",
            description=(
                "Search for academic papers using Elasticsearch. "
                "This tool automatically analyzes the query and applies appropriate filters. "
                "Returns structured search results with relevance scores."
            )
        )
        tools.append(search_tool)

        # Paper details tool
        detail_tool = self.search_tool.create_detail_tool()
        tools.append(detail_tool)

        # Refine search tool with detailed strategies
        refine_tool = FunctionTool.from_defaults(
            async_fn=self._refine_search,
            name="refine_search",
            description=(
                "Refine a previous search using intelligent strategies:\n"
                "- PARAPHRASE: Use alternative terms (e.g., 'RAG' → 'retrieval augmented generation')\n"
                "- FILTER: Apply date/author/category filters for precision\n"
                "- MODE_CHANGE: Switch between hybrid/semantic/bm25/title_only modes\n"
                "- BROADEN: Use more general terms for broader coverage\n"
                "- NARROW: Add constraints for more specific results\n"
                "Specify strategy like: 'PARAPHRASE: Try \"neural retrieval\" for better coverage'"
            )
        )
        tools.append(refine_tool)

        return tools

    async def _search_papers_with_analysis(
        self,
        query: str,
        override_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search papers with automatic query analysis.

        Args:
            query: Search query
            override_mode: Optional override for search mode

        Returns:
            Search results dictionary
        """
        try:
            # Analyze query if enabled
            if llama_config.enable_query_analysis:
                analysis = await self.query_analyzer.analyze_query(query)
                logger.info(f"Query analysis: type={analysis.query_type}, keywords={analysis.keywords}")
            else:
                analysis = AnalyzedQuery(
                    original_query=query,
                    keywords=[query],
                    paper_titles=[],
                    authors=[],
                    date_range={"from": None, "to": None},
                    categories=[],
                    query_type="broad_search",
                    search_strategy="Standard search"
                )

            # Determine search parameters based on analysis
            search_params = self._determine_search_params(analysis, override_mode)

            # Execute search
            results = self.search_tool.search_papers(**search_params)

            # Add analysis metadata to results
            results["analysis"] = {
                "query_type": analysis.query_type,
                "keywords": analysis.keywords,
                "search_strategy": analysis.search_strategy
            }

            return results

        except Exception as e:
            logger.error(f"Search with analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "papers": []
            }

    async def _refine_search(
        self,
        original_query: str,
        refinement_strategy: str
    ) -> Dict[str, Any]:
        """
        Refine a search based on evaluation feedback with intelligent strategies.

        Args:
            original_query: Original search query
            refinement_strategy: Strategy for refinement

        Returns:
            Refined search results
        """
        logger.info(f"Refining search: {refinement_strategy}")

        # Import refinement strategies
        from .config import REFINEMENT_STRATEGIES

        # Parse refinement strategy
        override_mode = None
        modified_query = original_query
        search_params = {}

        strategy_lower = refinement_strategy.lower()

        # Handle PARAPHRASE strategies
        if "paraphrase" in strategy_lower:
            modified_query = self._paraphrase_query(original_query, refinement_strategy)
            logger.info(f"Paraphrased query: '{original_query}' → '{modified_query}'")

        # Handle FILTER strategies
        elif "filter" in strategy_lower:
            search_params = self._apply_filters(refinement_strategy)
            logger.info(f"Applied filters: {search_params}")

        # Handle MODE_CHANGE strategies
        elif "mode_change" in strategy_lower or "semantic" in strategy_lower:
            if "semantic" in strategy_lower:
                override_mode = "semantic"
            elif "bm25" in strategy_lower:
                override_mode = "bm25"
            elif "title" in strategy_lower:
                override_mode = "title_only"
            elif "hybrid" in strategy_lower:
                override_mode = "hybrid"
            logger.info(f"Changed search mode to: {override_mode}")

        # Handle BROADEN strategies
        elif "broaden" in strategy_lower or "broader" in strategy_lower:
            # Use more general keywords
            words = original_query.split()
            modified_query = " ".join(words[:min(3, len(words))])
            override_mode = "semantic"  # Semantic search is better for broad queries
            logger.info(f"Broadened query: '{original_query}' → '{modified_query}'")

        # Handle NARROW strategies
        elif "narrow" in strategy_lower:
            # Add more specific terms or switch to exact matching
            override_mode = "bm25"  # BM25 is better for specific term matching
            logger.info(f"Narrowed search with BM25 mode")

        # Execute refined search with parameters
        return await self._search_papers_with_analysis_params(
            modified_query, override_mode, search_params
        )

    def _paraphrase_query(self, original_query: str, strategy: str) -> str:
        """Generate paraphrased query based on strategy"""
        from .config import REFINEMENT_STRATEGIES

        query_lower = original_query.lower()
        paraphrases = REFINEMENT_STRATEGIES["PARAPHRASE"]

        # Check for common terms and their paraphrases
        for term, alternatives in paraphrases.items():
            if term in query_lower:
                # Use the first alternative that's not already in the query
                for alt in alternatives:
                    if alt.lower() not in query_lower:
                        return original_query.replace(term, alt)

        # Extract specific paraphrase from strategy if provided
        if "try " in strategy.lower():
            import re
            match = re.search(r"try ['\"]([^'\"]+)['\"]", strategy.lower())
            if match:
                return match.group(1)

        return original_query

    def _apply_filters(self, strategy: str) -> Dict[str, Any]:
        """Apply search filters based on strategy"""
        from .config import REFINEMENT_STRATEGIES

        filters = {}
        strategy_lower = strategy.lower()
        filter_config = REFINEMENT_STRATEGIES["FILTERS"]

        # Date filters
        if "recent" in strategy_lower:
            if "very recent" in strategy_lower or "2024" in strategy_lower:
                filters["date_from"] = filter_config["very_recent"]
            else:
                filters["date_from"] = filter_config["recent"]

        # Category filters
        if any(cat in strategy_lower for cat in ["nlp", "ai", "ml", "ir"]):
            filters["categories"] = filter_config["categories"]

        return filters

    async def _search_papers_with_analysis_params(
        self,
        query: str,
        override_mode: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhanced search with additional parameters"""
        try:
            # Analyze query if enabled
            if llama_config.enable_query_analysis:
                analysis = await self.query_analyzer.analyze_query(query)
                logger.info(f"Query analysis: type={analysis.query_type}, keywords={analysis.keywords}")
            else:
                analysis = AnalyzedQuery(
                    original_query=query,
                    keywords=[query],
                    paper_titles=[],
                    authors=[],
                    date_range={"from": None, "to": None},
                    categories=[],
                    query_type="broad_search",
                    search_strategy="Standard search"
                )

            # Determine search parameters
            search_params = self._determine_search_params(analysis, override_mode)

            # Apply additional filters/parameters
            if extra_params:
                search_params.update(extra_params)

            # Execute search
            results = self.search_tool.search_papers(**search_params)

            # Add analysis metadata
            results["analysis"] = {
                "query_type": analysis.query_type,
                "keywords": analysis.keywords,
                "search_strategy": analysis.search_strategy,
                "applied_filters": extra_params or {}
            }

            return results

        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "papers": []
            }

    def _determine_search_params(
        self,
        analysis: AnalyzedQuery,
        override_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """Determine search parameters from query analysis"""
        params = {
            "query": " ".join(analysis.keywords) if analysis.keywords else analysis.original_query,
            "max_results": llama_config.max_results_per_search,
            "search_mode": override_mode or llama_config.default_search_mode
        }

        # Apply filters based on analysis
        if analysis.categories:
            params["categories"] = analysis.categories

        if analysis.authors:
            params["author"] = analysis.authors[0]  # Use first author

        if analysis.date_range.get("from"):
            params["date_from"] = analysis.date_range["from"]

        if analysis.date_range.get("to"):
            params["date_to"] = analysis.date_range["to"]

        # Adjust search mode based on query type
        if not override_mode:
            if analysis.query_type == "specific_paper":
                params["search_mode"] = "title_only"
            elif analysis.query_type == "author_search":
                params["search_mode"] = "bm25"  # Better for names

        return params

    async def search(self, query: str) -> FormattedResponse:
        """
        Main search method using the ReAct agent.

        Args:
            query: User search query

        Returns:
            FormattedResponse with search results
        """
        try:
            logger.info(f"Starting agent search for: {query}")

            # Initialize search state
            all_results = []
            iterations = 0

            # First search iteration
            initial_results = await self._search_papers_with_analysis(query)

            if initial_results.get("success") and initial_results.get("papers"):
                all_results.extend(initial_results["papers"])
                iterations = 1

                # Evaluate results if we have any
                if llama_config.enable_result_reranking and all_results:
                    evaluation = await self.query_analyzer.evaluate_results(
                        query,
                        all_results
                    )

                    # Perform refinement if needed
                    if evaluation.needs_refinement and iterations < llama_config.max_iterations:
                        logger.info(f"Refining search: {evaluation.refinement_strategy}")

                        refined_results = await self._refine_search(
                            query,
                            evaluation.refinement_strategy or "Try broader search"
                        )

                        if refined_results.get("success") and refined_results.get("papers"):
                            all_results.extend(refined_results["papers"])
                            iterations += 1

                            # One more evaluation and potential refinement
                            if iterations < llama_config.max_iterations:
                                evaluation2 = await self.query_analyzer.evaluate_results(
                                    query,
                                    all_results
                                )

                                if evaluation2.needs_refinement:
                                    final_results = await self._refine_search(
                                        query,
                                        evaluation2.refinement_strategy or "Try semantic search"
                                    )

                                    if final_results.get("success") and final_results.get("papers"):
                                        all_results.extend(final_results["papers"])
                                        iterations += 1

            # Build final response
            response = await self.response_builder.build_response(
                query=query,
                all_results=all_results,
                search_iterations=iterations
            )

            logger.info(
                f"Search completed: {response.total_found} papers found in {iterations} iterations"
            )
            return response

        except Exception as e:
            logger.error(f"Agent search failed: {e}")
            return FormattedResponse(
                success=False,
                query=query,
                summary="Search failed",
                papers=[],
                total_found=0,
                search_iterations=0,
                response_text="",
                error=str(e)
            )

    # Chat interface not supported by ReActAgent
    # Use the search() method for all queries