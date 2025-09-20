"""
LlamaIndex tools for paper search, wrapping Elasticsearch service.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from backend.services import ElasticsearchSearchService
from backend.config import config
from logs import log



class PaperSearchInput(BaseModel):
    """Input model for paper search tool"""
    query: str = Field(description="Search query text")
    max_results: int = Field(default=20, description="Maximum number of results")
    search_mode: str = Field(
        default="hybrid",
        description="Search mode: hybrid, semantic, bm25, title_only"
    )
    categories: Optional[List[str]] = Field(default=None, description="Filter by categories")
    date_from: Optional[str] = Field(default=None, description="Filter by start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(default=None, description="Filter by end date (YYYY-MM-DD)")
    author: Optional[str] = Field(default=None, description="Filter by author name")


class PaperSearchTool:
    """
    Elasticsearch-based paper search tool for LlamaIndex agents.
    """

    def __init__(self, es_service: Optional[ElasticsearchSearchService] = None):
        """
        Initialize the search tool.

        Args:
            es_service: Optional ElasticsearchSearchService instance
        """
        if es_service is None:
            self.es_service = ElasticsearchSearchService(
                es_host=config.ES_HOST,
                index_name=config.ES_INDEX_NAME,
                bge_model=config.BGE_MODEL_NAME,
                bge_cache_dir=config.BGE_CACHE_DIR
            )
        else:
            self.es_service = es_service

    async def search_papers(
        self,
        query: str,
        max_results: int = 50,
        search_mode: str = "hybrid",
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        author: Optional[str] = None,
        include_chunks: bool = False  # Default to False for LlamaIndex agent
    ) -> Dict[str, Any]:
        """
        Search for papers using Elasticsearch.

        Args:
            query: Search query text
            max_results: Maximum number of results
            search_mode: Search mode (hybrid, semantic, bm25, title_only)
            categories: Optional category filter
            date_from: Optional start date filter
            date_to: Optional end date filter
            author: Optional author filter

        Returns:
            Dictionary containing search results and metadata
        """
        try:
            log.info(f"Searching papers: query='{query}', mode={search_mode}, max_results={max_results}")
            log.debug(f"Additional params - categories: {categories}, date_from: {date_from}, date_to: {date_to}, author: {author}")

            # Execute search
            results = self.es_service.search(
                query=query,
                max_results=max_results,
                search_mode=search_mode,
                categories=categories,
                date_from=date_from,
                date_to=date_to,
                author=author,
                include_chunks=include_chunks
            )

            log.info(f"ES service returned {len(results)} results")

            # Format results for agent consumption
            papers = []
            for r in results:
                papers.append({
                    "paper_id": r.paper_id,
                    "title": r.title,
                    "authors": r.authors,
                    "abstract": r.abstract[:],
                    "score": r.score,
                    "categories": r.categories,
                    "publish_date": r.publish_date,
                    "word_count": r.word_count
                })

            response = {
                "success": True,
                "query": query,
                "total_results": len(papers),
                "papers": papers,
                "search_mode": search_mode,
                "filters_applied": {
                    "categories": categories,
                    "date_range": {"from": date_from, "to": date_to},
                    "author": author
                }
            }

            log.info(f"Search completed: found {len(papers)} papers")
            return response

        except Exception as e:
            log.error(f"Search failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "papers": []
            }

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Paper identifier

        Returns:
            Dictionary containing paper details
        """
        try:
            log.info(f"Getting details for paper: {paper_id}")

            details = self.es_service.get_paper_details(paper_id)

            if details:
                return {
                    "success": True,
                    "paper": {
                        "paper_id": details.paper_id,
                        "title": details.title,
                        "authors": details.authors,
                        "abstract": details.abstract,
                        "content": details.content[:],
                        "categories": details.categories,
                        "publish_date": details.publish_date,
                        "word_count": details.word_count,
                        "chunk_count": details.chunk_count
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Paper {paper_id} not found"
                }

        except Exception as e:
            log.error(f"Failed to get paper details: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def create_llama_tool(self) -> FunctionTool:
        """
        Create a LlamaIndex FunctionTool from this search tool.

        Returns:
            FunctionTool instance for use with LlamaIndex agents
        """
        return FunctionTool.from_defaults(
            async_fn=self.search_papers,
            name="search_papers",
            description=(
                "Search for academic papers using Elasticsearch. "
                "Supports keyword search, semantic search, and various filters. "
                "Use this tool to find relevant papers based on user queries."
            )
        )

    def create_detail_tool(self) -> FunctionTool:
        """
        Create a LlamaIndex FunctionTool for getting paper details.

        Returns:
            FunctionTool instance for paper detail retrieval
        """
        return FunctionTool.from_defaults(
            async_fn=self.get_paper_details,
            name="get_paper_details",
            description=(
                "Get detailed information about a specific paper by its ID. "
                "Use this when you need more information about a particular paper."
            )
        )


def create_search_tools(es_service: Optional[ElasticsearchSearchService] = None) -> List[FunctionTool]:
    """
    Create all paper search tools for LlamaIndex agent.

    Args:
        es_service: Optional ElasticsearchSearchService instance

    Returns:
        List of FunctionTool instances
    """
    search_tool = PaperSearchTool(es_service)
    return [
        search_tool.create_llama_tool(),
        search_tool.create_detail_tool()
    ]