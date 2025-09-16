"""
Web Search Service for supplementary paper search results.
Uses web search to find papers not in the local ES index.
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from urllib.parse import quote_plus

from backend.config import config
from backend.services import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """Result from web search"""
    title: str
    url: str
    snippet: str
    source: str
    score: float = 0.5


class WebSearchService:
    """Service for supplementary web search to enhance paper discovery"""

    def __init__(self):
        """Initialize web search service"""
        self.session = None
        self.timeout = config.WEB_SEARCH_TIMEOUT_SECONDS

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_academic_papers(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search for academic papers using web search.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        logger.info(f"Web searching for academic papers: {query}")

        if not config.ENABLE_WEB_SEARCH:
            logger.info("Web search is disabled")
            return []

        try:
            if not self.session:
                async with self:
                    return await self._perform_search(query, max_results)
            else:
                return await self._perform_search(query, max_results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _perform_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform the actual web search"""
        results = []

        # Try multiple search strategies
        search_strategies = [
            ("scholar", self._search_google_scholar),
            ("arxiv", self._search_arxiv),
            ("duckduckgo", self._search_duckduckgo_academic),
        ]

        for strategy_name, search_func in search_strategies:
            try:
                strategy_results = await search_func(query, max_results // len(search_strategies))
                results.extend(strategy_results)
                logger.info(f"{strategy_name} found {len(strategy_results)} results")
            except Exception as e:
                logger.warning(f"{strategy_name} search failed: {e}")

        # Remove duplicates and limit results
        unique_results = self._deduplicate_results(results)
        return unique_results[:max_results]

    async def _search_google_scholar(self, query: str, max_results: int) -> List[SearchResult]:
        """Search Google Scholar (simplified scraping approach)"""
        try:
            # Use DuckDuckGo to search Scholar
            scholar_query = f"site:scholar.google.com {query}"
            return await self._search_duckduckgo(scholar_query, max_results, source="Google Scholar")
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return []

    async def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """Search ArXiv using their API"""
        try:
            # ArXiv API endpoint
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            url = f"{base_url}?" + "&".join([f"{k}={quote_plus(str(v))}" for k, v in params.items()])

            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_arxiv_xml(content)
                else:
                    logger.warning(f"ArXiv API returned status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []

    async def _search_duckduckgo_academic(self, query: str, max_results: int) -> List[SearchResult]:
        """Search academic sources using DuckDuckGo"""
        # Add academic-focused terms to the query
        academic_query = f"{query} academic paper research"
        return await self._search_duckduckgo(academic_query, max_results, source="Academic Web Search")

    async def _search_duckduckgo(self, query: str, max_results: int, source: str = "DuckDuckGo") -> List[SearchResult]:
        """Search using DuckDuckGo instant answer API"""
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_duckduckgo_results(data, source, max_results)
                else:
                    logger.warning(f"DuckDuckGo API returned status {response.status}")
                    return []

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _parse_arxiv_xml(self, xml_content: str) -> List[SearchResult]:
        """Parse ArXiv API XML response"""
        results = []
        try:
            import xml.etree.ElementTree as ET

            # Parse XML
            root = ET.fromstring(xml_content)

            # ArXiv namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}

            entries = root.findall('.//atom:entry', ns)

            for entry in entries:
                try:
                    # Extract information
                    title_elem = entry.find('atom:title', ns)
                    summary_elem = entry.find('atom:summary', ns)
                    authors_elems = entry.findall('atom:author/atom:name', ns)
                    published_elem = entry.find('atom:published', ns)
                    id_elem = entry.find('atom:id', ns)

                    if title_elem is not None and summary_elem is not None:
                        title = title_elem.text.strip().replace('\n', ' ')
                        abstract = summary_elem.text.strip().replace('\n', ' ')
                        authors = [author.text for author in authors_elems]

                        # Extract paper ID from ArXiv URL
                        paper_id = ""
                        if id_elem is not None:
                            arxiv_url = id_elem.text
                            paper_id = re.search(r'abs/([^/]+)$', arxiv_url)
                            if paper_id:
                                paper_id = f"arxiv:{paper_id.group(1)}"

                        # Extract publication year
                        publish_date = None
                        if published_elem is not None:
                            date_str = published_elem.text
                            year_match = re.search(r'(\d{4})', date_str)
                            if year_match:
                                publish_date = year_match.group(1)

                        result = SearchResult(
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            score=0.7,  # Default web search score
                            categories=["cs"],  # Default for ArXiv CS
                            publish_date=publish_date
                        )
                        results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to parse ArXiv entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse ArXiv XML: {e}")

        return results

    def _parse_duckduckgo_results(self, data: Dict[str, Any], source: str, max_results: int) -> List[SearchResult]:
        """Parse DuckDuckGo JSON response"""
        results = []

        try:
            # Check different result types
            related_topics = data.get('RelatedTopics', [])

            for i, topic in enumerate(related_topics[:max_results]):
                if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                    text = topic['Text']
                    url = topic['FirstURL']

                    # Try to extract title and abstract from text
                    parts = text.split(' - ', 1)
                    if len(parts) >= 2:
                        title = parts[0].strip()
                        abstract = parts[1].strip()
                    else:
                        title = text[:100] + "..." if len(text) > 100 else text
                        abstract = text

                    # Generate a simple paper ID
                    paper_id = f"web:{hash(url) % 1000000}"

                    result = SearchResult(
                        paper_id=paper_id,
                        title=title,
                        authors=[],  # Not available from DuckDuckGo
                        abstract=abstract,
                        score=0.5 - (i * 0.05),  # Decreasing score by position
                        categories=[],
                        publish_date=None
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo results: {e}")

        return results

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on title similarity"""
        unique_results = []
        seen_titles = set()

        for result in results:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', result.title.lower())
            normalized_title = ' '.join(normalized_title.split())

            # Check for similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_similar(normalized_title, seen_title):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(normalized_title)

        return unique_results

    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar using simple word overlap"""
        words1 = set(title1.split())
        words2 = set(title2.split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    async def enhance_papers_with_web_context(self, papers: List[SearchResult],
                                            original_query: str) -> List[SearchResult]:
        """Enhance existing papers with additional web context"""
        if not config.ENABLE_WEB_SEARCH or not papers:
            return papers

        # For each paper, search for additional context
        enhanced_papers = []

        for paper in papers:
            try:
                # Search for additional information about this specific paper
                context_query = f'"{paper.title}" author citation'
                context_results = await self.search_academic_papers(context_query, 3)

                # Add any relevant context to the paper (this is a simplified approach)
                enhanced_paper = paper
                if context_results:
                    # Could enhance with citation counts, related work, etc.
                    # For now, just keep original paper
                    pass

                enhanced_papers.append(enhanced_paper)

            except Exception as e:
                logger.warning(f"Failed to enhance paper {paper.paper_id}: {e}")
                enhanced_papers.append(paper)

        return enhanced_papers

    def health_check(self) -> Dict[str, Any]:
        """Check health of web search service"""
        return {
            "enabled": config.ENABLE_WEB_SEARCH,
            "timeout": self.timeout,
            "status": "healthy" if config.ENABLE_WEB_SEARCH else "disabled"
        }