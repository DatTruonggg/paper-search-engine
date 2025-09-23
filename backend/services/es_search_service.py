"""
Elasticsearch-based search service for paper search engine.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.es_indexer import ESIndexer
from backend.config import config

from logs import log

@dataclass
class SearchResult:
    """Search result data model"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    score: float
    categories: List[str]
    publish_date: Optional[str] = None
    word_count: Optional[int] = None
    has_images: Optional[bool] = False
    pdf_size: Optional[int] = None
    chunk_matches: Optional[List[Dict]] = None
    # Chunk-specific fields (populated when searching chunks)
    chunk_text: Optional[str] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    chunk_index: Optional[int] = None


@dataclass
class PaperDetails:
    """Detailed paper information"""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    categories: List[str]
    publish_date: Optional[str] = None
    word_count: Optional[int] = None
    chunk_count: Optional[int] = None
    has_images: Optional[bool] = False
    pdf_size: Optional[int] = None
    downloaded_at: Optional[str] = None
    indexed_at: Optional[str] = None
    markdown_path: Optional[str] = None
    pdf_path: Optional[str] = None
    minio_pdf_url: Optional[str] = None
    minio_markdown_url: Optional[str] = None


class ElasticsearchSearchService:
    """
    Service providing paper search and analytics on a chunk-based index.

    Uses BM25 and BGE embeddings over an Elasticsearch backend.
    """

    def __init__(
        self,
        es_host: str = None,
        index_name: str = None,
        bge_model: str = None,
        bge_cache_dir: str = None
    ):
        """
        Initialize the search service.

        Args:
            es_host: Elasticsearch host URL
            index_name: Name of the ES index
            bge_model: BGE model name
            bge_cache_dir: Cache directory for BGE model
        """
        log.info(f"Initializing Elasticsearch search service...")

        # Initialize BGE embedder
        self.embedder = BGEEmbedder(
            model_name=bge_model or config.BGE_MODEL_NAME,
            cache_dir=bge_cache_dir or config.BGE_CACHE_DIR,
        )

        # Initialize ES indexer
        self.indexer = ESIndexer(
            es_host=es_host or config.ES_HOST,
            index_name=index_name or config.ES_INDEX_NAME,
            embedding_dim=self.embedder.embedding_dim,
        )

        log.info("Search service initialized successfully")

    def search(
        self,
        query: str,
        max_results: int = 20,
        search_mode: str = "hybrid",
        categories: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        author: Optional[str] = None,
        include_chunks: bool = False
    ) -> List[SearchResult]:
        """
        Execute a paper search with hybrid, semantic, or fulltext scoring.
        Now searches only paper documents for efficiency, unless include_chunks is True.

        Args:
            query: Search query text.
            max_results: Maximum number of results to return.
            search_mode: One of "hybrid", "semantic", "fulltext".
            categories: Optional category filter.
            date_from: Optional ISO date lower bound (inclusive).
            date_to: Optional ISO date upper bound (inclusive).
            author: Optional author substring filter (case-insensitive).
            include_chunks: Whether to search in chunk_text (full paper content).

        Returns:
            List of SearchResult with normalized scores in [0, 1].
        """
        log.info(f"Searching for: '{query}' (mode: {search_mode}, include_chunks: {include_chunks})")

        # Generate query embedding for semantic search
        query_embedding = None
        if search_mode in ["hybrid", "semantic"]:
            query_embedding = self.embedder.encode(query)
            log.debug(f"Generated query embedding with shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'unknown'}")

        # Configure search based on mode
        search_fields = self._get_search_fields(search_mode, include_chunks)
        use_semantic = search_mode in ["hybrid", "semantic"]
        use_bm25 = search_mode in ["hybrid", "fulltext"]

        log.info(f"Search config - Fields: {search_fields}, Semantic: {use_semantic}, BM25: {use_bm25}")

        # Build filter to search only paper documents (unless chunks are requested)
        doc_type_filter = {"term": {"doc_type": "chunk" if include_chunks else "paper"}}

        # Perform search with doc_type filter
        raw_results = self.indexer.search(
            query=query if use_bm25 else None,
            query_embedding=query_embedding,
            size=max_results * 2,  # Get more for filtering
            search_fields=search_fields,
            use_semantic=use_semantic,
            use_bm25=use_bm25,
            filter_query=doc_type_filter
        )

        log.info(f"ES returned {len(raw_results)} raw results")

        # Convert to SearchResult objects
        results = []
        log.info(f"Processing {len(raw_results)} raw results...")
        log.info(f"Active filters - categories: {categories}, author: {author}, date_from: {date_from}, date_to: {date_to}")

        for i, hit in enumerate(raw_results):
            # Debug first few results
            if i < 3:
                log.info(f"Raw result {i}: paper_id={hit.get('paper_id', 'N/A')}, title={hit.get('title', 'N/A')[:50]}, score={hit.get('_score', 0)}")

            # Apply filters
            # if categories and not any(cat in hit.get('categories', []) for cat in categories):
            #     log.info(f"Filtered out {hit.get('paper_id', 'unknown')} - categories mismatch. Looking for {categories}, found {hit.get('categories', [])}")
            #     continue

            if author and author.lower() not in ' '.join(hit.get('authors', [])).lower():
                log.info(f"Filtered out {hit.get('paper_id', 'unknown')} - author mismatch. Looking for '{author}', found {hit.get('authors', [])}")
                continue

            # Check date range
            if date_from or date_to:
                pub_date = hit.get('publish_date')
                if pub_date:
                    if date_from and pub_date < date_from:
                        log.debug(f"Filtered out {hit.get('paper_id', 'unknown')} - date too old")
                        continue
                    if date_to and pub_date > date_to:
                        log.debug(f"Filtered out {hit.get('paper_id', 'unknown')} - date too recent")
                        continue

            # Create result - ensure paper_id is a string
            paper_id = str(hit.get('paper_id', '')) if hit.get('paper_id') is not None else ''
            title = hit.get('title', 'Untitled')

            # Log what we got for debugging
            if i < 5:
                log.info(f"Processing result {i}: paper_id='{paper_id}' (type: {type(hit.get('paper_id'))}), title='{title[:50] if title else 'None'}'")

            # Skip results without paper_id or title
            if not paper_id or paper_id == '':
                log.info(f"Skipping result {i} - missing paper_id. Raw value: {hit.get('paper_id')}, Keys available: {list(hit.keys())[:10]}")
                continue

            if not title or title == 'Untitled':
                log.info(f"Skipping result {i} - missing title for paper_id {paper_id}")
                continue

            result = SearchResult(
                paper_id=paper_id,
                title=title,
                authors=hit.get('authors', []),
                abstract=hit.get('abstract', ''),
                score=hit.get('_score', 0.0),
                categories=hit.get('categories', []),
                publish_date=hit.get('publish_date'),
                word_count=hit.get('word_count'),
                has_images=hit.get('has_images', False),
                pdf_size=hit.get('pdf_size'),
                # Populate chunk fields if this is a chunk document
                chunk_text=hit.get('chunk_text'),
                chunk_start=hit.get('chunk_start'),
                chunk_end=hit.get('chunk_end'),
                chunk_index=hit.get('chunk_index')
            )

            results.append(result)

            if i < 3:
                log.debug(f"Added result {i}: {result.paper_id} - {result.title[:50]}")

            if len(results) >= max_results:
                break

        # Normalize scores to 0-1 range
        results = self._normalize_scores(results, search_mode)

        log.info(f"Found {len(results)} results")
        return results

    def get_paper_details(self, paper_id: str) -> Optional[PaperDetails]:
        """
        Retrieve full paper details by stitching chunk documents.

        Args:
            paper_id: Target paper identifier.

        Returns:
            PaperDetails if found, otherwise None.
        """
        log.info(f"Getting details for paper: {paper_id}")

        # Search for all chunks of this paper
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
                    "excludes": ["*_embedding"]
                }
            }

            response = self.indexer.es.search(index=self.indexer.index_name, body=search_body)
            chunks = response['hits']['hits']

            if not chunks:
                log.warning(f"Paper not found: {paper_id}")
                return None

            # Also fetch the single paper document for authoritative metadata
            paper_search_body = {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"paper_id": paper_id}},
                            {"term": {"doc_type": "paper"}}
                        ]
                    }
                },
                "size": 1,
                "_source": {"excludes": ["*_embedding"]}
            }

            paper_doc_response = self.indexer.es.search(
                index=self.indexer.index_name,
                body=paper_search_body
            )

            paper_doc_source = None
            if paper_doc_response.get('hits', {}).get('hits'):
                paper_doc_source = paper_doc_response['hits']['hits'][0].get('_source', {})
                log.debug(
                    f"Found paper document for {paper_id} with fields: "
                    f"authors={bool(paper_doc_source.get('authors'))}, abstract={bool(paper_doc_source.get('abstract'))}"
                )

            # Get paper metadata from first chunk
            first_chunk = chunks[0]['_source']

            # Reconstruct full content from chunks
            content = ""
            for hit in chunks:
                chunk_data = hit['_source']
                content += chunk_data.get('chunk_text', '') + " "

            return PaperDetails(
                paper_id=first_chunk.get('paper_id', ''),
                title=first_chunk.get('title', 'Untitled'),
                # Prefer authors/abstract from the paper document if available
                authors=(paper_doc_source.get('authors') if paper_doc_source and paper_doc_source.get('authors') is not None else first_chunk.get('authors', [])),
                abstract=(paper_doc_source.get('abstract') if paper_doc_source and paper_doc_source.get('abstract') is not None else first_chunk.get('abstract', '')),
                content=content.strip(),
                categories=first_chunk.get('categories', []),
                publish_date=first_chunk.get('publish_date'),
                word_count=first_chunk.get('word_count'),
                chunk_count=len(chunks),
                has_images=first_chunk.get('has_images', False),
                pdf_size=first_chunk.get('pdf_size'),
                downloaded_at=first_chunk.get('downloaded_at'),
                indexed_at=first_chunk.get('indexed_at'),
                markdown_path=first_chunk.get('markdown_path'),
                pdf_path=first_chunk.get('pdf_path'),
                minio_pdf_url=first_chunk.get('minio_pdf_url'),
                minio_markdown_url=first_chunk.get('minio_markdown_url')
            )

        except Exception as e:
            log.error(f"Error getting paper details: {e}")
            return None

    def get_chunks_for_papers(
        self,
        paper_ids: List[str],
        query: Optional[str] = None,
        max_chunks_per_paper: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve chunks for specific papers, optionally ranked by relevance to a query.

        Args:
            paper_ids: List of paper IDs to get chunks for
            query: Optional query to rank chunks by relevance
            max_chunks_per_paper: Maximum chunks to return per paper

        Returns:
            Dictionary mapping paper_id to list of chunk documents
        """
        log.info(f"Getting chunks for {len(paper_ids)} papers")

        result = {}

        for paper_id in paper_ids:
            try:
                # Build query for this paper's chunks
                must_clauses = [
                    {"term": {"paper_id": paper_id}},
                    {"term": {"doc_type": "chunk"}}
                ]

                # If query provided, add relevance scoring
                if query:
                    search_body = {
                        "query": {
                            "bool": {
                                "must": must_clauses,
                                "should": [
                                    {
                                        "match": {
                                            "chunk_text": {
                                                "query": query,
                                                "boost": 1.0
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "sort": [{"_score": {"order": "desc"}}],
                        "size": max_chunks_per_paper
                    }
                else:
                    # No query, just get chunks in order
                    search_body = {
                        "query": {"bool": {"must": must_clauses}},
                        "sort": [{"chunk_index": {"order": "asc"}}],
                        "size": max_chunks_per_paper
                    }

                response = self.indexer.es.search(
                    index=self.indexer.index_name,
                    body=search_body
                )

                chunks = []
                for hit in response['hits']['hits']:
                    chunk = hit['_source']
                    chunk['_score'] = hit.get('_score', 0)
                    chunks.append(chunk)

                result[paper_id] = chunks

            except Exception as e:
                log.error(f"Error getting chunks for paper {paper_id}: {e}")
                result[paper_id] = []

        return result

    def find_similar_papers(
        self,
        paper_id: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Find papers similar to the reference paper using semantic similarity.

        Args:
            paper_id: Reference paper ID to compare against.
            max_results: Maximum number of similar papers.

        Returns:
            List of SearchResult excluding the reference paper.
        """
        log.info(f"Finding papers similar to: {paper_id}")

        # Get the reference paper details
        paper = self.get_paper_details(paper_id)
        if not paper:
            log.warning(f"Reference paper not found: {paper_id}")
            return []

        # Use title + abstract as query for similarity
        query_text = f"{paper.title} {paper.abstract}"
        query_embedding = self.embedder.encode(query_text)

        # Search for similar papers (semantic only) using chunk-based search
        raw_results = self.indexer.search(
            query_embedding=query_embedding,
            size=max_results + 5,  # Get extra to account for filtering
            use_semantic=True,
            use_bm25=False
        )

        # Convert to SearchResult objects, excluding the reference paper
        results = []
        for hit in raw_results:
            if hit.get('paper_id') == paper_id:
                continue

            result = SearchResult(
                paper_id=hit.get('paper_id', ''),
                title=hit.get('title', 'Untitled'),
                authors=hit.get('authors', []),
                abstract=hit.get('abstract', ''),
                score=hit.get('_score', 0.0),
                categories=hit.get('categories', []),
                publish_date=hit.get('publish_date'),
                word_count=hit.get('word_count'),
                has_images=hit.get('has_images', False),
                pdf_size=hit.get('pdf_size')
            )
            results.append(result)

            if len(results) >= max_results:
                break

        # Normalize scores and return
        results = self._normalize_scores(results, "semantic")
        return results

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Return aggregate statistics for the chunk-based search index.

        Returns:
            Dictionary with counts and derived metrics.
        """
        base_stats = self.indexer.get_index_stats()

        # Compute paper and chunk specific stats using current doc_type structure
        try:
            # Chunk stats: total chunks and unique paper count derived from chunks
            chunk_stats_body = {
                "query": {"term": {"doc_type": "chunk"}},
                "size": 0,
                "aggs": {
                    "unique_papers": {"cardinality": {"field": "paper_id"}}
                }
            }
            chunk_resp = self.indexer.es.search(index=self.indexer.index_name, body=chunk_stats_body)
            total_chunks = chunk_resp.get("hits", {}).get("total", {}).get("value", 0)
            unique_papers_from_chunks = chunk_resp.get("aggregations", {}).get("unique_papers", {}).get("value", 0)

            # Paper stats: total papers and category distribution should be based on paper docs
            paper_stats_body = {
                "query": {"term": {"doc_type": "paper"}},
                "size": 0,
                "aggs": {
                    "categories": {"terms": {"field": "categories", "size": 100}}
                }
            }
            paper_resp = self.indexer.es.search(index=self.indexer.index_name, body=paper_stats_body)
            total_papers = paper_resp.get("hits", {}).get("total", {}).get("value", 0)
            category_buckets = paper_resp.get("aggregations", {}).get("categories", {}).get("buckets", [])
            category_counts = {bucket["key"]: bucket["doc_count"] for bucket in category_buckets}

            # Prefer explicit total_papers from paper docs; fallback to unique papers from chunks
            if not total_papers and unique_papers_from_chunks:
                total_papers = unique_papers_from_chunks

            result = {
                **base_stats,
                "total_papers": total_papers,
                "total_chunks": total_chunks,
                "category_distribution": category_counts,
            }

            if total_papers > 0:
                result["avg_chunks_per_paper"] = round(total_chunks / total_papers, 2)

            return result

        except Exception as e:
            log.error(f"Error getting additional stats: {e}")
            return base_stats

    def _normalize_scores(self, results: List[SearchResult], search_mode: str) -> List[SearchResult]:
        """
        Normalize raw scores to the [0, 1] interval for UI comparability.

        Args:
            results: Search results with raw engine scores.
            search_mode: Mode used to generate scores.

        Returns:
            Results with score field normalized.
        """
        if not results:
            return results

        # Define expected score ranges for each search mode
        score_ranges = {
            "bm25": {"min": 0.0, "max": 10.0},        # BM25 can vary, use conservative max
            "title_only": {"min": 0.0, "max": 10.0}, # Same as BM25
            "semantic": {"min": 1.0, "max": 7.0},     # cosine similarity + 1.0, with boosts
            "hybrid": {"min": 0.0, "max": 15.0}      # Combined BM25 + semantic with boosts
        }

        # Get score range for current mode
        score_range = score_ranges.get(search_mode, {"min": 0.0, "max": 15.0})
        min_score = score_range["min"]
        max_score = score_range["max"]

        # Find actual min/max in current results for dynamic normalization
        if results:
            actual_scores = [r.score for r in results]
            actual_min = min(actual_scores)
            actual_max = max(actual_scores)

            # Use actual range if it's tighter than expected range
            if actual_max - actual_min > 0:
                # Dynamic normalization based on actual score distribution
                for result in results:
                    normalized_score = (result.score - actual_min) / (actual_max - actual_min)
                    result.score = round(normalized_score, 6)
            else:
                # All scores are the same, set to 1.0
                for result in results:
                    result.score = 1.0

        return results

    def _get_search_fields(self, search_mode: str, include_chunks: bool = False) -> List[str]:
        """
        Resolve search fields and boosts per mode for chunk-based index.

        Args:
            search_mode: One of the supported modes.
            include_chunks: Whether to include chunk_text field.

        Returns:
            List of field expressions with boosts.
        """
        if include_chunks:
            # Full paper search - include chunk_text
            return ["title^5", "abstract^3", "chunk_text^0.5"]
        else:
            # Title/abstract only search
            return ["title^5", "abstract^3"]

    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of Elasticsearch, embedder, and the index.

        Returns:
            Dictionary capturing per-component and overall health.
        """
        health = {
            "status": "healthy",
            "components": {}
        }

        # Check Elasticsearch
        try:
            if self.indexer.es.ping():
                health["components"]["elasticsearch"] = {
                    "status": "healthy",
                    "cluster_health": self.indexer.es.cluster.health()["status"]
                }
            else:
                health["components"]["elasticsearch"] = {"status": "unhealthy"}
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["elasticsearch"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"

        # Check BGE embedder
        try:
            test_embedding = self.embedder.encode("test")
            health["components"]["bge_embedder"] = {
                "status": "healthy",
                "model": self.embedder.model_name,
                "embedding_dim": self.embedder.embedding_dim
            }
        except Exception as e:
            health["components"]["bge_embedder"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"

        # Check index
        try:
            stats = self.indexer.get_index_stats()
            health["components"]["search_index"] = {
                "status": "healthy",
                "document_count": stats.get("document_count", 0),
                "index_size_mb": stats.get("index_size_mb", 0)
            }
        except Exception as e:
            health["components"]["search_index"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"

        return health