import time
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from app.schemas import Paper, SearchRequest, SearchResponse, PaperResult, SearchAnalysis
from app.settings import settings
from app.services.tokenizer import TokenizerService


class ElasticsearchService:
    def __init__(self):
        self.client = AsyncElasticsearch([settings.es_url])
        self.index = settings.es_index
        self.tokenizer = TokenizerService()
    
    async def create_index(self):
        """Create the papers index with proper mappings"""
        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "abstract": {"type": "text", "analyzer": "standard"},
                    "authors": {"type": "text", "analyzer": "standard"},
                    "categories": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "doi": {"type": "keyword"},
                    "url_pdf": {"type": "keyword"},
                    "journal_ref": {"type": "text"},
                    "update_date": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        try:
            await self.client.indices.create(index=self.index, body=mapping)
        except Exception as e:
            if "already_exists" not in str(e).lower():
                raise e
    
    async def index_papers(self, papers: List[Paper]):
        """Bulk index papers"""
        if not papers:
            return
        
        # Prepare bulk operations
        operations = []
        for paper in papers:
            operations.append({"index": {"_index": self.index, "_id": paper.id}})
            operations.append(paper.dict())
        
        # Execute bulk index
        await self.client.bulk(operations=operations)
    
    async def search_papers(self, request: SearchRequest) -> SearchResponse:
        """Search papers with BM25 scoring and filters"""
        start_time = time.time()
        
        # Build query
        query = self._build_search_query(request)
        
        # Build aggregations for analysis
        aggs = {
            "categories": {"terms": {"field": "categories", "size": 20}},
            "years": {"terms": {"field": "year", "size": 20}}
        }
        
        # Calculate offset
        offset = (request.page - 1) * request.page_size
        
        # Execute search
        try:
            response = await self.client.search(
                index=self.index,
                query=query,
                size=request.page_size,
                from_=offset,
                sort=self._build_sort(request),
                aggs=aggs,
                highlight={
                    "fields": {
                        "title": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                        "abstract": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]}
                    }
                }
            )
        except Exception as e:
            # Fallback to basic search if complex query fails
            response = await self.client.search(
                index=self.index,
                query={"match_all": {}},
                size=request.page_size,
                from_=offset
            )
        
        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            
            # Get snippet from highlight or truncated abstract
            snippet = source["abstract"][:200] + "..." if len(source["abstract"]) > 200 else source["abstract"]
            if "highlight" in hit and "abstract" in hit["highlight"]:
                snippet = hit["highlight"]["abstract"][0]
            
            # Calculate relevance score and why_shown
            score = hit["_score"] if hit["_score"] else 0.0
            why_shown = self.tokenizer.get_why_shown(request.q, source)
            
            results.append(PaperResult(
                id=source["id"],
                title=source["title"],
                abstractSnippet=snippet,
                authors=source["authors"],
                categories=source["categories"],
                year=source["year"],
                doi=source.get("doi"),
                urlPdf=source.get("url_pdf"),
                score=score,
                whyShown=why_shown
            ))
        
        # Build analysis
        tokens = self.tokenizer.tokenize(request.q)
        filters_applied = {}
        if request.filters:
            if request.filters.year_from:
                filters_applied["yearFrom"] = request.filters.year_from
            if request.filters.year_to:
                filters_applied["yearTo"] = request.filters.year_to
            if request.filters.categories:
                filters_applied["categories"] = request.filters.categories
            if request.filters.author_query:
                filters_applied["authorQuery"] = request.filters.author_query
        
        analysis = SearchAnalysis(
            query=request.q,
            tokens=tokens,
            sort=request.sort,
            filtersApplied=filters_applied
        )
        
        took_ms = int((time.time() - start_time) * 1000)
        total = response["hits"]["total"]["value"] if isinstance(response["hits"]["total"], dict) else response["hits"]["total"]
        
        return SearchResponse(
            analysis=analysis,
            results=results,
            page=request.page,
            total=total,
            tookMs=took_ms
        )
    
    def _build_search_query(self, request: SearchRequest) -> Dict[str, Any]:
        """Build Elasticsearch query"""
        must_clauses = []
        filter_clauses = []
        
        # Text search with boosting
        if request.q.strip():
            must_clauses.append({
                "multi_match": {
                    "query": request.q,
                    "fields": ["title^2", "abstract", "authors", "categories"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        
        # Filters
        if request.filters:
            if request.filters.year_from or request.filters.year_to:
                year_range = {}
                if request.filters.year_from:
                    year_range["gte"] = request.filters.year_from
                if request.filters.year_to:
                    year_range["lte"] = request.filters.year_to
                filter_clauses.append({"range": {"year": year_range}})
            
            if request.filters.categories:
                filter_clauses.append({"terms": {"categories": request.filters.categories}})
            
            if request.filters.author_query:
                filter_clauses.append({
                    "match": {"authors": request.filters.author_query}
                })
        
        # Build final query
        if not must_clauses:
            must_clauses = [{"match_all": {}}]
        
        query = {"bool": {"must": must_clauses}}
        if filter_clauses:
            query["bool"]["filter"] = filter_clauses
        
        return query
    
    def _build_sort(self, request: SearchRequest) -> List[Dict[str, Any]]:
        """Build sort configuration"""
        if request.sort == "recency":
            return [{"year": {"order": "desc"}}, "_score"]
        else:  # relevance
            return ["_score", {"year": {"order": "desc"}}]
    
    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Paper]:
        """Retrieve papers by IDs"""
        if not paper_ids:
            return []
        
        try:
            response = await self.client.mget(
                index=self.index,
                ids=paper_ids
            )
            
            papers = []
            for doc in response["docs"]:
                if doc["found"]:
                    source = doc["_source"]
                    papers.append(Paper(**source))
            
            return papers
        except Exception:
            return []
    
    async def close(self):
        """Close Elasticsearch connection"""
        await self.client.close()
