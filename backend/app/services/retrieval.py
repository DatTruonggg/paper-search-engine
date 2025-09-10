from typing import List
from app.schemas import Paper, SearchRequest
from app.settings import settings


class RetrievalService:
    def __init__(self):
        if settings.data_backend == "es":
            from app.services.elasticsearch_service import ElasticsearchService
            self.search_service = ElasticsearchService()
        else:
            from app.services.postgres_service import PostgresService
            self.search_service = PostgresService()
    
    async def retrieve_papers(self, query: str, top_k: int = 10) -> List[Paper]:
        """Retrieve relevant papers for a query"""
        # Create search request
        search_request = SearchRequest(
            q=query,
            page=1,
            page_size=top_k,
            sort="relevance"
        )
        
        # Execute search
        response = await self.search_service.search_papers(search_request)
        
        # Convert results to Paper objects
        papers = []
        for result in response.results:
            paper = Paper(
                id=result.id,
                title=result.title,
                abstract=result.abstract_snippet,  # Use snippet for chat context
                authors=result.authors,
                categories=result.categories,
                year=result.year,
                doi=result.doi,
                url_pdf=result.url_pdf
            )
            papers.append(paper)
        
        return papers
    
    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Paper]:
        """Get papers by their IDs"""
        return await self.search_service.get_papers_by_ids(paper_ids)
