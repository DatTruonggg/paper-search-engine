import time
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert

from app.models import Base, PaperModel
from app.schemas import Paper, SearchRequest, SearchResponse, PaperResult, SearchAnalysis
from app.settings import settings
from app.services.tokenizer import TokenizerService


class PostgresService:
    def __init__(self):
        self.engine = create_engine(settings.pg_dsn)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.tokenizer = TokenizerService()
        self._create_tables()
        self._create_search_indexes()
    
    def _create_tables(self):
        """Create database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def _create_search_indexes(self):
        """Create full-text search indexes and triggers"""
        with self.engine.connect() as conn:
            # Create search vector trigger function
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_search_vector() RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := 
                        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
                        setweight(to_tsvector('english', COALESCE(NEW.abstract, '')), 'B') ||
                        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.authors, ' '), '')), 'C') ||
                        setweight(to_tsvector('english', COALESCE(array_to_string(NEW.categories, ' '), '')), 'D');
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # Create trigger
            conn.execute(text("""
                DROP TRIGGER IF EXISTS search_vector_update ON papers;
                CREATE TRIGGER search_vector_update
                    BEFORE INSERT OR UPDATE ON papers
                    FOR EACH ROW EXECUTE FUNCTION update_search_vector();
            """))
            
            # Create GIN index for full-text search
            try:
                conn.execute(text("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_search ON papers USING gin(search_vector);"))
            except Exception:
                # Index might already exist
                pass
            
            # Create other indexes
            try:
                conn.execute(text("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_year ON papers(year);"))
                conn.execute(text("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_categories ON papers USING gin(categories);"))
                conn.execute(text("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_papers_authors ON papers USING gin(authors);"))
            except Exception:
                pass
            
            conn.commit()
    
    async def upsert_papers(self, papers: List[Paper]):
        """Upsert papers using PostgreSQL UPSERT"""
        if not papers:
            return
        
        with self.SessionLocal() as db:
            for paper in papers:
                # Convert to model
                paper_data = paper.dict()
                
                # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
                stmt = insert(PaperModel).values(**paper_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_=paper_data
                )
                
                db.execute(stmt)
            
            db.commit()
    
    async def search_papers(self, request: SearchRequest) -> SearchResponse:
        """Search papers using PostgreSQL full-text search"""
        start_time = time.time()
        
        with self.SessionLocal() as db:
            # Build base query
            query = db.query(PaperModel)
            
            # Text search using tsvector
            if request.q.strip():
                search_query = self._build_search_query(request.q)
                query = query.filter(
                    PaperModel.search_vector.match(search_query)
                )
            
            # Apply filters
            if request.filters:
                if request.filters.year_from:
                    query = query.filter(PaperModel.year >= request.filters.year_from)
                if request.filters.year_to:
                    query = query.filter(PaperModel.year <= request.filters.year_to)
                if request.filters.categories:
                    query = query.filter(PaperModel.categories.overlap(request.filters.categories))
                if request.filters.author_query:
                    # Search in authors array
                    author_conditions = []
                    for author_term in request.filters.author_query.split():
                        author_conditions.append(
                            PaperModel.authors.any(text(f"'{author_term}' ILIKE ANY(authors)"))
                        )
                    query = query.filter(and_(*author_conditions))
            
            # Get total count
            total = query.count()
            
            # Apply sorting
            if request.sort == "recency":
                query = query.order_by(PaperModel.year.desc(), PaperModel.id)
            else:  # relevance
                if request.q.strip():
                    # Order by search rank then year
                    search_query = self._build_search_query(request.q)
                    query = query.order_by(
                        text(f"ts_rank(search_vector, to_tsquery('english', '{search_query}')) DESC"),
                        PaperModel.year.desc()
                    )
                else:
                    query = query.order_by(PaperModel.year.desc())
            
            # Apply pagination
            offset = (request.page - 1) * request.page_size
            papers = query.offset(offset).limit(request.page_size).all()
            
            # Convert to results
            results = []
            for paper in papers:
                # Calculate relevance score
                score = 1.0
                if request.q.strip():
                    # Simple scoring based on year and text match
                    year_score = 0.15 * self._sigmoid_year_score(paper.year)
                    text_score = 0.85  # Assume good match since it was returned
                    score = text_score + year_score
                
                # Get snippet
                snippet = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
                
                # Get why_shown
                why_shown = self.tokenizer.get_why_shown(request.q, paper.to_dict())
                
                results.append(PaperResult(
                    id=paper.id,
                    title=paper.title,
                    abstractSnippet=snippet,
                    authors=paper.authors,
                    categories=paper.categories,
                    year=paper.year,
                    doi=paper.doi,
                    urlPdf=paper.url_pdf,
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
            
            return SearchResponse(
                analysis=analysis,
                results=results,
                page=request.page,
                total=total,
                tookMs=took_ms
            )
    
    def _build_search_query(self, query_text: str) -> str:
        """Build PostgreSQL full-text search query"""
        # Clean and prepare query for tsquery
        terms = query_text.split()
        # Join with & for AND search
        return " & ".join(f"'{term}'" for term in terms if term.strip())
    
    def _sigmoid_year_score(self, year: int) -> float:
        """Calculate sigmoid score for year (newer papers score higher)"""
        import math
        # Sigmoid function centered around 2018
        x = year - 2018
        return 1 / (1 + math.exp(-x))
    
    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Paper]:
        """Retrieve papers by IDs"""
        if not paper_ids:
            return []
        
        with self.SessionLocal() as db:
            papers = db.query(PaperModel).filter(PaperModel.id.in_(paper_ids)).all()
            return [Paper(**paper.to_dict()) for paper in papers]
