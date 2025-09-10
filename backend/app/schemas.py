from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str] 
    year: int
    doi: Optional[str] = None
    url_pdf: Optional[str] = None
    journal_ref: Optional[str] = None
    update_date: Optional[str] = None


class SearchFilters(BaseModel):
    year_from: Optional[int] = Field(None, alias="yearFrom")
    year_to: Optional[int] = Field(None, alias="yearTo")
    categories: Optional[List[str]] = None
    author_query: Optional[str] = Field(None, alias="authorQuery")


class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100, alias="pageSize")
    filters: Optional[SearchFilters] = None
    sort: str = Field("relevance", pattern="^(relevance|recency)$")


class SearchAnalysis(BaseModel):
    query: str
    tokens: List[str]
    sort: str
    filters_applied: dict = Field(alias="filtersApplied")


class PaperResult(BaseModel):
    id: str
    title: str
    abstract_snippet: str = Field(alias="abstractSnippet")
    authors: List[str]
    categories: List[str]
    year: int
    doi: Optional[str] = None
    url_pdf: Optional[str] = Field(None, alias="urlPdf")
    score: float
    why_shown: List[str] = Field(alias="whyShown")


class SearchResponse(BaseModel):
    analysis: SearchAnalysis
    results: List[PaperResult]
    page: int
    total: int
    took_ms: int = Field(alias="tookMs")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(10, ge=1, le=50, alias="topK")
    history: Optional[List[ChatMessage]] = None


class Citation(BaseModel):
    id: str
    doi: Optional[str] = None
    where: str = Field(..., description="Location where citation was found")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_papers: List[Paper] = Field(alias="usedPapers")
    summary: Optional[str] = None


class IngestRequest(BaseModel):
    download_pdfs: bool = Field(False, alias="downloadPdfs")
    batch_size: int = Field(1000, ge=1, le=10000, alias="batchSize")


class IngestResponse(BaseModel):
    message: str
    processed: int
    errors: int
    took_ms: int = Field(alias="tookMs")
