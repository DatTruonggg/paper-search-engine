"""
Semantic Scholar API compatible endpoints.

Provides exact compatibility with Semantic Scholar's API specification based on your ES structure.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from backend.api.main import search_service
from backend.llama_agent.config import llama_config
from backend.llama_agent.query_analyzer import QueryAnalyzer
from logs import log
from asta.api.scholarqa.rag.retrieval import PaperFinder
from asta.api.scholarqa.rag.retriever_base import FullTextRetriever, AbstractRetriever
from asta.api.scholarqa.scholar_qa import ScholarQA
from pydantic import BaseModel
from asta.api.scholarqa.llms.constants import GEMINI_25_PRO, GPT_5_CHAT

router = APIRouter(prefix="/graph/v1", tags=["Semantic Scholar API"])

# Initialize analyzer
analyzer = QueryAnalyzer()


# Semantic Scholar API Response Models (exact match)
class Author(BaseModel):
    authorId: Optional[str] = None
    name: str


class Paper(BaseModel):
    paperId: str
    externalIds: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    publicationVenue: Optional[Dict[str, Any]] = None
    year: Optional[int] = None
    referenceCount: Optional[int] = None
    citationCount: Optional[int] = None
    influentialCitationCount: Optional[int] = None
    isOpenAccess: Optional[bool] = None
    openAccessPdf: Optional[Dict[str, str]] = None
    fieldsOfStudy: Optional[List[str]] = None
    s2FieldsOfStudy: Optional[List[Dict[str, Any]]] = None
    publicationTypes: Optional[List[str]] = None
    publicationDate: Optional[str] = None
    journal: Optional[Dict[str, Any]] = None
    authors: Optional[List[Author]] = None


class SnippetAnnotations(BaseModel):
    refMentions: Optional[List[Dict]] = None
    sentences: Optional[List[Dict]] = None


class Snippet(BaseModel):
    text: str
    snippetKind: Optional[str] = None
    snippetOffset: Optional[int] = None
    section: Optional[str] = None
    annotations: Optional[SnippetAnnotations] = None


class SnippetMatch(BaseModel):
    paperId: str
    snippet: Snippet


class PaperRelevanceSearchResponse(BaseModel):
    total: int
    offset: int
    next: Optional[int] = None
    data: List[Dict[str, Any]]


class SnippetSearchResponse(BaseModel):
    data: List[Dict[str, Any]]
    retrievalVersion: Optional[str] = "v1.0"


def format_paper_for_s2(paper_data: Any, fields: str = None) -> Dict[str, Any]:
    """Format paper data to match Semantic Scholar API response exactly."""
    if not fields:
        fields = "paperId,title"

    field_list = [f.strip().lower() for f in fields.split(",")]
    result = {}

    # Always include paperId and corpusId - use actual arXiv ID
    result["paperId"] = paper_data.paper_id  # This should be like "2509.01324"
    result["corpusId"] = paper_data.paper_id  # Same as paperId for arXiv papers

    log.debug(f"Formatting paper {paper_data.paper_id} with fields: {field_list}")

    for field in field_list:
        if field in ["paperid", "corpusid"]:
            continue  # Already included

        elif field == "externalids":
            # For arXiv papers, include ArXiv ID
            external_ids = {"ArXiv": paper_data.paper_id}
            result["externalIds"] = external_ids

        elif field == "url":
            # Always point to arXiv for your papers
            result["url"] = f"https://arxiv.org/abs/{paper_data.paper_id}"

        elif field == "title":
            result["title"] = getattr(paper_data, 'title', None)

        elif field == "abstract":
            result["abstract"] = getattr(paper_data, 'abstract', None)

        elif field == "venue":
            categories = getattr(paper_data, 'categories', None)
            if categories and len(categories) > 0:
                result["venue"] = categories[0].replace("cs.", "").replace("_", " ").title()

        elif field == "publicationvenue":
            categories = getattr(paper_data, 'categories', None)
            if categories and len(categories) > 0:
                venue_name = categories[0].replace("cs.", "").replace("_", " ").title()
                result["publicationVenue"] = {
                    "id": str(hash(venue_name) % 1000000),
                    "name": venue_name,
                    "type": "conference" if "cs." in categories[0] else "journal",
                    "alternate_names": [],
                    "url": None
                }

        elif field == "year":
            publish_date = getattr(paper_data, 'publish_date', None)
            if publish_date:
                try:
                    result["year"] = int(publish_date[:4])
                except:
                    result["year"] = None

        elif field == "referencecount":
            result["referenceCount"] = 0  # Not available

        elif field == "citationcount":
            result["citationCount"] = 0  # Not available

        elif field == "influentialcitationcount":
            result["influentialCitationCount"] = 0  # Not available

        elif field == "isopenaccess":
            result["isOpenAccess"] = True  # Assume arXiv papers are open access

        elif field == "openaccesspdf":
            minio_url = getattr(paper_data, 'minio_pdf_url', None)
            if minio_url or "." in paper_data.paper_id:
                pdf_url = minio_url if minio_url else f"https://arxiv.org/pdf/{paper_data.paper_id}.pdf"
                result["openAccessPdf"] = {
                    "url": pdf_url,
                    "status": "GREEN",
                    "license": "CC0",
                    "disclaimer": f"Open access paper available at {pdf_url}"
                }

        elif field == "fieldsofstudy":
            categories = getattr(paper_data, 'categories', None)
            if categories:
                # Convert cs.XX to readable format
                readable_fields = []
                for cat in categories:
                    if cat.startswith("cs."):
                        readable_fields.append("Computer Science")
                    elif cat.startswith("math."):
                        readable_fields.append("Mathematics")
                    elif cat.startswith("physics."):
                        readable_fields.append("Physics")
                    else:
                        readable_fields.append(cat.replace("_", " ").title())
                result["fieldsOfStudy"] = list(set(readable_fields))  # Remove duplicates
            else:
                result["fieldsOfStudy"] = []

        elif field == "s2fieldsofstudy":
            categories = getattr(paper_data, 'categories', None)
            if categories:
                s2_fields = []
                for cat in categories:
                    if cat.startswith("cs."):
                        s2_fields.append({"category": "Computer Science", "source": "external"})
                    elif cat.startswith("math."):
                        s2_fields.append({"category": "Mathematics", "source": "external"})
                    elif cat.startswith("physics."):
                        s2_fields.append({"category": "Physics", "source": "external"})
                    else:
                        s2_fields.append({"category": cat.replace("_", " ").title(), "source": "external"})

                # Remove duplicates
                seen = set()
                unique_fields = []
                for field_obj in s2_fields:
                    key = field_obj["category"]
                    if key not in seen:
                        seen.add(key)
                        unique_fields.append(field_obj)
                result["s2FieldsOfStudy"] = unique_fields

        elif field == "publicationtypes":
            result["publicationTypes"] = ["JournalArticle"]

        elif field == "publicationdate":
            result["publicationDate"] = getattr(paper_data, 'publish_date', None)

        elif field == "journal":
            categories = getattr(paper_data, 'categories', None)
            if categories and len(categories) > 0:
                result["journal"] = {
                    "name": categories[0].replace("cs.", "").replace("_", " ").title(),
                    "pages": None,
                    "volume": None
                }

        elif field == "citationstyles":
            # Generate basic bibtex
            title = getattr(paper_data, 'title', '')
            authors = getattr(paper_data, 'authors', [])
            year = None
            publish_date = getattr(paper_data, 'publish_date', None)
            if publish_date:
                try:
                    year = int(publish_date[:4])
                except:
                    pass

            author_str = " and ".join(authors) if authors else "Unknown"
            bibtex = f"@article{{{paper_data.paper_id},\n  title={{{title}}},\n  author={{{author_str}}},\n  year={{{year or 'Unknown'}}}\n}}"

            result["citationStyles"] = {
                "bibtex": bibtex
            }

        elif field == "authors":
            authors = getattr(paper_data, 'authors', None)
            if authors:
                result["authors"] = [
                    {
                        "authorId": None,  # No author IDs for arXiv papers in your data
                        "name": name
                    }
                    for name in authors
                ]

        elif field == "citations":
            result["citations"] = []  # Not available

        elif field == "references":
            result["references"] = []  # Not available

        elif field == "embedding":
            result["embedding"] = {
                "model": "specter@v0.1.1",
                "vector": [0.0, 0.0]  # Placeholder
            }

        elif field == "tldr":
            abstract = getattr(paper_data, 'abstract', None)
            if abstract:
                tldr_text = abstract[:150].strip()
                if len(abstract) > 150:
                    tldr_text += "..."
                result["tldr"] = {
                    "model": "tldr@v2.0.0",
                    "text": tldr_text
                }

    return result


def format_snippet_for_s2(paper_data: Any, chunk_text: str, chunk_start: int, score: float, fields: str = None) -> Dict[str, Any]:
    """Format snippet data to match Semantic Scholar API response."""

    # Create snippet object with proper S2 format
    snippet_data = {
        "text": chunk_text,
        "snippetKind": "paragraph",
        "section": "body",
        "snippetOffset": {
            "start": chunk_start,
            "end": chunk_start + len(chunk_text)
        }
    }

    # Add annotations if requested
    if fields and ("annotations" in fields.lower() or "sentences" in fields.lower() or "refmentions" in fields.lower()):
        # Basic sentence segmentation
        sentences = []
        current_pos = 0
        for sentence in chunk_text.split('. '):
            if sentence.strip():
                sentence_end = current_pos + len(sentence) + 1
                sentences.append({
                    "start": current_pos,
                    "end": sentence_end
                })
                current_pos = sentence_end + 1

        snippet_data["annotations"] = {
            "sentences": sentences,
            "refMentions": []  # Empty for now
        }

    # Create paper object
    paper_obj = {
        "corpusId": paper_data.paper_id,  # Use actual arXiv ID, not hash
        "title": getattr(paper_data, 'title', ''),
        "authors": []
    }

    # Format authors properly
    authors = getattr(paper_data, 'authors', [])
    if authors:
        paper_obj["authors"] = [{"authorId": None, "name": name} for name in authors]

    # Add openAccessInfo if available
    if hasattr(paper_data, 'minio_pdf_url') and paper_data.minio_pdf_url:
        paper_obj["openAccessInfo"] = {
            "license": "UNKNOWN",
            "status": "GREEN",
            "disclaimer": f"Open access paper available at {paper_data.minio_pdf_url}"
        }

    return {
        "snippet": snippet_data,
        "score": score,
        "paper": paper_obj
    }


@router.get("/paper/search", response_model=PaperRelevanceSearchResponse)
async def paper_search(
    query: str = Query(..., description="Plain-text search string"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to return"),
    publicationTypes: Optional[str] = Query(None, description="Publication types filter"),
    openAccessPdf: Optional[bool] = Query(None, description="Filter by open access PDF availability"),
    minCitationCount: Optional[int] = Query(None, description="Minimum citation count"),
    publicationDateOrYear: Optional[str] = Query(None, description="Publication date or year filter"),
    year: Optional[str] = Query(None, description="Publication year filter"),
    venue: Optional[str] = Query(None, description="Venue filter"),
    fieldsOfStudy: Optional[str] = Query(None, description="Fields of study filter"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: Optional[int] = Query(0, ge=0, description="Result offset for pagination")
):
    """
    Search for papers by relevance (up to 1,000 results).
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Parse date filters
        date_from = None
        date_to = None
        if year:
            date_from = f"{year}-01-01"
            date_to = f"{year}-12-31"
        elif publicationDateOrYear:
            if "-" in publicationDateOrYear:
                parts = publicationDateOrYear.split("-")
                if len(parts) >= 3:  # Full date
                    date_from = publicationDateOrYear
                    date_to = publicationDateOrYear
                else:  # Year range
                    date_from = f"{parts[0]}-01-01"
                    date_to = f"{parts[1]}-12-31" if len(parts) > 1 else f"{parts[0]}-12-31"
            else:  # Single year
                date_from = f"{publicationDateOrYear}-01-01"
                date_to = f"{publicationDateOrYear}-12-31"

        # Parse categories
        categories = None
        if fieldsOfStudy:
            categories = [f.strip() for f in fieldsOfStudy.split(",")]
        elif venue:
            categories = [venue]

        # Enhance query
        enhanced_query = query
        if llama_config.enable_query_analysis:
            try:
                enhanced_query = await analyzer.enhance_query(query)
            except Exception as e:
                log.warning(f"Query enhancement failed: {e}")

        # Search
        log.info(f"S2 Paper search: query='{enhanced_query}', limit={limit}, offset={offset}")
        results = search_service.search(
            query=enhanced_query,
            max_results=min(limit + offset, 1000),  # S2 API limit
            search_mode="hybrid",
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            include_chunks=False
        )

        log.info(f"S2 Paper search returned {len(results)} results")

        # Apply offset
        paginated_results = results[offset:offset + limit]

        # Format results
        papers = []
        log.info(f"Formatting {len(paginated_results)} paginated results from {len(results)} total results")
        for i, result in enumerate(paginated_results):
            log.info(f"Formatting result {i}: paper_id={result.paper_id}, title={result.title[:50] if result.title else 'NO_TITLE'}...")
            try:
                paper_dict = format_paper_for_s2(result, fields or "paperId,title")
                papers.append(paper_dict)
                log.info(f"Successfully formatted paper {i}: {list(paper_dict.keys())}")
            except Exception as e:
                log.error(f"Error formatting paper {i} ({result.paper_id}): {e}")
                # Add a minimal paper dict even if formatting fails
                papers.append({
                    "paperId": result.paper_id,
                    "title": result.title
                })
                log.info(f"Added minimal paper {i} as fallback")

        # Calculate next offset
        next_offset = None
        if offset + limit < len(results) and offset + limit < 1000:
            next_offset = offset + limit

        final_response = PaperRelevanceSearchResponse(
            total=min(len(results), 1000),
            offset=offset,
            next=next_offset,
            data=papers
        )

        log.info(f"S2 Paper search final response: total={final_response.total}, data_count={len(papers)}, offset={offset}")
        return final_response

    except Exception as e:
        log.error(f"Paper search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/snippet/search", response_model=SnippetSearchResponse)
async def snippet_search(
    query: str = Query(..., description="Plain-text search string"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to return"),
    paperIds: Optional[str] = Query(None, description="Comma-separated list of paper IDs"),
    minCitationCount: Optional[int] = Query(None, description="Minimum citation count"),
    insertedBefore: Optional[str] = Query(None, description="Filter by insertion date"),
    publicationDateOrYear: Optional[str] = Query(None, description="Publication date or year filter"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: Optional[int] = Query(0, ge=0, description="Result offset for pagination")
):
    """
    Search for text snippets within papers (up to 1,000 results).
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Parse paper IDs filter
        paper_id_list = None
        if paperIds:
            paper_id_list = [pid.strip() for pid in paperIds.split(",")]

        # Parse date filter
        date_from = None
        date_to = None
        if publicationDateOrYear:
            if "-" in publicationDateOrYear:
                parts = publicationDateOrYear.split("-")
                if len(parts) >= 3:  # Full date
                    date_from = publicationDateOrYear
                    date_to = publicationDateOrYear
                else:  # Year range
                    date_from = f"{parts[0]}-01-01"
                    date_to = f"{parts[1]}-12-31" if len(parts) > 1 else f"{parts[0]}-12-31"
            else:  # Single year
                date_from = f"{publicationDateOrYear}-01-01"
                date_to = f"{publicationDateOrYear}-12-31"

        # Enhance query
        enhanced_query = query
        if llama_config.enable_query_analysis:
            try:
                enhanced_query = await analyzer.enhance_query(query)
            except Exception as e:
                log.warning(f"Query enhancement failed: {e}")

        # Search with chunks enabled for snippet extraction
        log.info(f"S2 Snippet search: query='{enhanced_query}', limit={limit}, offset={offset}")
        results = search_service.search(
            query=enhanced_query,
            max_results=min(limit * 3, 1000),  # Get more results to extract snippets
            search_mode="hybrid",
            date_from=date_from,
            date_to=date_to,
            include_chunks=True  # Enable chunk search for snippets
        )

        log.info(f"S2 Snippet search returned {len(results)} results")

        # Extract snippets from results
        # When include_chunks=True, results are individual chunk documents
        snippets = []

        log.info(f"Processing {len(results)} chunk results for snippet extraction")
        for result_idx, result in enumerate(results):
            # Filter by paper IDs if specified
            if paper_id_list and result.paper_id not in paper_id_list:
                continue

            if len(snippets) >= limit + offset:
                break

            log.debug(f"Processing chunk result {result_idx}: {result.paper_id}")

            # Since include_chunks=True, each result is a chunk document
            # The search service should have populated chunk-specific fields

            # Get chunk text and position from the result
            chunk_text = result.chunk_text
            chunk_start = result.chunk_start or 0

            if not chunk_text:
                # Fallback to abstract if no chunk text
                chunk_text = result.abstract
                chunk_start = 0
                log.debug(f"No chunk_text, using abstract: {len(chunk_text) if chunk_text else 0} chars")
            else:
                log.debug(f"Found chunk_text: {len(chunk_text)} chars, start: {chunk_start}")

            if chunk_text and len(chunk_text.strip()) > 20:
                try:
                    snippet_dict = format_snippet_for_s2(result, chunk_text[:1000], chunk_start, result.score, fields)
                    snippets.append(snippet_dict)
                    log.debug(f"Added snippet {len(snippets)} for paper {result.paper_id}")
                except Exception as e:
                    log.error(f"Error formatting snippet for {result.paper_id}: {e}")
            else:
                log.debug(f"No valid chunk text found for {result.paper_id}")

            if len(snippets) >= limit + offset:
                break

        # Apply offset and limit
        paginated_snippets = snippets[offset:offset + limit]

        final_response = SnippetSearchResponse(
            data=paginated_snippets,
            retrievalVersion="v1.0"
        )

        log.info(f"S2 Snippet search final response: data_count={len(paginated_snippets)}, offset={offset}")
        return final_response

    except Exception as e:
        log.error(f"Snippet search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paper/batch")
async def paper_batch(
    request: Request,
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to return")
):
    """
    Retrieve batches of papers by ID (up to 500 IDs, 10MB response limit).
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        body = await request.json()
        paper_ids = body.get("ids", [])

        if not paper_ids:
            raise HTTPException(status_code=400, detail="ids field required")

        if len(paper_ids) > 500:
            raise HTTPException(status_code=400, detail="Cannot process more than 500 paper IDs")

        # Retrieve papers
        papers = []
        for paper_id in paper_ids:
            try:
                # Clean up paper ID - remove CorpusId: prefix if present
                clean_paper_id = paper_id
                if paper_id.startswith("CorpusId:"):
                    clean_paper_id = paper_id.replace("CorpusId:", "")
                elif paper_id.startswith("ArXiv:"):
                    clean_paper_id = paper_id.replace("ArXiv:", "")

                log.debug(f"Batch lookup: {paper_id} -> {clean_paper_id}")

                paper = search_service.get_paper_details(clean_paper_id)
                if paper:
                    paper_dict = format_paper_for_s2(paper, fields)
                    papers.append(paper_dict)
                else:
                    papers.append(None)  # Maintain order, include null for missing papers
            except Exception as e:
                log.error(f"Error retrieving paper {paper_id}: {e}")
                papers.append(None)

        log.info(f"Batch request: {len(paper_ids)} requested, {len([p for p in papers if p is not None])} found")
        return papers  # S2 API returns raw array, not wrapped in object

    except Exception as e:
        log.error(f"Paper batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query", response_model=PaperRelevanceSearchResponse)
async def query_demo(
    query: str = Query(..., description="User query to retrieve top relevant papers"),
    limit: int = Query(10, ge=1, le=100, description="Number of papers to return"),
    fields: Optional[str] = Query("paperId,title,abstract,year,venue,authors", description="Fields to include in response")
):
    """
    Simple demo endpoint: from a user query return top relevant papers.

    This is a convenience wrapper around the paper search flow with sane defaults,
    suitable for quick demos.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        # Direct search for papers (title/abstract only)
        results = search_service.search(
            query=query,
            max_results=min(limit, 100),
            search_mode="hybrid",
            include_chunks=False
        )

        # Format results
        papers = []
        for result in results[:limit]:
            try:
                papers.append(format_paper_for_s2(result, fields))
            except Exception as e:
                log.error(f"Error formatting paper {result.paper_id}: {e}")
                papers.append({
                    "paperId": result.paper_id,
                    "title": getattr(result, 'title', None)
                })

        return PaperRelevanceSearchResponse(
            total=len(papers),
            offset=0,
            next=None,
            data=papers
        )
    except Exception as e:
        log.error(f"Query demo error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QABody(BaseModel):
    query: str
    inline_tags: bool = True
    n_retrieval: int | None = None
    n_keyword_srch: int | None = None


@router.post("/qa")
async def graph_qa(body: QABody):
    """
    Run full ScholarQA pipeline using the S2-like graph endpoints under the hood.
    This executes: preprocess → retrieval → rerank+aggregate → quotes → clustering → summary.
    """
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        log.info(f"[GRAPH][QA] start | query='{body.query}'")
        # Use in-process ES retriever to avoid HTTP self-calls
        class InProcessESRetriever(AbstractRetriever):
            def __init__(self, svc, n_retrieval: int = 128, n_keyword_srch: int = 10, search_mode: str = "hybrid"):
                self.svc = svc
                self.n_retrieval = n_retrieval
                self.n_keyword_srch = n_keyword_srch
                self.search_mode = search_mode

            def _parse_year(self, year_range: str | None) -> tuple[Optional[str], Optional[str]]:
                if not year_range:
                    return None, None
                try:
                    parts = str(year_range).split("-")
                    start = parts[0].strip() if parts and parts[0].strip() else None
                    end = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
                    return (f"{start}-01-01" if start else None, f"{end}-12-31" if end else None)
                except Exception:
                    return None, None

            def retrieve_passages(self, query: str, **filter_kwargs) -> list[dict[str, Any]]:
                date_from, date_to = self._parse_year(filter_kwargs.get("year"))
                results = self.svc.search(
                    query=query,
                    max_results=self.n_retrieval,
                    search_mode=self.search_mode,
                    date_from=date_from,
                    date_to=date_to,
                    include_chunks=True,
                )
                snippets: list[dict[str, Any]] = []
                for hit in results:
                    chunk_text = getattr(hit, "chunk_text", None)
                    if not chunk_text:
                        continue
                    snippets.append({
                        "corpus_id": str(hit.paper_id) if hit.paper_id is not None else "",
                        "title": hit.title or "",
                        "text": chunk_text or "",
                        "score": float(getattr(hit, "score", 0.0) or 0.0),
                        "section_title": "chunk",
                        "char_start_offset": int(getattr(hit, "chunk_start", 0) or 0),
                        "sentence_offsets": [],
                        "ref_mentions": [],
                        "pdf_hash": "",
                        "stype": "es",
                    })
                return snippets

            def retrieve_additional_papers(self, query: str, **filter_kwargs) -> list[dict[str, Any]]:
                if not self.n_keyword_srch:
                    return []
                date_from, date_to = self._parse_year(filter_kwargs.get("year"))
                results = self.svc.search(
                    query=query,
                    max_results=self.n_keyword_srch,
                    search_mode=self.search_mode,
                    date_from=date_from,
                    date_to=date_to,
                    include_chunks=False,
                )
                papers: list[dict[str, Any]] = []
                for hit in results:
                    papers.append({
                        "corpus_id": str(hit.paper_id) if hit.paper_id is not None else "",
                        "title": hit.title or "",
                        "abstract": hit.abstract or "",
                        "text": hit.abstract or "",
                        "section_title": "abstract",
                        "char_start_offset": 0,
                        "sentence_offsets": [],
                        "ref_mentions": [],
                        "score": 0.0,
                        "stype": "es_api",
                        "pdf_hash": "",
                        "authors": [{"name": a} for a in (hit.authors or [])],
                        "year": int(hit.publish_date[:4]) if getattr(hit, "publish_date", None) else 0,
                        "venue": (hit.categories[0] if getattr(hit, "categories", None) else ""),
                        "citationCount": 0,
                        "referenceCount": 0,
                        "influentialCitationCount": 0,
                    })
                return papers

        retriever = InProcessESRetriever(
            search_service,
            n_retrieval=body.n_retrieval or 128,
            n_keyword_srch=body.n_keyword_srch or 10,
        )
        paper_finder = PaperFinder(retriever)
        # Prefer Gemini by default unless overridden
        llm_model = GEMINI_25_PRO
        decomposer_llm = llm_model
        sqa = ScholarQA(
            paper_finder=paper_finder,
            llm_model=llm_model,
            decomposer_llm=decomposer_llm,
            fallback_llm=GPT_5_CHAT
        )
        result = sqa.answer_query(body.query, inline_tags=body.inline_tags)
        log.info(f"[GRAPH][QA] done | sections={len(result.get('sections', [])) if isinstance(result, dict) else 'N/A'}")
        return result
    except Exception as e:
        log.exception("[GRAPH][QA] failed")
        raise HTTPException(status_code=500, detail=str(e))