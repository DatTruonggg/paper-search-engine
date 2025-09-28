"""
Elasticsearch-related API routes.

Provides endpoints for index statistics and detailed insights.
"""

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/api/v1/es", tags=["Elasticsearch"])


@router.get("/stats")
async def get_index_stats(request: Request):
    """Return basic Elasticsearch index statistics."""
    search_service = getattr(request.app.state, "search_service", None)
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        return search_service.get_index_stats()
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stats/detailed")
async def get_detailed_stats(request: Request):
    """Return detailed paper and chunk statistics from Elasticsearch.

    Includes totals, averages, and category breakdowns.
    """
    search_service = getattr(request.app.state, "search_service", None)
    if not search_service:
        raise HTTPException(status_code=503, detail="Search service not initialized")

    try:
        basic_stats = search_service.get_index_stats()

        search_body = {
            "size": 10000,
            "_source": [
                "paper_id",
                "title",
                "categories",
                "chunk_count",
                "content_chunks",
            ],
            "query": {"match_all": {}},
        }

        response = search_service.indexer.es.search(
            index=search_service.indexer.index_name,
            body=search_body,
        )

        papers = response["hits"]["hits"]

        total_papers = len(papers)
        total_chunks = 0
        category_paper_count = {}
        category_chunk_count = {}
        paper_details = []

        for hit in papers:
            source = hit["_source"]
            paper_id = source.get("paper_id", "")
            title = source.get("title", "")
            categories = source.get("categories", [])
            chunk_count = source.get("chunk_count", 0)
            content_chunks = source.get("content_chunks", [])

            actual_chunks = len(content_chunks) if content_chunks else chunk_count
            total_chunks += actual_chunks

            paper_details.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "categories": categories,
                    "chunk_count": actual_chunks,
                }
            )

            for category in categories:
                category_paper_count[category] = category_paper_count.get(category, 0) + 1
                category_chunk_count[category] = category_chunk_count.get(category, 0) + actual_chunks

        avg_chunks_per_paper = total_chunks / total_papers if total_papers > 0 else 0

        return {
            "summary": {
                "total_papers": total_papers,
                "total_chunks": total_chunks,
                "avg_chunks_per_paper": round(avg_chunks_per_paper, 2),
                "index_size_mb": basic_stats.get("index_size_mb", 0),
            },
            "by_category": {
                "paper_count": category_paper_count,
                "chunk_count": category_chunk_count,
            },
            "papers": paper_details,
            "elasticsearch_info": {
                "cluster_name": basic_stats.get("cluster_name", "unknown"),
                "document_count": basic_stats.get("document_count", 0),
                "index_size": basic_stats.get("index_size", 0),
            },
        }
    except Exception as exc:  # pragma: no cover - fastapi error surfacing
        raise HTTPException(status_code=500, detail=str(exc))