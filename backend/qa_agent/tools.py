"""
QA Agent tools for document retrieval and context building.
Following llama_agent design pattern.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re

from backend.services import ElasticsearchSearchService
from backend.config import config
from .config import qa_config

from logs import log

@dataclass
class ContextChunk:
    """Represents a context chunk with metadata"""
    paper_id: str
    paper_title: str
    chunk_index: int
    chunk_text: str
    section_path: Optional[str] = None
    page_number: Optional[int] = None
    image_urls: List[str] = None
    relevance_score: float = 0.0
    chunk_start: int = 0
    chunk_end: int = 0


@dataclass
class ImageContext:
    """Represents an image with context"""
    image_url: str
    paper_id: str
    paper_title: str
    section_path: str
    page_number: int
    caption: Optional[str] = None
    description: Optional[str] = None


class QARetrievalTool:
    """
    Tool for retrieving relevant context chunks for QA.
    
    Supports MinIO bucket structure:
    papers/{paperId}/
      pdf/{paperId}.pdf
      metadata/{paperId}.json
      markdown/index.md
      images/fig_p{page}_{idx}_{sha16}.{ext}
      manifest.json
    """
    
    def __init__(self, es_service: Optional[ElasticsearchSearchService] = None):
        """
        Initialize QA retrieval tool.
        
        Args:
            es_service: Optional ElasticsearchSearchService instance
        """
        if es_service is None:
            self.es_service = ElasticsearchSearchService(
                es_host=qa_config.es_host or config.ES_HOST,
                index_name=config.ES_INDEX_NAME,
                bge_model=config.BGE_MODEL_NAME,
                bge_cache_dir=config.BGE_CACHE_DIR
            )
        else:
            self.es_service = es_service
        # cache embedder reference for query reuse
        self._embedder = self.es_service.embedder

    # ---------------- New helper methods -----------------
    def _build_minio_urls(self, paper_id: str) -> Dict[str, str]:
        base = f"{qa_config.minio_endpoint}/{qa_config.minio_bucket}/papers/{paper_id}"
        primary_md = f"{base}/markdown/{paper_id}.md"
        fallback_md = f"{base}/markdown/index.md"
        return {
            "pdf": f"{base}/pdf/{paper_id}.pdf",
            "metadata": f"{base}/metadata/{paper_id}.json",
            "markdown_primary": primary_md,
            "markdown_fallback": fallback_md,
            # unified key consumed by ContextBuilder (prefer primary)
            "markdown": primary_md
        }

    def _query_chunks_for_paper(self, paper_id: str, question: str, size: int, use_semantic: bool = True) -> List[Dict[str, Any]]:
        """Query ES directly for chunk documents of a paper with optional semantic scoring."""
        # Build base bool filter
        must_filters = [
            {"term": {"paper_id": paper_id}},
            {"term": {"doc_type": "chunk"}}
        ]

        should_clauses: List[Dict[str, Any]] = []
        if question:
            should_clauses.append({
                "match": {
                    "chunk_text": {
                        "query": question,
                        "boost": 1.0
                    }
                }
            })
        script_scores = []
        query_vector = None
        if use_semantic and question:
            query_vector = self._embedder.encode(question)
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            # semantic scoring on chunk_embedding
            script_scores.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.qvec, 'chunk_embedding') + 1.0",
                        "params": {"qvec": query_vector}
                    },
                    "boost": 1.0
                }
            })
        # If we have script scores, we treat them as should clauses too
        should_clauses.extend(script_scores)
        if not should_clauses:
            # fallback to match_all
            should_clauses.append({"match_all": {}})

        body = {
            "query": {
                "bool": {
                    "filter": must_filters,
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            },
            "size": size,
            "_source": {
                "excludes": ["*_embedding"]
            }
        }
        es = self.es_service.indexer.es
        resp = es.search(index=self.es_service.indexer.index_name, body=body)
        hits = resp.get('hits', {}).get('hits', [])
        return [h['_source'] | {'_score': h.get('_score', 0)} for h in hits]

    def _convert_chunk_docs(self, paper_id: str, chunks: List[Dict[str, Any]], paper_title: str, limit: int) -> List[ContextChunk]:
        out: List[ContextChunk] = []
        for d in chunks[:limit]:
            text = d.get('chunk_text')
            if not text:
                continue
            out.append(ContextChunk(
                paper_id=paper_id,
                paper_title=paper_title,
                chunk_index=d.get('chunk_index', 0),
                chunk_text=text,
                relevance_score=d.get('_score', 0.0),
                chunk_start=d.get('chunk_start', 0),
                chunk_end=d.get('chunk_end', 0),
                section_path=None,
                page_number=None,
                image_urls=[]
            ))
        return out

    async def retrieve_single_paper_context(self, paper_id: str, question: str, max_chunks: int = None) -> List[ContextChunk]:
        max_chunks = max_chunks or qa_config.max_context_chunks or 10
        try:
            # fetch paper title from a paper doc (doc_type=paper)
            es = self.es_service.indexer.es
            meta_body = {
                "query": {"bool": {"filter": [{"term": {"paper_id": paper_id}}, {"term": {"doc_type": "paper"}}]}},
                "size": 1,
                "_source": {"excludes": ["*_embedding"]}
            }
            meta_resp = es.search(index=self.es_service.indexer.index_name, body=meta_body)
            paper_title = "Untitled"
            hits_meta = meta_resp.get('hits', {}).get('hits', [])
            if hits_meta:
                paper_title = hits_meta[0]['_source'].get('title', paper_title)

            raw_chunks = self._query_chunks_for_paper(paper_id, question, size=max_chunks * 3, use_semantic=True)
            if not raw_chunks:
                return []
            context = self._convert_chunk_docs(paper_id, raw_chunks, paper_title, max_chunks)
            return context
        except Exception as e:
            log.error(f"retrieve_single_paper_context error: {e}")
            return []

    async def retrieve_multi_paper_context(self, paper_ids: List[str], question: str, max_chunks_per_paper: int = 3) -> List[ContextChunk]:
        results: List[ContextChunk] = []
        for pid in paper_ids: #iteration
            chunks = await self.retrieve_single_paper_context(pid, question, max_chunks=max_chunks_per_paper)
            results.extend(chunks)
        # sort global by score desc then cut overall max
        results.sort(key=lambda c: c.relevance_score, reverse=True)
        max_total = qa_config.max_context_chunks or 10
        return results[:max_total]

    async def retrieve_search_results_context(self, search_query: str, question: str, max_papers: int = 5, max_chunks_per_paper: int = 2) -> List[ContextChunk]:
        # First get top paper docs using search service (paper-level search)
        papers = self.es_service.search(
            query=search_query,
            max_results=max_papers,
            search_mode="hybrid",
            include_chunks=False
        )
        if not papers:
            return []
        paper_ids = [p.paper_id for p in papers]
        return await self.retrieve_multi_paper_context(paper_ids, question, max_chunks_per_paper)

    async def get_paper_minio_urls(self, paper_id: str) -> Dict[str, str]:
        # simplified builder without manifest / MinIOStorage
        return self._build_minio_urls(paper_id)

    async def get_paper_images(self, paper_id: str) -> List[Dict[str, str]]:
        # images not indexed; return empty list for now
        return []


class ImageAnalysisTool:
    """
    Tool for analyzing images from papers using MinIO bucket structure.
    
    Supports structured image filenames: fig_p{page}_{idx}_{sha16}.{ext}
    """
    
    def __init__(self):
        """Initialize image analysis tool"""
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=qa_config.timeout_seconds)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def analyze_image(
        self, 
        image_url: str, 
        paper_title: str, 
        section_path: str = None,
        image_info: Dict[str, Any] = None
    ) -> str:
        """
        Analyze an image and provide a description.
        
        Args:
            image_url: URL of the image to analyze
            paper_title: Title of the paper containing the image
            section_path: Section where the image appears
            image_info: Additional image information from MinIO structure
            
        Returns:
            Description of the image
        """
        try:
            if not qa_config.include_images:
                return "Image analysis disabled"
            
            # Build description with structured information
            description_parts = [f"Image from paper '{paper_title}'"]
            
            if image_info:
                if image_info.get("page"):
                    description_parts.append(f"page {image_info['page']}")
                if image_info.get("filename"):
                    # Extract figure type from filename if possible
                    filename = image_info["filename"]
                    if "fig" in filename.lower():
                        description_parts.append("figure")
                    elif "table" in filename.lower():
                        description_parts.append("table")
                    elif "chart" in filename.lower() or "graph" in filename.lower():
                        description_parts.append("chart/graph")
            
            if section_path:
                description_parts.append(f"in section '{section_path}'")
            
            description = " ".join(description_parts)
            
            # For now, return a structured description
            # In a full implementation, you would use a vision model here
            description += f" (URL: {image_url})"
            
            return description
            
        except Exception as e:
            log.error(f"Error analyzing image {image_url}: {e}")
            return f"Error analyzing image: {str(e)}"
    
    async def analyze_images_in_chunks(
        self, 
        context_chunks: List[ContextChunk]
    ) -> Dict[str, str]:
        """
        Analyze all images mentioned in context chunks.
        
        Args:
            context_chunks: List of context chunks that may contain images
            
        Returns:
            Dictionary mapping image URLs to descriptions
        """
        image_descriptions = {}
        
        if not qa_config.include_images:
            return image_descriptions
        
        async with self:
            for chunk in context_chunks:
                for image_url in chunk.image_urls:
                    if image_url not in image_descriptions:
                        # Try to extract image info from URL
                        image_info = self._extract_image_info_from_url(image_url)
                        
                        description = await self.analyze_image(
                            image_url, chunk.paper_title, chunk.section_path, image_info
                        )
                        image_descriptions[image_url] = description
        
        return image_descriptions
    
    def _extract_image_info_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        Extract image information from MinIO URL.
        
        Args:
            image_url: MinIO image URL
            
        Returns:
            Dictionary with image information
        """
        try:
            # Extract filename from URL
            filename = image_url.split('/')[-1]
            
            # Parse structured filename: fig_p{page}_{idx}_{sha16}.{ext}
            match = re.match(r'fig_p(\d+)_(\d+)_([a-f0-9]{16})\.(\w+)', filename)
            
            if match:
                page_num, idx, sha16, ext = match.groups()
                return {
                    "filename": filename,
                    "page": int(page_num),
                    "index": int(idx),
                    "sha16": sha16,
                    "extension": ext
                }
            
            return {"filename": filename}
            
        except Exception as e:
            log.warning(f"Could not extract image info from URL {image_url}: {e}")
            return {}


class ContextBuilder:
    """
    Tool for building context for LLM prompts.
    
    Supports MinIO bucket structure and structured image information.
    """
    
    def __init__(self):
        """Initialize context builder"""
        pass
    
    def build_single_paper_context(
        self, 
        context_chunks: List[ContextChunk],
        image_descriptions: Dict[str, str] = None,
        minio_urls: Dict[str, str] = None
    ) -> str:
        """
        Build context string for single paper QA.
        
        Args:
            context_chunks: List of context chunks
            image_descriptions: Dictionary of image descriptions
            minio_urls: Dictionary of MinIO URLs for paper components
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        
        # Add paper metadata if available
        if minio_urls:
            paper_info = f"Paper Resources:\n"
            if minio_urls.get("pdf"):
                paper_info += f"- PDF: {minio_urls['pdf']}\n"
            if minio_urls.get("markdown"):
                paper_info += f"- Full Text: {minio_urls['markdown']}\n"
            if minio_urls.get("metadata"):
                paper_info += f"- Metadata: {minio_urls['metadata']}\n"
            context_parts.append(paper_info)
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = f"Chunk {i} (Score: {chunk.relevance_score:.3f}):\n{chunk.chunk_text}"
            
            # Add image information if available
            if chunk.image_urls and image_descriptions:
                for image_url in chunk.image_urls:
                    if image_url in image_descriptions:
                        chunk_text += f"\n\nImage: {image_descriptions[image_url]}"
            
            context_parts.append(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def build_multi_paper_context(
        self, 
        context_chunks: List[ContextChunk],
        image_descriptions: Dict[str, str] = None,
        papers_minio_urls: Dict[str, Dict[str, str]] = None
    ) -> str:
        """
        Build context string for multi-paper QA.
        
        Args:
            context_chunks: List of context chunks from multiple papers
            image_descriptions: Dictionary of image descriptions
            papers_minio_urls: Dictionary mapping paper IDs to their MinIO URLs
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant context found."
        
        # Group chunks by paper
        papers = {}
        for chunk in context_chunks:
            if chunk.paper_id not in papers:
                papers[chunk.paper_id] = {
                    'title': chunk.paper_title,
                    'chunks': []
                }
            papers[chunk.paper_id]['chunks'].append(chunk)
        
        context_parts = []
        
        for paper_id, paper_data in papers.items():
            paper_title = paper_data['title']
            chunks = paper_data['chunks']
            
            paper_context = f"Paper: {paper_title} (ID: {paper_id})\n"
            paper_context += "=" * (len(paper_title) + 20) + "\n"
            
            # Add paper resources if available
            if papers_minio_urls and paper_id in papers_minio_urls:
                minio_urls = papers_minio_urls[paper_id]
                paper_context += "\nPaper Resources:\n"
                if minio_urls.get("pdf"):
                    paper_context += f"- PDF: {minio_urls['pdf']}\n"
                if minio_urls.get("markdown"):
                    paper_context += f"- Full Text: {minio_urls['markdown']}\n"
                if minio_urls.get("metadata"):
                    paper_context += f"- Metadata: {minio_urls['metadata']}\n"
                paper_context += "\n"
            
            for i, chunk in enumerate(chunks, 1):
                chunk_text = f"Chunk {i} (Score: {chunk.relevance_score:.3f}):\n{chunk.chunk_text}"
                
                # Add image information if available
                if chunk.image_urls and image_descriptions:
                    for image_url in chunk.image_urls:
                        if image_url in image_descriptions:
                            chunk_text += f"\n\nImage: {image_descriptions[image_url]}"
                
                paper_context += chunk_text + "\n\n"
            
            context_parts.append(paper_context)
        
        return "\n" + "="*80 + "\n\n".join(context_parts)
    
    def build_search_results_context(
        self, 
        context_chunks: List[ContextChunk],
        image_descriptions: Dict[str, str] = None,
        papers_minio_urls: Dict[str, Dict[str, str]] = None
    ) -> str:
        """
        Build context string for search results QA.
        
        Args:
            context_chunks: List of context chunks from search results
            image_descriptions: Dictionary of image descriptions
            papers_minio_urls: Dictionary mapping paper IDs to their MinIO URLs
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = f"Chunk {i} (Score: {chunk.relevance_score:.3f}):\n"
            chunk_text += f"Paper: {chunk.paper_title} (ID: {chunk.paper_id})\n"
            chunk_text += f"Section: {chunk.section_path}\n"
            chunk_text += f"Content: {chunk.chunk_text}"
            
            # Add paper resources if available
            if papers_minio_urls and chunk.paper_id in papers_minio_urls:
                minio_urls = papers_minio_urls[chunk.paper_id]
                chunk_text += f"\n\nPaper Resources:"
                if minio_urls.get("pdf"):
                    chunk_text += f"\n- PDF: {minio_urls['pdf']}"
                if minio_urls.get("markdown"):
                    chunk_text += f"\n- Full Text: {minio_urls['markdown']}"
                if minio_urls.get("metadata"):
                    chunk_text += f"\n- Metadata: {minio_urls['metadata']}"
            
            # Add image information if available
            if chunk.image_urls and image_descriptions:
                for image_url in chunk.image_urls:
                    if image_url in image_descriptions:
                        chunk_text += f"\n\nImage: {image_descriptions[image_url]}"
            
            context_parts.append(chunk_text)
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_context_for_prompt(
        self, 
        context_chunks: List[ContextChunk],
        question: str,
        is_multi_paper: bool = False,
        image_descriptions: Dict[str, str] = None,
        papers_minio_urls: Dict[str, Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Format context for LLM prompt.
        
        Args:
            context_chunks: List of context chunks
            question: User question
            is_multi_paper: Whether this is multi-paper QA
            image_descriptions: Dictionary of image descriptions
            papers_minio_urls: Dictionary mapping paper IDs to their MinIO URLs
            
        Returns:
            Dictionary with formatted context for prompt
        """
        if is_multi_paper:
            context_text = self.build_multi_paper_context(
                context_chunks, 
                image_descriptions, 
                papers_minio_urls
            )
            num_papers = len(set(chunk.paper_id for chunk in context_chunks))
            # Provide a generic title placeholder to satisfy templates expecting {paper_title}
            # This avoids KeyError when prompts include {paper_title} for multi-paper flows
            paper_title_placeholder = f"{num_papers} papers" if num_papers != 1 else (context_chunks[0].paper_title if context_chunks else "Unknown Paper")
            return {
                "context_chunks": context_text,
                "question": question,
                "num_papers": str(num_papers),
                "paper_title": paper_title_placeholder
            }
        else:
            # For single paper, extract MinIO URLs for the first paper
            single_paper_minio_urls = None
            if context_chunks and papers_minio_urls:
                first_paper_id = context_chunks[0].paper_id
                single_paper_minio_urls = papers_minio_urls.get(first_paper_id)
            
            context_text = self.build_single_paper_context(
                context_chunks, 
                image_descriptions, 
                single_paper_minio_urls
            )
            paper_title = context_chunks[0].paper_title if context_chunks else "Unknown Paper"
            return {
                "context_chunks": context_text,
                "question": question,
                "paper_title": paper_title
            }
