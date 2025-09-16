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
                es_host=qa_config.es_host,
                index_name=config.ES_INDEX_NAME,
                bge_model=config.BGE_MODEL_NAME,
                bge_cache_dir=config.BGE_CACHE_DIR
            )
        else:
            self.es_service = es_service
    
    async def retrieve_single_paper_context(
        self, 
        paper_id: str, 
        question: str, 
        max_chunks: int = None
    ) -> List[ContextChunk]:
        """
        Retrieve relevant context chunks from a single paper.
        
        Args:
            paper_id: Paper ID to search within
            question: Question to find relevant context for
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of relevant context chunks
        """
        max_chunks = max_chunks or qa_config.max_context_chunks
        
        try:
            log.info(f"Retrieving context for paper {paper_id} with question: {question}")
            
            # Search within the specific paper using hybrid search
            results = self.es_service.search(
                query=question,
                max_results=max_chunks * 2,  # Get more for filtering
                search_mode="hybrid",
                include_chunks=True
            )
            
            # Filter results to only include the specified paper
            paper_results = [r for r in results if r.paper_id == paper_id]
            
            if not paper_results:
                log.warning(f"No results found for paper {paper_id}")
                return []
            
            # Convert to ContextChunk objects
            context_chunks = []
            for result in paper_results[:max_chunks]:
                # Get detailed paper information
                paper_details = self.es_service.get_paper_details(paper_id)
                if not paper_details:
                    continue
                
                # Extract chunk information
                chunk = ContextChunk(
                    paper_id=paper_id,
                    paper_title=paper_details.title,
                    chunk_index=0,  # Will be updated if we have chunk info
                    chunk_text=result.abstract or "",  # Use abstract as context
                    relevance_score=result.score,
                    section_path=None,
                    page_number=None,
                    image_urls=[]
                )
                
                # If we have chunk matches, use the best one
                if hasattr(result, 'chunk_matches') and result.chunk_matches:
                    best_chunk = result.chunk_matches[0]
                    chunk.chunk_text = best_chunk.get('chunk_text', chunk.chunk_text)
                    chunk.chunk_index = best_chunk.get('chunk_index', 0)
                    chunk.chunk_start = best_chunk.get('chunk_start', 0)
                    chunk.chunk_end = best_chunk.get('chunk_end', 0)
                    chunk.section_path = best_chunk.get('section_path')
                    chunk.page_number = best_chunk.get('page_number')
                    chunk.image_urls = best_chunk.get('image_urls', [])
                
                context_chunks.append(chunk)
            
            log.info(f"Retrieved {len(context_chunks)} context chunks for paper {paper_id}")
            return context_chunks
            
        except Exception as e:
            log.error(f"Error retrieving single paper context: {e}")
            return []
    
    async def retrieve_multi_paper_context(
        self, 
        paper_ids: List[str], 
        question: str, 
        max_chunks_per_paper: int = 3
    ) -> List[ContextChunk]:
        """
        Retrieve relevant context chunks from multiple papers.
        
        Args:
            paper_ids: List of paper IDs to search within
            question: Question to find relevant context for
            max_chunks_per_paper: Maximum chunks per paper
            
        Returns:
            List of relevant context chunks from all papers
        """
        all_chunks = []
        
        for paper_id in paper_ids:
            chunks = await self.retrieve_single_paper_context(
                paper_id, question, max_chunks_per_paper
            )
            all_chunks.extend(chunks)
        
        # Sort by relevance score
        all_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit total chunks
        max_total = qa_config.max_context_chunks
        return all_chunks[:max_total]
    
    async def retrieve_search_results_context(
        self, 
        search_query: str, 
        question: str, 
        max_papers: int = 5,
        max_chunks_per_paper: int = 2
    ) -> List[ContextChunk]:
        """
        Retrieve context from papers returned by a search query.
        
        Args:
            search_query: Original search query that returned papers
            question: Question to find relevant context for
            max_papers: Maximum number of papers to include
            max_chunks_per_paper: Maximum chunks per paper
            
        Returns:
            List of relevant context chunks
        """
        try:
            log.info(f"Retrieving context from search results for query: {search_query}")
            
            # First, get papers from the search query
            search_results = self.es_service.search(
                query=search_query,
                max_results=max_papers,
                search_mode="hybrid",
                include_chunks=False
            )
            
            if not search_results:
                log.warning(f"No search results found for query: {search_query}")
                return []
            
            # Extract paper IDs
            paper_ids = [result.paper_id for result in search_results]
            
            # Get context chunks from these papers
            context_chunks = await self.retrieve_multi_paper_context(
                paper_ids, question, max_chunks_per_paper
            )
            
            log.info(f"Retrieved {len(context_chunks)} context chunks from {len(paper_ids)} papers")
            return context_chunks
            
        except Exception as e:
            log.error(f"Error retrieving search results context: {e}")
            return []
    
    async def get_paper_minio_urls(self, paper_id: str) -> Dict[str, str]:
        """
        Get MinIO URLs for all components of a paper.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Dictionary with MinIO URLs for paper components
        """
        try:
            # Try to get manifest from MinIO
            from data_pipeline.minio_storage import MinIOStorage
            
            minio_storage = MinIOStorage(endpoint=qa_config.minio_endpoint)
            manifest_path = f"{paper_id}/manifest.json"
            
            try:
                # Download manifest
                manifest_content = minio_storage.download_file_content(
                    bucket_name=qa_config.minio_bucket,
                    object_name=manifest_path
                )
                
                manifest = json.loads(manifest_content)
                return manifest.get("urls", {})
                
            except Exception as e:
                log.warning(f"Could not retrieve manifest for {paper_id}: {e}")
                
                # Fallback: construct URLs based on bucket structure
                base_url = f"{qa_config.minio_endpoint}/{qa_config.minio_bucket}"
                return {
                    "pdf": f"{base_url}/{paper_id}/pdf/{paper_id}.pdf",
                    "metadata": f"{base_url}/{paper_id}/metadata/{paper_id}.json",
                    "markdown": f"{base_url}/{paper_id}/markdown/index.md",
                    "images": f"{base_url}/{paper_id}/images/"
                }
                
        except Exception as e:
            log.error(f"Error getting MinIO URLs for {paper_id}: {e}")
            return {}
    
    async def get_paper_images(self, paper_id: str) -> List[Dict[str, str]]:
        """
        Get list of images for a paper from MinIO.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            List of image information dictionaries
        """
        try:
            from data_pipeline.minio_storage import MinIOStorage
            
            minio_storage = MinIOStorage(endpoint=qa_config.minio_endpoint)
            images_path = f"{paper_id}/images/"
            
            # List objects in images directory
            objects = minio_storage.list_objects(
                bucket_name=qa_config.minio_bucket,
                prefix=images_path
            )
            
            images = []
            for obj in objects:
                if obj.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    # Extract image info from filename: fig_p{page}_{idx}_{sha16}.{ext}
                    filename = obj.split('/')[-1]
                    match = re.match(r'fig_p(\d+)_(\d+)_([a-f0-9]{16})\.(\w+)', filename)
                    
                    if match:
                        page_num, idx, sha16, ext = match.groups()
                        images.append({
                            "filename": filename,
                            "page": int(page_num),
                            "index": int(idx),
                            "sha16": sha16,
                            "extension": ext,
                            "url": f"{qa_config.minio_endpoint}/{qa_config.minio_bucket}/{obj}"
                        })
            
            return images
            
        except Exception as e:
            log.error(f"Error getting images for {paper_id}: {e}")
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
            return {
                "context_chunks": context_text,
                "question": question,
                "num_papers": str(num_papers)
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
