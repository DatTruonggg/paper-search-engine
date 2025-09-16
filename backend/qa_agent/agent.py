"""
Main QA Agent for single-paper and multi-paper question answering.
Following llama_agent design pattern.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from .config import qa_config
from .tools import QARetrievalTool, ImageAnalysisTool, ContextBuilder, ContextChunk

from logs import log

@dataclass
class QAResponse:
    """Response from QA agent"""
    answer: str
    context_chunks: List[ContextChunk]
    image_descriptions: Dict[str, str]
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float


class QAAgent:
    """
    Intelligent QA Agent for answering questions about research papers.
    
    This agent:
    1. Retrieves relevant context chunks from papers
    2. Analyzes images when available
    3. Builds comprehensive context for LLM
    4. Generates accurate answers with citations
    
    Supports:
    - Single-paper QA: Answer questions about a specific paper
    - Multi-paper QA: Answer questions across multiple papers
    - Search results QA: Answer using papers from search results
    """
    
    def __init__(self, es_service=None, llm=None):
        """
        Initialize QA Agent.
        
        Args:
            es_service: Optional ElasticsearchSearchService instance
            llm: Optional LLM instance
        """
        # Initialize LLM
        if llm is None:
            if qa_config.default_llm_provider == "openai" and qa_config.openai_api_key:
                self.llm = OpenAI(
                    model=qa_config.openai_model,
                    api_key=qa_config.openai_api_key,
                    temperature=qa_config.temperature,
                    max_tokens=qa_config.max_tokens,
                    timeout=qa_config.timeout_seconds
                )
            elif qa_config.google_api_key:
                self.llm = Gemini(
                    model=qa_config.google_model,
                    api_key=qa_config.google_api_key,
                    temperature=qa_config.temperature,
                    timeout=qa_config.timeout_seconds
                )
            else:
                raise ValueError("No valid LLM configuration found")
        else:
            self.llm = llm
        
        # Initialize tools
        self.retrieval_tool = QARetrievalTool(es_service)
        self.image_tool = ImageAnalysisTool()
        self.context_builder = ContextBuilder()
        
        log.info(f"QA Agent initialized with {qa_config.default_llm_provider} LLM")
    
    async def answer_single_paper_question(
        self, 
        paper_id: str, 
        question: str,
        max_chunks: int = None
    ) -> QAResponse:
        """
        Answer a question about a specific paper.
        
        Args:
            paper_id: ID of the paper to ask about
            question: Question to answer
            max_chunks: Maximum number of context chunks to use
            
        Returns:
            QAResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        try:
            log.info(f"Answering single-paper question for paper {paper_id}: {question}")
            
            # Retrieve relevant context chunks
            context_chunks = await self.retrieval_tool.retrieve_single_paper_context(
                paper_id, question, max_chunks
            )
            
            if not context_chunks:
                return QAResponse(
                    answer="I couldn't find any relevant information in the specified paper to answer your question.",
                    context_chunks=[],
                    image_descriptions={},
                    sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Get MinIO URLs for the paper
            minio_urls = await self.retrieval_tool.get_paper_minio_urls(paper_id)
            
            # Analyze images if enabled
            image_descriptions = {}
            if qa_config.include_images:
                async with self.image_tool:
                    image_descriptions = await self.image_tool.analyze_images_in_chunks(context_chunks)
            
            # Build context for prompt
            prompt_data = self.context_builder.format_context_for_prompt(
                context_chunks, question, is_multi_paper=False, 
                image_descriptions=image_descriptions,
                papers_minio_urls={paper_id: minio_urls} if minio_urls else None
            )
            
            # Generate answer using LLM
            prompt = qa_config.single_paper_prompt.format(**prompt_data)
            
            if qa_config.verbose:
                log.info(f"Generated prompt for single-paper QA: {prompt[:200]}...")
            
            response = await self.llm.acomplete(prompt)
            answer = str(response)
            
            # Build sources list
            sources = []
            for chunk in context_chunks:
                sources.append({
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "chunk_index": chunk.chunk_index,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "relevance_score": chunk.relevance_score
                })
            
            # Calculate confidence score based on relevance scores
            confidence_score = sum(chunk.relevance_score for chunk in context_chunks) / len(context_chunks)
            
            processing_time = time.time() - start_time
            
            log.info(f"Single-paper QA completed in {processing_time:.2f}s with confidence {confidence_score:.3f}")
            
            return QAResponse(
                answer=answer,
                context_chunks=context_chunks,
                image_descriptions=image_descriptions,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            log.error(f"Error in single-paper QA: {e}")
            return QAResponse(
                answer=f"Error processing your question: {str(e)}",
                context_chunks=[],
                image_descriptions={},
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    async def answer_multi_paper_question(
        self, 
        paper_ids: List[str], 
        question: str,
        max_chunks_per_paper: int = 3
    ) -> QAResponse:
        """
        Answer a question across multiple papers.
        
        Args:
            paper_ids: List of paper IDs to search across
            question: Question to answer
            max_chunks_per_paper: Maximum chunks per paper
            
        Returns:
            QAResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        try:
            log.info(f"Answering multi-paper question across {len(paper_ids)} papers: {question}")
            
            # Retrieve relevant context chunks from all papers
            context_chunks = await self.retrieval_tool.retrieve_multi_paper_context(
                paper_ids, question, max_chunks_per_paper
            )
            
            if not context_chunks:
                return QAResponse(
                    answer="I couldn't find any relevant information in the specified papers to answer your question.",
                    context_chunks=[],
                    image_descriptions={},
                    sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Get MinIO URLs for all papers
            papers_minio_urls = {}
            for paper_id in paper_ids:
                minio_urls = await self.retrieval_tool.get_paper_minio_urls(paper_id)
                if minio_urls:
                    papers_minio_urls[paper_id] = minio_urls
            
            # Analyze images if enabled
            image_descriptions = {}
            if qa_config.include_images:
                async with self.image_tool:
                    image_descriptions = await self.image_tool.analyze_images_in_chunks(context_chunks)
            
            # Build context for prompt
            prompt_data = self.context_builder.format_context_for_prompt(
                context_chunks, question, is_multi_paper=True, 
                image_descriptions=image_descriptions,
                papers_minio_urls=papers_minio_urls if papers_minio_urls else None
            )
            
            # Generate answer using LLM
            prompt = qa_config.multi_paper_prompt.format(**prompt_data)
            
            if qa_config.verbose:
                log.info(f"Generated prompt for multi-paper QA: {prompt[:200]}...")
            
            response = await self.llm.acomplete(prompt)
            answer = str(response)
            
            # Build sources list
            sources = []
            for chunk in context_chunks:
                sources.append({
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "chunk_index": chunk.chunk_index,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "relevance_score": chunk.relevance_score
                })
            
            # Calculate confidence score based on relevance scores
            confidence_score = sum(chunk.relevance_score for chunk in context_chunks) / len(context_chunks)
            
            processing_time = time.time() - start_time
            
            log.info(f"Multi-paper QA completed in {processing_time:.2f}s with confidence {confidence_score:.3f}")
            
            return QAResponse(
                answer=answer,
                context_chunks=context_chunks,
                image_descriptions=image_descriptions,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            log.error(f"Error in multi-paper QA: {e}")
            return QAResponse(
                answer=f"Error processing your question: {str(e)}",
                context_chunks=[],
                image_descriptions={},
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    async def answer_search_results_question(
        self, 
        search_query: str, 
        question: str,
        max_papers: int = 5,
        max_chunks_per_paper: int = 2
    ) -> QAResponse:
        """
        Answer a question using papers from a search query.
        
        Args:
            search_query: Original search query that returned papers
            question: Question to answer
            max_papers: Maximum number of papers to include
            max_chunks_per_paper: Maximum chunks per paper
            
        Returns:
            QAResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        try:
            log.info(f"Answering question using search results for query: {search_query}")
            
            # Retrieve context from search results
            context_chunks = await self.retrieval_tool.retrieve_search_results_context(
                search_query, question, max_papers, max_chunks_per_paper
            )
            
            if not context_chunks:
                return QAResponse(
                    answer="I couldn't find any relevant information in the search results to answer your question.",
                    context_chunks=[],
                    image_descriptions={},
                    sources=[],
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Get MinIO URLs for all papers in search results
            papers_minio_urls = {}
            unique_paper_ids = list(set(chunk.paper_id for chunk in context_chunks))
            for paper_id in unique_paper_ids:
                minio_urls = await self.retrieval_tool.get_paper_minio_urls(paper_id)
                if minio_urls:
                    papers_minio_urls[paper_id] = minio_urls
            
            # Analyze images if enabled
            image_descriptions = {}
            if qa_config.include_images:
                async with self.image_tool:
                    image_descriptions = await self.image_tool.analyze_images_in_chunks(context_chunks)
            
            # Build context for prompt
            prompt_data = self.context_builder.format_context_for_prompt(
                context_chunks, question, is_multi_paper=True, 
                image_descriptions=image_descriptions,
                papers_minio_urls=papers_minio_urls if papers_minio_urls else None
            )
            
            # Generate answer using LLM
            prompt = qa_config.multi_paper_prompt.format(**prompt_data)
            
            if qa_config.verbose:
                log.info(f"Generated prompt for search results QA: {prompt[:200]}...")
            
            response = await self.llm.acomplete(prompt)
            answer = str(response)
            
            # Build sources list
            sources = []
            for chunk in context_chunks:
                sources.append({
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "chunk_index": chunk.chunk_index,
                    "section_path": chunk.section_path,
                    "page_number": chunk.page_number,
                    "relevance_score": chunk.relevance_score
                })
            
            # Calculate confidence score based on relevance scores
            confidence_score = sum(chunk.relevance_score for chunk in context_chunks) / len(context_chunks)
            
            processing_time = time.time() - start_time
            
            log.info(f"Search results QA completed in {processing_time:.2f}s with confidence {confidence_score:.3f}")
            
            return QAResponse(
                answer=answer,
                context_chunks=context_chunks,
                image_descriptions=image_descriptions,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            log.error(f"Error in search results QA: {e}")
            return QAResponse(
                answer=f"Error processing your question: {str(e)}",
                context_chunks=[],
                image_descriptions={},
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of QA agent components.
        
        Returns:
            Dictionary with health status
        """
        health = {
            "status": "healthy",
            "components": {}
        }
        
        # Check LLM
        try:
            test_response = await self.llm.acomplete("Test")
            health["components"]["llm"] = {
                "status": "healthy",
                "provider": qa_config.default_llm_provider,
                "model": qa_config.openai_model if qa_config.default_llm_provider == "openai" else qa_config.google_model
            }
        except Exception as e:
            health["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        # Check Elasticsearch
        try:
            es_health = self.retrieval_tool.es_service.health_check()
            health["components"]["elasticsearch"] = es_health
            if es_health["status"] != "healthy":
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["elasticsearch"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
        
        # Check MinIO (basic connectivity)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{qa_config.minio_endpoint}/minio/health/live", timeout=5) as response:
                    if response.status == 200:
                        health["components"]["minio"] = {
                            "status": "healthy",
                            "endpoint": qa_config.minio_endpoint
                        }
                    else:
                        health["components"]["minio"] = {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
                        health["status"] = "degraded"
        except Exception as e:
            health["components"]["minio"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        return health
