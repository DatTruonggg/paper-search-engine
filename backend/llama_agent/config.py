"""
Configuration for LlamaIndex paper search agent.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LlamaAgentConfig(BaseModel):
    """Configuration for LlamaIndex agent"""

    # Gemini configuration
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = Field(default="gemini-2.5-flash")

    # Agent configuration
    max_iterations: int = Field(default=3, description="Maximum search iterations")
    max_results_per_search: int = Field(default=20, description="Max results per ES query")
    min_relevance_score: float = Field(default=0.5, description="Minimum relevance score")

    # Search configuration
    default_search_mode: str = Field(default="hybrid", description="Default search mode")
    enable_query_analysis: bool = Field(default=True, description="Enable query analysis")
    enable_result_reranking: bool = Field(default=True, description="Enable result reranking")
    quality_min_score: float = Field(default=0.2, description="Min acceptable quality score before fallback")
    quality_min_results: int = Field(default=5, description="Min acceptable number of results before fallback")

    # Response configuration
    response_max_papers: int = Field(default=10, description="Maximum papers in response")
    include_explanations: bool = Field(default=True, description="Include relevance explanations")
    include_summaries: bool = Field(default=True, description="Include paper summaries")

    # Timeout configuration
    search_timeout: int = Field(default=60, description="Search timeout in seconds")
    llm_timeout: int = Field(default=30, description="LLM call timeout in seconds")

    # Debug configuration
    verbose: bool = Field(default=False, description="Enable verbose logging")

    class Config:
        env_prefix = "LLAMA_AGENT_"


# Global configuration instance
llama_config = LlamaAgentConfig()


# Enhanced prompt for agent tools
AGENT_SYSTEM_PROMPT = """You are an intelligent paper search agent. Your goal is to find the most relevant academic papers for user queries.

SEARCH REFINEMENT STRATEGIES:
1. PARAPHRASE: If initial results are poor, try alternative terms:
   - "RAG" → "retrieval augmented generation" or "neural retrieval"
   - "LLM" → "large language model" or "transformer model"
   - "NLP" → "natural language processing"

2. APPLY FILTERS: Use filters to narrow down results:
   - Date filters: Use recent=True for latest research
   - Author filters: If user mentions specific researchers
   - Category filters: cs.CL for NLP, cs.AI for AI papers

3. ADJUST SEARCH MODE:
   - hybrid: Balanced BM25 + semantic (default)
   - semantic: For conceptual/meaning-based search
   - bm25: For exact keyword matching
   - title_only: For finding specific paper titles

4. BROADEN/NARROW:
   - If too few results: Use broader, more general terms
   - If too many irrelevant results: Add specific constraints

EVALUATION CRITERIA:
- Relevance: Do titles/abstracts match the query intent?
- Diversity: Coverage of different approaches/aspects
- Quality: Reputable venues, good citation counts
- Completeness: Sufficient breadth of topic coverage

Always aim to provide the most relevant and comprehensive results possible."""