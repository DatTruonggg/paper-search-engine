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

    # Response configuration
    response_max_papers: int = Field(default=10, description="Maximum papers in response")
    include_explanations: bool = Field(default=True, description="Include relevance explanations")
    include_summaries: bool = Field(default=True, description="Include paper summaries")

    # Timeout configuration
    search_timeout: int = Field(default=30, description="Search timeout in seconds")
    llm_timeout: int = Field(default=20, description="LLM call timeout in seconds")

    # Debug configuration
    verbose: bool = Field(default=False, description="Enable verbose logging")

    class Config:
        env_prefix = "LLAMA_AGENT_"


# Global configuration instance
llama_config = LlamaAgentConfig()


# Prompt templates
QUERY_ANALYSIS_PROMPT = """Analyze the following research paper search query and extract key information.

Query: {query}

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanation or additional text.

{{
    "keywords": ["keyword1", "keyword2"],
    "paper_titles": [],
    "authors": [],
    "date_range": {{"from": null, "to": null}},
    "categories": [],
    "query_type": "broad_search",
    "search_strategy": "Standard keyword search"
}}"""

RESULT_EVALUATION_PROMPT = """Evaluate the quality of these search results for the query: {query}

Results: {results}

Analyze the results based on:
1. RELEVANCE: How well do titles and abstracts match the query intent?
2. DIVERSITY: Are results covering different aspects/approaches of the topic?
3. QUALITY: Are these from reputable venues/authors with good citations?
4. COMPLETENESS: Do we have sufficient coverage of the topic?

If refinement is needed, suggest specific strategies:
- PARAPHRASE: Rephrase query with synonyms/alternative terms
- FILTER: Apply author, date, or category filters
- BROADEN: Use more general terms if too specific
- NARROW: Add specific constraints if too broad
- MODE_CHANGE: Switch search modes (hybrid/semantic/bm25/title_only)

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any explanation.

{{
    "quality_score": 0.8,
    "has_sufficient_results": true,
    "needs_refinement": false,
    "refinement_strategy": "PARAPHRASE: Try 'neural retrieval' instead of 'RAG' for broader coverage"
}}"""

RESPONSE_SUMMARY_PROMPT = """Generate a comprehensive response for this paper search query:

Query: {query}

Papers found:
{papers}

Please provide:
1. A brief summary of the search results
2. Top 3-5 most relevant papers with explanations
3. Key insights from the papers
4. Suggestions for further exploration

Format your response in a clear, structured way for the user."""

# Refinement strategy constants
REFINEMENT_STRATEGIES = {
    "PARAPHRASE": {
        "rag": ["retrieval augmented generation", "neural retrieval", "document retrieval", "knowledge retrieval"],
        "llm": ["large language model", "language model", "neural language model", "transformer model"],
        "nlp": ["natural language processing", "computational linguistics", "text processing"],
        "ai": ["artificial intelligence", "machine learning", "deep learning"],
        "bert": ["bidirectional encoder", "transformer encoder", "pre-trained transformer"],
        "gpt": ["generative pre-trained transformer", "autoregressive model", "generative model"]
    },
    "FILTERS": {
        "recent": "2023-01-01",  # Papers from 2023 onwards
        "very_recent": "2024-01-01",  # Papers from 2024 onwards
        "categories": ["cs.CL", "cs.AI", "cs.LG", "cs.IR"],  # Common ML/NLP categories
        "venues": ["ACL", "EMNLP", "ICLR", "NeurIPS", "ICML", "AAAI"]
    },
    "MODES": {
        "broad": "semantic",      # For broad exploratory searches
        "specific": "bm25",       # For specific term matching
        "title": "title_only",    # For finding specific papers
        "balanced": "hybrid"      # Default balanced approach
    }
}

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