"""
Configuration for QA Agent system following llama_agent design pattern.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class QAAgentConfig(BaseModel):
    """Configuration for QA Agent (Gemini-only)."""

    # Gemini LLM Configuration
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_model: str = Field(default_factory=lambda: os.getenv("QA_AGENT_GOOGLE_MODEL", "gemini-1.5-pro"))

    # QA Agent Behavior Settings
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_MAX_TOKENS", "4000")))
    temperature: float = Field(default_factory=lambda: float(os.getenv("QA_AGENT_TEMPERATURE", "0.1")))
    max_iterations: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_MAX_ITERATIONS", "5")))
    verbose: bool = Field(default_factory=lambda: os.getenv("QA_AGENT_VERBOSE", "true").lower() == "true")

    # Context and Retrieval Settings
    max_context_chunks: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_MAX_CONTEXT_CHUNKS", "10")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_CHUNK_OVERLAP", "2")))
    rerank_results: bool = Field(default_factory=lambda: os.getenv("QA_AGENT_RERANK_RESULTS", "true").lower() == "true")
    include_images: bool = Field(default_factory=lambda: os.getenv("QA_AGENT_INCLUDE_IMAGES", "true").lower() == "true")

    # Performance Settings
    timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_TIMEOUT_SECONDS", "30")))
    retry_attempts: int = Field(default_factory=lambda: int(os.getenv("QA_AGENT_RETRY_ATTEMPTS", "3")))

    # Service Endpoints
    es_host: str = Field(default_factory=lambda: os.getenv("QA_AGENT_ES_HOST", "http://103.3.247.120:9202"))
    minio_endpoint: str = Field(default_factory=lambda: os.getenv("QA_AGENT_MINIO_ENDPOINT", "103.3.247.120:9002"))
    minio_bucket: str = Field(default_factory=lambda: os.getenv("QA_AGENT_MINIO_BUCKET", "papers"))
    class Config:
        env_prefix = "QA_AGENT_"


# Global configuration instance
qa_config = QAAgentConfig()

# System Prompts
SINGLE_PAPER_QA_PROMPT = """
You are an expert research assistant specializing in academic paper analysis.
You will be given a question about a specific research paper and relevant context chunks from that paper.

Instructions:
1. Read the context carefully and answer the question as if you are directly talking to the user. 
2. Your answer must be based ONLY on the provided context chunks.
3. If the context doesn't contain enough information, say so clearly and naturally.
4. Refer to specific chunks when needed, but explain in plain human language (avoid rigid citations).
5. Be precise and factual – do not make assumptions beyond what's in the context.
6. If multiple chunks provide conflicting information, acknowledge it and explain the difference.
7. Do not include chunk IDs, figures, or paper IDs in the output.
8. Write as if you are an intelligent agent helping the user understand the paper, not just quoting text.
9. Output format: 
{paper_title} - (Answer)

Context Chunks:
{context_chunks}

User Question: {question}

Answer:"""

MULTI_PAPER_QA_PROMPT = """
You are an expert research assistant specializing in comparative analysis of academic papers.
You will be given a question and relevant context chunks from multiple research papers.

Instructions:
1. Read the context carefully and answer the question in a natural, conversational way.
2. Synthesize information across the provided papers instead of just repeating text.
3. Compare and contrast findings across different papers in plain language, pointing out consensus, disagreements, or gaps.
4. If useful, mention which paper found what, but phrase it like a human (e.g., "One paper suggests..., while another points out...").
5. Be objective and analytical, but also approachable – like a knowledgeable assistant guiding the user.
6. If the context doesn't contain enough information, say so directly and naturally.
7. Do not include raw chunk IDs, paper IDs, or figures in the answer.
8. Write as if you are reasoning aloud, showing the user how you connect the pieces of information.
9. Output format: 
{paper_title} - (Answer)

Context chunks from {num_papers} papers:
{context_chunks}

Question: {question}

Answer:"""

IMAGE_ANALYSIS_PROMPT = """
Analyze the following image from a research paper and provide a detailed description of what it shows.
Focus on:
1. The main visual elements (charts, graphs, diagrams, etc.)
2. Key data points, trends, or patterns visible
3. Labels, axes, legends, and other textual elements
4. The relationship between different visual components
5. Any conclusions that can be drawn from the visual

Image URL: {image_url}
Paper: {paper_title}
Section: {section_path}

Description:"""

# NOTE: Removed legacy QAConfig (untyped) that overwrote qa_config with None values.
# qa_config now strictly refers to QAAgentConfig instance above.
