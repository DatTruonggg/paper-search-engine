"""
Configuration settings for Paper Search Engine Backend.

Provides a `load_env` utility to load environment variables from common
locations, including the project `.env` and ASTA agent secret files.
"""

import os
from pathlib import Path
from typing import Optional, Iterable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in some deployments
    load_dotenv = None  # type: ignore


def load_env(extra_paths: Optional[Iterable[str]] = None) -> None:
    """Load environment variables from standard and optional locations.

    This function loads variables from:
    - Project root `.env` if present
    - Any additional files provided via `extra_paths`

    Existing environment variables are preserved (no override).
    """

    if load_dotenv is None:
        return

    project_root = Path(__file__).resolve().parents[2]
    default_env = project_root / ".env"
    # Do not override existing env values
    try:
        load_dotenv(dotenv_path=str(default_env), override=False)
    except Exception:
        pass

    if extra_paths:
        for p in extra_paths:
            try:
                load_dotenv(dotenv_path=str(Path(p)), override=False)
            except Exception:
                pass


class Config:
    """Configuration class for ES-based search engine"""

    # Server Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8001"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

    # Elasticsearch Configuration
    ES_HOST = os.getenv("ES_HOST", "http://103.3.247.120:9200")
    ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "papers")

    # BGE Embedding Configuration
    BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    BGE_CACHE_DIR = os.getenv("BGE_CACHE_DIR", "./models")

    # Search Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
    DEFAULT_SEARCH_MODE = os.getenv("DEFAULT_SEARCH_MODE", "hybrid")

    # Document Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Data Paths
    MARKDOWN_DATA_PATH = os.getenv("MARKDOWN_DATA_PATH", "./data/processed/markdown")
    JSON_METADATA_PATH = os.getenv("JSON_METADATA_PATH", "/Users/admin/code/cazoodle/data/pdfs")
    PDF_LOCAL_DIR = os.getenv("PDF_LOCAL_DIR", "/Users/admin/code/cazoodle/data/pdfs")

    # MinIO / Object Storage
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://103.3.247.120:9002")

    # LlamaIndex Agent Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")

    # LLM Model Configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_MINI_MODEL = os.getenv("OPENAI_MINI_MODEL", "gpt-4o-mini")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")

    # Agent Behavior Settings
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4000"))
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

    # Memory and Context Settings
    AGENT_MEMORY_TOKEN_LIMIT = int(os.getenv("AGENT_MEMORY_TOKEN_LIMIT", "3000"))
    AGENT_CONTEXT_WINDOW = int(os.getenv("AGENT_CONTEXT_WINDOW", "5"))  # Number of past exchanges to keep

    # Tool Configuration
    ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"
    WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    PAPER_SEARCH_MAX_RESULTS = int(os.getenv("PAPER_SEARCH_MAX_RESULTS", "20"))

    # Performance and Reliability Settings
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    WEB_SEARCH_TIMEOUT_SECONDS = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10"))
    AGENT_RETRY_ATTEMPTS = int(os.getenv("AGENT_RETRY_ATTEMPTS", "3"))

    # Feature Flags
    ENABLE_TOOL_CACHING = os.getenv("ENABLE_TOOL_CACHING", "false").lower() == "true"
    ENABLE_CONVERSATION_PERSISTENCE = os.getenv("ENABLE_CONVERSATION_PERSISTENCE", "false").lower() == "true"


# Global configuration instance
config = Config()