"""
Configuration settings for Paper Search Engine Backend.
"""

import os
from typing import Optional


class Config:
    """Configuration class for ES-based search engine"""

    # Server Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

    # Elasticsearch Configuration
    ES_HOST = os.getenv("ES_HOST", "localhost:9202")
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


# Global configuration instance
config = Config()