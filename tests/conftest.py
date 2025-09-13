"""
Test configuration and fixtures for paper search engine tests.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.document_chunker import DocumentChunker
from data_pipeline.es_indexer import ESIndexer
# Note: SearchService import removed to avoid config issues


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_paper_content():
    """Sample paper content for testing"""
    return """
# Attention Is All You Need

Submitted by
Ashish Vaswani, Noam Shazeer, Niki Parmar

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.

## Methodology

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

## Results

Experimental results demonstrate the superiority of our approach. We achieve state-of-the-art performance on multiple benchmarks including WMT 2014 English-to-German and English-to-French translation tasks.

## Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.
"""


@pytest.fixture
def sample_paper_metadata():
    """Sample paper metadata for testing"""
    return {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "categories": ["cs.CL"],
        "publish_date": "2017-06-12"
    }


@pytest.fixture
def mock_embedder():
    """Mock BGE embedder for testing"""
    embedder = Mock(spec=BGEEmbedder)
    embedder.embedding_dim = 1024
    embedder.model_name = "BAAI/bge-large-en-v1.5"

    # Mock encode method to return random embeddings
    def mock_encode(texts, **kwargs):
        if isinstance(texts, str):
            return np.random.randn(1024).astype(np.float32)
        else:
            return np.random.randn(len(texts), 1024).astype(np.float32)

    embedder.encode = mock_encode
    embedder.encode_queries = mock_encode

    return embedder


@pytest.fixture
def mock_elasticsearch():
    """Mock Elasticsearch client for testing"""
    es_mock = Mock()
    es_mock.ping.return_value = True
    es_mock.indices.create.return_value = {"acknowledged": True}
    es_mock.indices.exists.return_value = False
    es_mock.indices.delete.return_value = {"acknowledged": True}
    es_mock.index.return_value = {"_id": "test_id", "result": "created"}
    es_mock.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [
                {
                    "_id": "test_id",
                    "_score": 1.0,
                    "_source": {
                        "paper_id": "test_001",
                        "title": "Test Paper",
                        "authors": ["Test Author"],
                        "abstract": "Test abstract",
                        "categories": ["cs.AI"]
                    }
                }
            ]
        }
    }
    es_mock.get.return_value = {
        "_source": {
            "paper_id": "test_001",
            "title": "Test Paper"
        }
    }
    es_mock.count.return_value = {"count": 1}
    es_mock.indices.stats.return_value = {
        "indices": {
            "papers": {
                "total": {
                    "store": {"size_in_bytes": 1024}
                }
            }
        }
    }

    return es_mock


@pytest.fixture
def sample_markdown_files(test_data_dir):
    """Create sample markdown files for testing"""
    markdown_dir = test_data_dir / "markdown"
    markdown_dir.mkdir()

    # Create sample files
    papers = [
        ("2210.14275.md", "# Test Paper 1\n\nThis is a test paper about NLP.\n\n## Abstract\n\nThis paper explores natural language processing."),
        ("1706.03762.md", "# Attention Is All You Need\n\nSubmitted by Ashish Vaswani\n\n## Abstract\n\nThe Transformer architecture."),
        ("2301.06375.md", "# Another Test Paper\n\nBy Test Author\n\n## Abstract\n\nThis is another test paper.")
    ]

    created_files = []
    for filename, content in papers:
        file_path = markdown_dir / filename
        file_path.write_text(content)
        created_files.append(file_path)

    return created_files


@pytest.fixture
def chunker():
    """Document chunker instance for testing"""
    return DocumentChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def mock_indexer(mock_elasticsearch):
    """Mock ES indexer for testing"""
    indexer = Mock(spec=ESIndexer)
    indexer.es = mock_elasticsearch
    indexer.index_name = "test_papers"
    indexer.embedding_dim = 1024

    # Mock methods
    indexer.create_index.return_value = True
    indexer.index_document.return_value = "test_id"
    indexer.bulk_index.return_value = None
    indexer.search.return_value = [
        {
            "paper_id": "test_001",
            "title": "Test Paper",
            "score": 1.0
        }
    ]
    indexer.get_document.return_value = {"paper_id": "test_001"}
    indexer.get_index_stats.return_value = {
        "document_count": 1,
        "index_size_mb": 1.0
    }

    return indexer


@pytest.fixture
def mock_search_service(mock_embedder, mock_indexer):
    """Mock search service for testing"""
    service = Mock()  # Generic mock without spec
    service.embedder = mock_embedder
    service.indexer = mock_indexer

    # Mock search results
    service.search.return_value = [
        {
            "paper_id": "test_001",
            "title": "Test Paper",
            "authors": ["Test Author"],
            "abstract": "Test abstract",
            "score": 1.0,
            "categories": ["cs.AI"],
            "publish_date": "2023-01-01",
            "pdf_path": None,
            "markdown_path": "/test/path.md"
        }
    ]

    service.get_paper_details.return_value = {"paper_id": "test_001"}
    service.suggest_papers.return_value = []
    service.get_search_stats.return_value = {
        "total_papers": 1,
        "index_size_mb": 1.0
    }

    return service


# Test configuration
TEST_ES_HOST = "localhost:9202"
TEST_INDEX_NAME = "test_papers"

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "es_host": TEST_ES_HOST,
        "index_name": TEST_INDEX_NAME,
        "bge_model": "BAAI/bge-large-en-v1.5",
        "chunk_size": 100,
        "chunk_overlap": 20
    }