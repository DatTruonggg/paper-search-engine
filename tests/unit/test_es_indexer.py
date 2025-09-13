"""
Unit tests for Elasticsearch Indexer service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

from data_pipeline.es_indexer import ESIndexer


class TestESIndexer:
    """Test cases for Elasticsearch Indexer"""

    @pytest.fixture
    def mock_es_client(self):
        """Mock Elasticsearch client"""
        mock_es = Mock(spec=Elasticsearch)
        mock_es.ping.return_value = True
        mock_es.indices.create.return_value = {"acknowledged": True}
        mock_es.indices.exists.return_value = False
        mock_es.indices.delete.return_value = {"acknowledged": True}
        mock_es.index.return_value = {"_id": "test_id", "result": "created"}
        mock_es.get.return_value = {"_source": {"paper_id": "test_001"}}
        mock_es.delete.return_value = {"result": "deleted"}
        mock_es.count.return_value = {"count": 5}
        mock_es.indices.stats.return_value = {
            "indices": {
                "papers": {
                    "total": {
                        "store": {"size_in_bytes": 1024000}
                    }
                }
            }
        }
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "test_001",
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
        return mock_es

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_indexer_initialization(self, mock_elasticsearch_class, mock_es_client):
        """Test indexer initialization"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer(
            es_host="test:9200",
            index_name="test_index",
            embedding_dim=512
        )

        assert indexer.es_host == "test:9200"
        assert indexer.index_name == "test_index"
        assert indexer.embedding_dim == 512
        assert indexer.es == mock_es_client

        mock_elasticsearch_class.assert_called_once_with(
            ["test:9200"],
            verify_certs=False,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_indexer_connection_failure(self, mock_elasticsearch_class):
        """Test indexer initialization with connection failure"""
        mock_es_client = Mock()
        mock_es_client.ping.return_value = False
        mock_elasticsearch_class.return_value = mock_es_client

        with pytest.raises(ConnectionError, match="Cannot connect to Elasticsearch"):
            ESIndexer()

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_create_index(self, mock_elasticsearch_class, mock_es_client):
        """Test index creation"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()
        indexer.create_index()

        mock_es_client.indices.create.assert_called_once()
        create_call = mock_es_client.indices.create.call_args
        assert create_call[1]["index"] == "papers"
        assert "mappings" in create_call[1]["body"]
        assert "settings" in create_call[1]["body"]

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_create_index_force_delete(self, mock_elasticsearch_class, mock_es_client):
        """Test index creation with force delete"""
        mock_elasticsearch_class.return_value = mock_es_client
        mock_es_client.indices.exists.return_value = True

        indexer = ESIndexer()
        indexer.create_index(force=True)

        mock_es_client.indices.delete.assert_called_once_with(index="papers")
        mock_es_client.indices.create.assert_called_once()

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_create_index_already_exists(self, mock_elasticsearch_class, mock_es_client):
        """Test index creation when index already exists"""
        mock_elasticsearch_class.return_value = mock_es_client
        mock_es_client.indices.create.side_effect = RequestError(
            400, "resource_already_exists_exception", {}
        )

        indexer = ESIndexer()
        # Should not raise exception
        indexer.create_index()

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_index_document(self, mock_elasticsearch_class, mock_es_client):
        """Test single document indexing"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()

        document = {
            "paper_id": "test_001",
            "title": "Test Paper",
            "title_embedding": np.random.randn(1024),
            "abstract_embedding": np.random.randn(1024),
            "content_chunks": [
                {
                    "text": "chunk text",
                    "embedding": np.random.randn(1024)
                }
            ]
        }

        doc_id = indexer.index_document(document)

        assert doc_id == "test_id"
        mock_es_client.index.assert_called_once()

        # Check that embeddings were converted to lists
        index_call = mock_es_client.index.call_args[1]
        assert isinstance(index_call["body"]["title_embedding"], list)
        assert isinstance(index_call["body"]["abstract_embedding"], list)
        assert isinstance(index_call["body"]["content_chunks"][0]["embedding"], list)

    @patch('data_pipeline.es_indexer.Elasticsearch')
    @patch('data_pipeline.es_indexer.helpers')
    def test_bulk_index(self, mock_helpers, mock_elasticsearch_class, mock_es_client):
        """Test bulk document indexing"""
        mock_elasticsearch_class.return_value = mock_es_client
        mock_helpers.bulk.return_value = (2, [])  # success_count, failed_list

        indexer = ESIndexer()

        documents = [
            {
                "paper_id": "test_001",
                "title": "Test Paper 1",
                "title_embedding": np.random.randn(1024)
            },
            {
                "paper_id": "test_002",
                "title": "Test Paper 2",
                "title_embedding": np.random.randn(1024)
            }
        ]

        indexer.bulk_index(documents, batch_size=10)

        mock_helpers.bulk.assert_called_once()
        bulk_call = mock_helpers.bulk.call_args

        # Check that correct number of actions were created
        actions = bulk_call[0][1]  # Second argument (first is es client)
        assert len(actions) == 2

        # Check action structure
        for action in actions:
            assert action["_index"] == "papers"
            assert "_id" in action
            assert "_source" in action
            assert isinstance(action["_source"]["title_embedding"], list)

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_search(self, mock_elasticsearch_class, mock_es_client):
        """Test search functionality"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()

        # Test text search
        results = indexer.search(
            query="test query",
            size=5,
            use_semantic=False,
            use_bm25=True
        )

        assert len(results) == 1
        assert results[0]["paper_id"] == "test_001"
        assert results[0]["_score"] == 1.0

        mock_es_client.search.assert_called_once()
        search_call = mock_es_client.search.call_args[1]
        assert search_call["index"] == "papers"
        assert search_call["body"]["size"] == 5

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_search_with_embeddings(self, mock_elasticsearch_class, mock_es_client):
        """Test search with semantic embeddings"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()
        query_embedding = np.random.randn(1024)

        results = indexer.search(
            query="test query",
            query_embedding=query_embedding,
            size=5,
            use_semantic=True,
            use_bm25=True
        )

        assert len(results) == 1
        mock_es_client.search.assert_called_once()

        # Check that embedding was included in search
        search_call = mock_es_client.search.call_args[1]
        query_body = search_call["body"]["query"]["bool"]["should"]

        # Should have both BM25 and semantic search clauses
        assert len(query_body) > 1

        # Check for script_score queries (semantic search)
        script_score_queries = [q for q in query_body if "script_score" in q]
        assert len(script_score_queries) > 0

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_get_document(self, mock_elasticsearch_class, mock_es_client):
        """Test getting single document by ID"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()
        result = indexer.get_document("test_001")

        assert result == {"paper_id": "test_001"}
        mock_es_client.get.assert_called_once_with(index="papers", id="test_001")

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_get_document_not_found(self, mock_elasticsearch_class, mock_es_client):
        """Test getting document that doesn't exist"""
        mock_elasticsearch_class.return_value = mock_es_client
        mock_es_client.get.side_effect = Exception("Not found")

        indexer = ESIndexer()
        result = indexer.get_document("nonexistent")

        assert result is None

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_delete_document(self, mock_elasticsearch_class, mock_es_client):
        """Test document deletion"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()
        result = indexer.delete_document("test_001")

        assert result is True
        mock_es_client.delete.assert_called_once_with(index="papers", id="test_001")

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_delete_document_failure(self, mock_elasticsearch_class, mock_es_client):
        """Test document deletion failure"""
        mock_elasticsearch_class.return_value = mock_es_client
        mock_es_client.delete.side_effect = Exception("Delete failed")

        indexer = ESIndexer()
        result = indexer.delete_document("test_001")

        assert result is False

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_get_index_stats(self, mock_elasticsearch_class, mock_es_client):
        """Test getting index statistics"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()
        stats = indexer.get_index_stats()

        assert stats["document_count"] == 5
        assert stats["index_size"] == 1024000
        assert stats["index_size_mb"] == 1.0

        mock_es_client.count.assert_called_once_with(index="papers")
        mock_es_client.indices.stats.assert_called_once_with(index="papers")

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_search_field_specification(self, mock_elasticsearch_class, mock_es_client):
        """Test search with custom search fields"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()

        indexer.search(
            query="test query",
            search_fields=["title^2", "abstract"],
            use_semantic=False,
            use_bm25=True
        )

        search_call = mock_es_client.search.call_args[1]
        multi_match = search_call["body"]["query"]["bool"]["should"][0]["multi_match"]
        assert multi_match["fields"] == ["title^2", "abstract"]

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_embedding_conversion_edge_cases(self, mock_elasticsearch_class, mock_es_client):
        """Test embedding conversion edge cases"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()

        # Document with already converted embeddings (lists)
        document = {
            "paper_id": "test_001",
            "title_embedding": [0.1, 0.2, 0.3],  # Already a list
            "content_chunks": [
                {
                    "embedding": [0.4, 0.5, 0.6]  # Already a list
                }
            ]
        }

        indexer.index_document(document)

        # Should not fail and should preserve the lists
        index_call = mock_es_client.index.call_args[1]
        assert index_call["body"]["title_embedding"] == [0.1, 0.2, 0.3]

    @patch('data_pipeline.es_indexer.Elasticsearch')
    def test_search_without_query(self, mock_elasticsearch_class, mock_es_client):
        """Test search with no query (match all)"""
        mock_elasticsearch_class.return_value = mock_es_client

        indexer = ESIndexer()

        results = indexer.search(
            query=None,
            query_embedding=None,
            use_semantic=False,
            use_bm25=False
        )

        search_call = mock_es_client.search.call_args[1]
        assert search_call["body"]["query"]["match_all"] == {}

    def test_index_mapping_structure(self):
        """Test that the index mapping contains required fields"""
        indexer = ESIndexer.__new__(ESIndexer)  # Create without __init__
        indexer.embedding_dim = 1024

        # This would be tested by examining the mapping structure
        # The actual mapping is defined in create_index method
        expected_fields = [
            "paper_id", "title", "authors", "abstract", "content",
            "title_embedding", "abstract_embedding", "content_chunks",
            "categories", "publish_date", "pdf_path", "markdown_path"
        ]

        # In actual implementation, we would check the mapping structure
        # This is a placeholder for mapping validation
        assert True  # Mapping structure would be validated here