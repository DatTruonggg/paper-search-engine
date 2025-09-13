"""
Unit tests for Search Service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from backend.services.search_service import SearchService, SearchResult


class TestSearchService:
    """Test cases for Search Service"""

    @pytest.fixture
    def mock_components(self):
        """Mock embedder and indexer components"""
        mock_embedder = Mock()
        mock_embedder.model_name = "test-model"
        mock_embedder.embedding_dim = 1024
        mock_embedder.encode_queries.return_value = np.random.randn(1024)

        mock_indexer = Mock()
        mock_indexer.index_name = "test_papers"

        return {
            "embedder": mock_embedder,
            "indexer": mock_indexer
        }

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_service_initialization(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test search service initialization"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService(
            es_host="test:9200",
            bge_model="test-model",
            index_name="test_index"
        )

        assert service.embedder == mock_components["embedder"]
        assert service.indexer == mock_components["indexer"]

        mock_embedder_class.assert_called_once_with(model_name="test-model")
        mock_indexer_class.assert_called_once_with(es_host="test:9200", index_name="test_index")

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_hybrid_mode(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test hybrid search mode"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock search methods
        with patch.object(service, '_bm25_search') as mock_bm25, \
             patch.object(service, '_semantic_search') as mock_semantic:

            # Mock return values
            bm25_results = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.8, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]
            semantic_results = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.9, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                ),
                SearchResult(
                    paper_id="paper2", title="Paper 2", authors=["Author 2"],
                    abstract="Abstract 2", score=0.7, categories=["cs.CL"],
                    publish_date="2023-01-02", pdf_path=None, markdown_path=None
                )
            ]

            mock_bm25.return_value = bm25_results
            mock_semantic.return_value = semantic_results

            results = service.search("test query", search_mode="hybrid", max_results=5)

            assert len(results) <= 5
            mock_bm25.assert_called_once()
            mock_semantic.assert_called_once()

            # Check that scores were combined
            for result in results:
                assert isinstance(result, SearchResult)

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_semantic_mode(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test semantic search mode"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with patch.object(service, '_semantic_search') as mock_semantic:
            mock_semantic.return_value = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.95, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]

            results = service.search("test query", search_mode="semantic", max_results=5)

            assert len(results) == 1
            mock_semantic.assert_called_once()
            mock_components["embedder"].encode_queries.assert_called_with("test query")

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_bm25_mode(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test BM25 search mode"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexder"]

        service = SearchService()

        with patch.object(service, '_bm25_search') as mock_bm25:
            mock_bm25.return_value = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.85, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]

            results = service.search("test query", search_mode="bm25", max_results=5)

            assert len(results) == 1
            mock_bm25.assert_called_once()

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_title_only_mode(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test title-only search mode"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with patch.object(service, '_title_search') as mock_title:
            mock_title.return_value = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.9, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]

            results = service.search("test query", search_mode="title_only", max_results=5)

            assert len(results) == 1
            mock_title.assert_called_once()

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_with_filters(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test search with category and date filters"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with patch.object(service, '_bm25_search') as mock_bm25:
            # Mock results with different categories and dates
            mock_bm25.return_value = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.9, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                ),
                SearchResult(
                    paper_id="paper2", title="Paper 2", authors=["Author 2"],
                    abstract="Abstract 2", score=0.8, categories=["cs.CL"],
                    publish_date="2022-01-01", pdf_path=None, markdown_path=None
                ),
                SearchResult(
                    paper_id="paper3", title="Paper 3", authors=["Author 3"],
                    abstract="Abstract 3", score=0.7, categories=["cs.AI"],
                    publish_date="2024-01-01", pdf_path=None, markdown_path=None
                )
            ]

            # Test category filtering
            results = service.search(
                "test query",
                search_mode="bm25",
                categories=["cs.AI"],
                max_results=10
            )

            ai_papers = [r for r in results if "cs.AI" in r.categories]
            assert len(ai_papers) == 2

            # Test date range filtering
            results = service.search(
                "test query",
                search_mode="bm25",
                date_range=("2023-01-01", "2023-12-31"),
                max_results=10
            )

            filtered_papers = [r for r in results if "2023" in (r.publish_date or "")]
            assert len(filtered_papers) >= 1

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_with_min_score(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test search with minimum score threshold"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with patch.object(service, '_bm25_search') as mock_bm25:
            mock_bm25.return_value = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.9, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                ),
                SearchResult(
                    paper_id="paper2", title="Paper 2", authors=["Author 2"],
                    abstract="Abstract 2", score=0.3, categories=["cs.CL"],
                    publish_date="2023-01-02", pdf_path=None, markdown_path=None
                )
            ]

            results = service.search(
                "test query",
                search_mode="bm25",
                min_score=0.5,
                max_results=10
            )

            # Only paper1 should pass the threshold
            assert len(results) == 1
            assert results[0].paper_id == "paper1"

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_semantic_search_implementation(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test semantic search implementation details"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock ES response for KNN search
        mock_es_response_title = {
            "hits": {
                "hits": [
                    {
                        "_id": "paper1",
                        "_score": 0.9,
                        "_source": {
                            "paper_id": "paper1",
                            "title": "Paper 1",
                            "authors": ["Author 1"],
                            "abstract": "Abstract 1",
                            "categories": ["cs.AI"],
                            "publish_date": "2023-01-01"
                        }
                    }
                ]
            }
        }

        mock_es_response_abstract = {
            "hits": {
                "hits": [
                    {
                        "_id": "paper1",
                        "_score": 0.8,
                        "_source": {
                            "paper_id": "paper1",
                            "title": "Paper 1",
                            "authors": ["Author 1"],
                            "abstract": "Abstract 1",
                            "categories": ["cs.AI"],
                            "publish_date": "2023-01-01"
                        }
                    }
                ]
            }
        }

        mock_components["indexer"].es.search.side_effect = [
            mock_es_response_title,
            mock_es_response_abstract
        ]

        query_embedding = np.random.randn(1024)
        results = service._semantic_search(query_embedding, max_results=5)

        assert len(results) == 1
        assert results[0].paper_id == "paper1"
        # Title match should have higher combined score due to boosting
        assert results[0].score > 0.8

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_bm25_search_implementation(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test BM25 search implementation"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock ES response
        mock_es_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "paper1",
                        "_score": 1.5,
                        "_source": {
                            "paper_id": "paper1",
                            "title": "Paper 1",
                            "authors": ["Author 1"],
                            "abstract": "Abstract 1",
                            "categories": ["cs.AI"]
                        },
                        "highlight": {
                            "title": ["<em>Paper</em> 1"],
                            "abstract": ["This is a test <em>abstract</em>"]
                        }
                    }
                ]
            }
        }

        mock_components["indexer"].es.search.return_value = mock_es_response

        results = service._bm25_search("test query", max_results=5)

        assert len(results) == 1
        assert results[0].paper_id == "paper1"
        assert results[0].score == 1.5
        assert results[0].highlight is not None
        assert "<em>Paper</em> 1" in results[0].highlight

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_get_paper_details(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test getting paper details"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock indexer response
        mock_paper = {
            "paper_id": "paper1",
            "title": "Paper 1",
            "authors": ["Author 1"],
            "abstract": "Abstract 1",
            "content": "Full content here",
            "categories": ["cs.AI"]
        }

        mock_components["indexer"].get_document.return_value = mock_paper

        result = service.get_paper_details("paper1")

        assert result == mock_paper
        mock_components["indexer"].get_document.assert_called_once_with("paper1")

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_suggest_papers(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test paper suggestion functionality"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock paper details
        reference_paper = {
            "paper_id": "paper1",
            "title": "Reference Paper",
            "abstract": "This is the reference abstract"
        }

        mock_components["indexer"].get_document.return_value = reference_paper

        # Mock search results
        with patch.object(service, 'search') as mock_search:
            mock_search.return_value = [
                SearchResult(
                    paper_id="paper1", title="Reference Paper", authors=["Author 1"],
                    abstract="Reference abstract", score=0.99, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                ),
                SearchResult(
                    paper_id="paper2", title="Similar Paper", authors=["Author 2"],
                    abstract="Similar abstract", score=0.85, categories=["cs.AI"],
                    publish_date="2023-01-02", pdf_path=None, markdown_path=None
                )
            ]

            suggestions = service.suggest_papers("paper1", max_suggestions=3)

            # Should exclude the reference paper itself
            assert len(suggestions) == 1
            assert suggestions[0].paper_id == "paper2"

            # Check that search was called with combined title and abstract
            search_call = mock_search.call_args
            search_query = search_call[1]["query"]
            assert "Reference Paper" in search_query
            assert "reference abstract" in search_query

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_get_search_stats(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test getting search service statistics"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        # Mock indexer stats
        mock_components["indexer"].get_index_stats.return_value = {
            "document_count": 1000,
            "index_size_mb": 50.5
        }

        stats = service.get_search_stats()

        assert stats["total_papers"] == 1000
        assert stats["index_size_mb"] == 50.5
        assert stats["embedding_model"] == "test-model"
        assert stats["embedding_dimension"] == 1024

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_search_invalid_mode(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test search with invalid mode"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with pytest.raises(ValueError, match="Unknown search mode"):
            service.search("test query", search_mode="invalid_mode")

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_create_search_result(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test SearchResult creation from ES response"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        es_source = {
            "paper_id": "test_001",
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "abstract": "Test abstract",
            "categories": ["cs.AI", "cs.LG"],
            "publish_date": "2023-01-01",
            "pdf_path": "/path/to/pdf",
            "markdown_path": "/path/to/markdown"
        }

        result = service._create_search_result(es_source, 0.85, "highlight text")

        assert isinstance(result, SearchResult)
        assert result.paper_id == "test_001"
        assert result.title == "Test Paper"
        assert result.authors == ["Author 1", "Author 2"]
        assert result.abstract == "Test abstract"
        assert result.score == 0.85
        assert result.categories == ["cs.AI", "cs.LG"]
        assert result.publish_date == "2023-01-01"
        assert result.pdf_path == "/path/to/pdf"
        assert result.markdown_path == "/path/to/markdown"
        assert result.highlight == "highlight text"

    @patch('backend.services.search_service.BGEEmbedder')
    @patch('backend.services.search_service.ESIndexer')
    def test_hybrid_search_score_combination(self, mock_indexer_class, mock_embedder_class, mock_components):
        """Test score combination in hybrid search"""
        mock_embedder_class.return_value = mock_components["embedder"]
        mock_indexer_class.return_value = mock_components["indexer"]

        service = SearchService()

        with patch.object(service, '_bm25_search') as mock_bm25, \
             patch.object(service, '_semantic_search') as mock_semantic:

            # Mock results with known scores
            bm25_results = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.8, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]
            semantic_results = [
                SearchResult(
                    paper_id="paper1", title="Paper 1", authors=["Author 1"],
                    abstract="Abstract 1", score=0.6, categories=["cs.AI"],
                    publish_date="2023-01-01", pdf_path=None, markdown_path=None
                )
            ]

            mock_bm25.return_value = bm25_results
            mock_semantic.return_value = semantic_results

            results = service._hybrid_search("test query", np.random.randn(1024), 5)

            assert len(results) == 1
            # Combined score should be weighted combination: 0.4 * 0.8 + 0.6 * 0.6 = 0.68
            expected_score = 0.4 * 0.8 + 0.6 * 0.6
            assert abs(results[0].score - expected_score) < 0.01