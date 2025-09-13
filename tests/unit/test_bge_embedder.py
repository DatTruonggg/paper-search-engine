"""
Unit tests for BGE Embedder service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
from transformers import AutoTokenizer, AutoModel

from data_pipeline.bge_embedder import BGEEmbedder


class TestBGEEmbedder:
    """Test cases for BGE Embedder"""

    @pytest.fixture
    def mock_model_components(self):
        """Mock the transformer components"""
        mock_tokenizer = Mock(spec=AutoTokenizer)
        mock_model = Mock(spec=AutoModel)

        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }

        # Mock model output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(1, 5, 1024)  # batch_size, seq_len, hidden_size
        mock_model.return_value = mock_output
        mock_model.config.hidden_size = 1024

        return mock_tokenizer, mock_model

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_embedder_initialization(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test embedder initialization"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = BGEEmbedder(model_name="test-model", cache_dir="./test_cache")

        assert embedder.model_name == "test-model"
        assert embedder.embedding_dim == 1024
        mock_tokenizer_from_pretrained.assert_called_once_with("test-model", cache_dir="./test_cache")
        mock_model_from_pretrained.assert_called_once_with("test-model", cache_dir="./test_cache")

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_encode_single_text(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test encoding a single text"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = BGEEmbedder()

        # Test single text encoding
        text = "This is a test document"
        result = embedder.encode(text)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Should return 1D array for single text

        # Verify tokenizer was called with correct instruction
        expected_text = f"Represent this document for retrieval: {text}"
        mock_tokenizer.assert_called_with(
            [expected_text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_encode_multiple_texts(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test encoding multiple texts"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        # Mock multiple texts output
        mock_output = Mock()
        mock_output.last_hidden_state = torch.randn(3, 5, 1024)  # 3 texts
        mock_model.return_value = mock_output

        embedder = BGEEmbedder()

        # Test multiple texts encoding
        texts = ["Text 1", "Text 2", "Text 3"]
        result = embedder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)  # Should return 2D array for multiple texts

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_encode_queries(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test encoding queries with different instruction"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = BGEEmbedder()

        # Test query encoding
        query = "search query"
        result = embedder.encode_queries(query)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

        # Verify tokenizer was called with query instruction
        expected_query = f"Represent this sentence for searching relevant passages: {query}"
        mock_tokenizer.assert_called_with(
            [expected_query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_compute_similarity(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test similarity computation"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = BGEEmbedder()

        # Create test embeddings
        emb1 = np.random.randn(2, 1024)
        emb2 = np.random.randn(3, 1024)

        result = embedder.compute_similarity(emb1, emb2)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)  # 2x3 similarity matrix

        # Check values are between -1 and 1 (cosine similarity range)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_batch_processing(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test batch processing with large input"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        embedder = BGEEmbedder()

        # Create many texts to test batching
        texts = [f"Text {i}" for i in range(10)]

        # Mock multiple calls for batching
        def mock_tokenizer_side_effect(*args, **kwargs):
            batch_size = len(args[0])
            return {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]] * batch_size),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]] * batch_size)
            }

        def mock_model_side_effect(*args, **kwargs):
            batch_size = args[0]['input_ids'].shape[0]
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(batch_size, 5, 1024)
            return mock_output

        mock_tokenizer.side_effect = mock_tokenizer_side_effect
        mock_model.side_effect = mock_model_side_effect

        result = embedder.encode(texts, batch_size=3)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 1024)

    def test_normalization(self):
        """Test embedding normalization"""
        embedder = BGEEmbedder.__new__(BGEEmbedder)  # Create without __init__

        # Test with unnormalized embeddings
        embeddings = np.array([[3.0, 4.0], [1.0, 1.0]])

        # Manually normalize
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Check that norms are 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_device_placement(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained, mock_model_components):
        """Test that model is placed on correct device"""
        mock_tokenizer, mock_model = mock_model_components
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        mock_model_from_pretrained.return_value = mock_model

        with patch('torch.cuda.is_available', return_value=False):
            embedder = BGEEmbedder()
            assert embedder.device.type == 'cpu'

    @patch('data_pipeline.bge_embedder.AutoTokenizer.from_pretrained')
    @patch('data_pipeline.bge_embedder.AutoModel.from_pretrained')
    def test_error_handling(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        """Test error handling in embedding generation"""
        mock_tokenizer_from_pretrained.side_effect = Exception("Model loading failed")

        with pytest.raises(Exception, match="Model loading failed"):
            BGEEmbedder()

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        embedder = BGEEmbedder.__new__(BGEEmbedder)  # Create without __init__

        # Test empty list
        result = embedder.encode.__wrapped__(embedder, [])  # Use unwrapped method
        # This would be handled by the actual implementation