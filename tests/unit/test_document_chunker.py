"""
Unit tests for Document Chunker service.
"""

import pytest
from unittest.mock import Mock, patch

from data_pipeline.document_chunker import DocumentChunker, Chunk


class TestDocumentChunker:
    """Test cases for Document Chunker"""

    @pytest.fixture
    def chunker(self):
        """Create chunker instance for testing"""
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    @pytest.fixture
    def small_chunker(self):
        """Create small chunker for testing edge cases"""
        return DocumentChunker(chunk_size=10, chunk_overlap=5)

    def test_chunker_initialization(self):
        """Test chunker initialization with different parameters"""
        chunker = DocumentChunker(chunk_size=256, chunk_overlap=50)

        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 50
        assert chunker.tokenizer is not None

    @patch('data_pipeline.document_chunker.tiktoken.get_encoding')
    def test_count_tokens(self, mock_get_encoding, chunker):
        """Test token counting"""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_get_encoding.return_value = mock_encoder

        chunker.tokenizer = mock_encoder

        text = "This is a test"
        count = chunker.count_tokens(text)

        assert count == 5
        mock_encoder.encode.assert_called_once_with(text)

    def test_split_into_sentences(self, chunker):
        """Test sentence splitting"""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."

        sentences = chunker.split_into_sentences(text)

        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"
        assert sentences[3] == "Fourth sentence."

    def test_split_into_sentences_edge_cases(self, chunker):
        """Test sentence splitting edge cases"""
        # Empty text
        assert chunker.split_into_sentences("") == []

        # Single sentence without punctuation
        sentences = chunker.split_into_sentences("This is one sentence")
        assert len(sentences) == 1
        assert sentences[0] == "This is one sentence"

        # Text with abbreviations (should not split)
        text = "Dr. Smith went to the U.S. yesterday."
        sentences = chunker.split_into_sentences(text)
        assert len(sentences) == 1

    def test_split_markdown_sections(self, chunker):
        """Test markdown section splitting"""
        markdown_text = """
# Introduction
This is the introduction section.
Some more intro text.

## Background
This is the background section.

### Subsection
This is a subsection.

## Methodology
This is the methodology section.
"""

        sections = chunker.split_markdown_sections(markdown_text)

        assert len(sections) == 4  # Introduction, Background, Subsection, Methodology

        # Check section titles
        section_titles = [title for title, _ in sections]
        assert "Introduction" in section_titles
        assert "Background" in section_titles
        assert "Subsection" in section_titles
        assert "Methodology" in section_titles

    def test_split_markdown_sections_no_headers(self, chunker):
        """Test markdown splitting with no headers"""
        text = "Just plain text without any headers."

        sections = chunker.split_markdown_sections(text)

        assert len(sections) == 1
        assert sections[0][0] == ""  # Empty title
        assert sections[0][1] == text

    @patch.object(DocumentChunker, 'count_tokens')
    def test_chunk_text_basic(self, mock_count_tokens, chunker):
        """Test basic text chunking"""
        # Mock token counting to return predictable values
        def mock_token_count(text):
            return len(text.split())  # Simple word count

        mock_count_tokens.side_effect = mock_token_count

        text = "This is a test document. " * 20  # Long text

        chunks = chunker.chunk_text(text, respect_sections=False)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)

    @patch.object(DocumentChunker, 'count_tokens')
    def test_chunk_text_with_sections(self, mock_count_tokens, chunker):
        """Test text chunking that respects markdown sections"""
        def mock_token_count(text):
            return len(text.split())

        mock_count_tokens.side_effect = mock_token_count

        markdown_text = """
# Introduction
This is a long introduction section with many words. """ + "Word " * 50 + """

## Methodology
This is the methodology section. """ + "Method " * 40

        chunks = chunker.chunk_text(markdown_text, respect_sections=True)

        assert len(chunks) > 0
        # Check that chunks contain section headers
        chunk_texts = [chunk.text for chunk in chunks]
        intro_chunks = [text for text in chunk_texts if "Introduction" in text]
        method_chunks = [text for text in chunk_texts if "Methodology" in text]

        assert len(intro_chunks) > 0
        assert len(method_chunks) > 0

    @patch.object(DocumentChunker, 'count_tokens')
    def test_chunk_overlap(self, mock_count_tokens, small_chunker):
        """Test that chunks have proper overlap"""
        def mock_token_count(text):
            return len(text.split())

        mock_count_tokens.side_effect = mock_token_count

        # Create text that will definitely need chunking
        text = "Word " * 30  # 30 words, chunk_size=10, overlap=5

        chunks = small_chunker.chunk_text(text, respect_sections=False)

        assert len(chunks) > 1  # Should create multiple chunks

        # Check that there's overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # There should be some overlap in content
            current_words = current_chunk.text.split()
            next_words = next_chunk.text.split()

            # Check if there are common words (indicating overlap)
            common_words = set(current_words) & set(next_words)
            assert len(common_words) > 0  # Should have some overlap

    def test_chunk_paper(self, chunker, sample_paper_content):
        """Test chunking a complete paper"""
        paper_data = {
            "paper_id": "test_001",
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "content": sample_paper_content,
            "authors": ["Test Author"]
        }

        with patch.object(chunker, 'count_tokens', side_effect=lambda x: len(x.split())):
            result = chunker.chunk_paper(paper_data)

        # Check that original data is preserved
        assert result["paper_id"] == "test_001"
        assert result["title"] == "Test Paper"
        assert result["abstract"] == "This is a test abstract"

        # Check that chunks were created
        assert "content_chunks" in result
        assert len(result["content_chunks"]) > 0

        # Check chunk structure
        for chunk in result["content_chunks"]:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "token_count" in chunk
            assert "start_pos" in chunk
            assert "end_pos" in chunk

    def test_chunk_paper_empty_content(self, chunker):
        """Test chunking paper with empty content"""
        paper_data = {
            "paper_id": "test_002",
            "title": "Empty Paper",
            "content": "",
            "authors": []
        }

        result = chunker.chunk_paper(paper_data)

        assert result["paper_id"] == "test_002"
        assert result["content_chunks"] == []

    def test_chunk_paper_no_content(self, chunker):
        """Test chunking paper without content field"""
        paper_data = {
            "paper_id": "test_003",
            "title": "No Content Paper",
            "authors": []
        }

        result = chunker.chunk_paper(paper_data)

        assert result["paper_id"] == "test_003"
        assert result["content_chunks"] == []

    @patch.object(DocumentChunker, 'count_tokens')
    def test_long_sentence_splitting(self, mock_count_tokens, small_chunker):
        """Test handling of sentences longer than chunk size"""
        def mock_token_count(text):
            return len(text.split())

        mock_count_tokens.side_effect = mock_token_count

        # Create a very long sentence (longer than chunk size)
        long_sentence = "This is a very long sentence " * 20  # Much longer than chunk_size=10

        chunks = small_chunker.chunk_text(long_sentence, respect_sections=False)

        assert len(chunks) > 0
        # Long sentence should be split even without proper sentence boundaries

    def test_chunk_indices(self, chunker):
        """Test that chunk indices are assigned correctly"""
        text = "Sentence one. Sentence two. Sentence three. " * 10

        with patch.object(chunker, 'count_tokens', side_effect=lambda x: len(x.split())):
            chunks = chunker.chunk_text(text, respect_sections=False)

        # Check that chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_positions(self, chunker):
        """Test that chunk positions are calculated correctly"""
        text = "First part. Second part. Third part."

        with patch.object(chunker, 'count_tokens', side_effect=lambda x: len(x.split())):
            chunks = chunker.chunk_text(text, respect_sections=False)

        # Check that positions make sense
        for chunk in chunks:
            assert chunk.start_pos >= 0
            assert chunk.end_pos > chunk.start_pos
            assert chunk.end_pos <= len(text)

    def test_chunker_with_different_tokenizers(self):
        """Test chunker with different tokenizer models"""
        chunker1 = DocumentChunker(tokenizer_model="cl100k_base")
        chunker2 = DocumentChunker(tokenizer_model="p50k_base")

        # Both should initialize successfully
        assert chunker1.tokenizer is not None
        assert chunker2.tokenizer is not None

    def test_error_handling(self, chunker):
        """Test error handling for invalid inputs"""
        # Test with None input
        chunks = chunker.chunk_text(None, respect_sections=False)
        assert chunks == []

        # Test with very small chunk size
        tiny_chunker = DocumentChunker(chunk_size=1, chunk_overlap=0)
        text = "Test"

        with patch.object(tiny_chunker, 'count_tokens', side_effect=lambda x: len(x)):
            chunks = tiny_chunker.chunk_text(text, respect_sections=False)

        # Should handle gracefully
        assert isinstance(chunks, list)