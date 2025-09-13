"""
Unit tests for Paper Ingestion Pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from data_pipeline.ingest_papers import PaperProcessor


class TestPaperProcessor:
    """Test cases for Paper Processor"""

    @pytest.fixture
    def mock_components(self):
        """Mock all the processor components"""
        with patch('data_pipeline.ingest_papers.BGEEmbedder') as mock_embedder_class, \
             patch('data_pipeline.ingest_papers.DocumentChunker') as mock_chunker_class, \
             patch('data_pipeline.ingest_papers.ESIndexer') as mock_indexer_class:

            # Mock embedder
            mock_embedder = Mock()
            mock_embedder.embedding_dim = 1024
            mock_embedder.encode.return_value = np.random.randn(1024)
            mock_embedder_class.return_value = mock_embedder

            # Mock chunker
            mock_chunker = Mock()
            mock_chunker.chunk_paper.return_value = {
                "paper_id": "test_001",
                "title": "Test Paper",
                "content_chunks": [
                    {"text": "chunk 1", "chunk_index": 0},
                    {"text": "chunk 2", "chunk_index": 1}
                ]
            }
            mock_chunker_class.return_value = mock_chunker

            # Mock indexer
            mock_indexer = Mock()
            mock_indexer.embedding_dim = 1024
            mock_indexer.create_index.return_value = True
            mock_indexer.index_document.return_value = "doc_id"
            mock_indexer.bulk_index.return_value = None
            mock_indexer_class.return_value = mock_indexer

            return {
                "embedder": mock_embedder,
                "chunker": mock_chunker,
                "indexer": mock_indexer
            }

    def test_processor_initialization(self, mock_components):
        """Test processor initialization"""
        processor = PaperProcessor(
            es_host="test:9200",
            bge_model="test-model",
            chunk_size=256,
            chunk_overlap=50
        )

        assert processor.embedder == mock_components["embedder"]
        assert processor.chunker == mock_components["chunker"]
        assert processor.indexer == mock_components["indexer"]

    def test_extract_metadata_from_markdown(self, mock_components):
        """Test metadata extraction from markdown files"""
        processor = PaperProcessor()

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            markdown_content = """
# Attention Is All You Need

Submitted by
Ashish Vaswani, Noam Shazeer, Niki Parmar

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.

## Introduction

Recurrent neural networks have been established as state of the art approaches in sequence modeling.
"""
            f.write(markdown_content)
            f.flush()

            markdown_path = Path(f.name)

        try:
            metadata = processor.extract_metadata_from_markdown(markdown_path)

            # Check extracted metadata
            assert metadata["paper_id"] == markdown_path.stem
            assert "Attention Is All You Need" in metadata["title"]
            assert len(metadata["authors"]) > 0
            assert "Ashish Vaswani" in metadata["authors"]
            assert "transformer" in metadata["abstract"].lower()
            assert len(metadata["content"]) > 0
            assert metadata["word_count"] > 0

        finally:
            markdown_path.unlink()

    def test_extract_metadata_arxiv_paper_id(self, mock_components):
        """Test metadata extraction for ArXiv paper IDs"""
        processor = PaperProcessor()

        # Create temporary file with ArXiv ID format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Paper\n\nContent here.")
            f.flush()

            # Rename to ArXiv format
            arxiv_path = Path(f.name).parent / "1706.03762.md"
            Path(f.name).rename(arxiv_path)

        try:
            metadata = processor.extract_metadata_from_markdown(arxiv_path)

            assert metadata["paper_id"] == "1706.03762"
            assert metadata["publish_date"] == "2017-06-01"  # June 2017

        finally:
            arxiv_path.unlink()

    def test_extract_metadata_file_not_found(self, mock_components):
        """Test metadata extraction with non-existent file"""
        processor = PaperProcessor()

        non_existent_path = Path("non_existent_file.md")
        metadata = processor.extract_metadata_from_markdown(non_existent_path)

        assert metadata == {}

    def test_process_paper(self, mock_components):
        """Test processing a single paper"""
        processor = PaperProcessor()

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("""
# Test Paper

By Test Author

## Abstract

This is a test abstract.

## Content

This is the main content of the paper.
""")
            f.flush()
            markdown_path = Path(f.name)

        try:
            # Mock the chunker to return specific chunks
            mock_components["chunker"].chunk_paper.return_value = {
                "paper_id": markdown_path.stem,
                "title": "Test Paper",
                "abstract": "This is a test abstract.",
                "content_chunks": [
                    {"text": "chunk 1", "chunk_index": 0},
                    {"text": "chunk 2", "chunk_index": 1}
                ]
            }

            result = processor.process_paper(markdown_path)

            assert result is not None
            assert result["paper_id"] == markdown_path.stem
            assert "title_embedding" in result
            assert "abstract_embedding" in result
            assert len(result["content_chunks"]) == 2
            assert all("embedding" in chunk for chunk in result["content_chunks"])
            assert result["chunk_count"] == 2

        finally:
            markdown_path.unlink()

    def test_process_paper_with_pdf(self, mock_components):
        """Test processing paper when corresponding PDF exists"""
        processor = PaperProcessor()

        # Create temporary markdown and PDF files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create structure similar to real data
            markdown_dir = temp_path / "processed" / "markdown"
            pdf_dir = temp_path / "pdfs"
            markdown_dir.mkdir(parents=True)
            pdf_dir.mkdir(parents=True)

            markdown_file = markdown_dir / "test_paper.md"
            pdf_file = pdf_dir / "test_paper.pdf"

            markdown_file.write_text("# Test Paper\n\nContent here.")
            pdf_file.write_text("fake pdf content")

            result = processor.process_paper(markdown_file)

            assert result is not None
            assert "pdf_path" in result
            assert str(pdf_file) == result["pdf_path"]

    def test_process_paper_error_handling(self, mock_components):
        """Test error handling in paper processing"""
        processor = PaperProcessor()

        # Test with file that causes processing error
        mock_components["chunker"].chunk_paper.side_effect = Exception("Processing error")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Paper")
            f.flush()
            markdown_path = Path(f.name)

        try:
            result = processor.process_paper(markdown_path)
            assert result is None

        finally:
            markdown_path.unlink()

    def test_ingest_directory(self, mock_components):
        """Test ingesting a directory of markdown files"""
        processor = PaperProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_dir = Path(temp_dir)

            # Create multiple markdown files
            files = []
            for i in range(5):
                file_path = markdown_dir / f"paper_{i:03d}.md"
                file_path.write_text(f"# Paper {i}\n\nContent for paper {i}")
                files.append(file_path)

            # Mock successful processing
            def mock_process_paper(path):
                return {
                    "paper_id": path.stem,
                    "title": f"Paper {path.stem}",
                    "content_chunks": [],
                    "chunk_count": 0
                }

            with patch.object(processor, 'process_paper', side_effect=mock_process_paper):
                processor.ingest_directory(
                    markdown_dir=markdown_dir,
                    batch_size=2,
                    max_files=3
                )

            # Check that indexer methods were called
            mock_components["indexer"].create_index.assert_called_once()
            mock_components["indexer"].bulk_index.assert_called()

    def test_ingest_directory_with_resume(self, mock_components):
        """Test ingesting directory with resume functionality"""
        processor = PaperProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_dir = Path(temp_dir)

            # Create files with specific names for resume testing
            file_names = ["paper_001.md", "paper_002.md", "paper_003.md", "paper_004.md"]
            for name in file_names:
                (markdown_dir / name).write_text(f"# {name}\n\nContent")

            def mock_process_paper(path):
                return {
                    "paper_id": path.stem,
                    "title": f"Paper {path.stem}",
                    "content_chunks": [],
                    "chunk_count": 0
                }

            with patch.object(processor, 'process_paper', side_effect=mock_process_paper) as mock_process:
                processor.ingest_directory(
                    markdown_dir=markdown_dir,
                    batch_size=2,
                    resume_from="paper_003"
                )

                # Should only process files from paper_003 onwards
                processed_files = [call[0][0].name for call in mock_process.call_args_list]
                assert "paper_001.md" not in processed_files
                assert "paper_002.md" not in processed_files
                assert "paper_003.md" in processed_files
                assert "paper_004.md" in processed_files

    def test_ingest_directory_bulk_indexing_error(self, mock_components):
        """Test handling of bulk indexing errors"""
        processor = PaperProcessor()

        # Mock bulk indexing to fail
        mock_components["indexer"].bulk_index.side_effect = Exception("Bulk index failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_dir = Path(temp_dir)
            file_path = markdown_dir / "test.md"
            file_path.write_text("# Test Paper")

            def mock_process_paper(path):
                return {
                    "paper_id": path.stem,
                    "title": "Test Paper",
                    "content_chunks": [],
                    "chunk_count": 0
                }

            with patch.object(processor, 'process_paper', side_effect=mock_process_paper):
                # Should not raise exception, should handle gracefully
                processor.ingest_directory(
                    markdown_dir=markdown_dir,
                    batch_size=1
                )

            # Should fall back to individual indexing
            mock_components["indexer"].index_document.assert_called()

    def test_ingest_directory_empty_directory(self, mock_components):
        """Test ingesting empty directory"""
        processor = PaperProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            markdown_dir = Path(temp_dir)

            processor.ingest_directory(markdown_dir=markdown_dir)

            # Should create index but not index any documents
            mock_components["indexer"].create_index.assert_called_once()
            mock_components["indexer"].bulk_index.assert_not_called()

    def test_embedding_generation_failure(self, mock_components):
        """Test handling of embedding generation failures"""
        processor = PaperProcessor()

        # Mock embedder to fail
        mock_components["embedder"].encode.side_effect = Exception("Embedding failed")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Paper\n\nContent")
            f.flush()
            markdown_path = Path(f.name)

        try:
            result = processor.process_paper(markdown_path)
            assert result is None

        finally:
            markdown_path.unlink()

    def test_chunk_embedding_generation(self, mock_components):
        """Test embedding generation for content chunks"""
        processor = PaperProcessor()

        # Mock chunker to return chunks
        mock_components["chunker"].chunk_paper.return_value = {
            "paper_id": "test_001",
            "title": "Test Paper",
            "abstract": "Test abstract",
            "content_chunks": [
                {"text": "chunk 1", "chunk_index": 0},
                {"text": "chunk 2", "chunk_index": 1}
            ]
        }

        # Mock embedder to return different embeddings for different inputs
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.randn(1024)
            else:
                return np.random.randn(len(texts), 1024)

        mock_components["embedder"].encode.side_effect = mock_encode

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Paper\n\n## Abstract\n\nTest abstract\n\n## Content\n\nTest content")
            f.flush()
            markdown_path = Path(f.name)

        try:
            result = processor.process_paper(markdown_path)

            assert result is not None
            assert "title_embedding" in result
            assert "abstract_embedding" in result
            assert len(result["content_chunks"]) == 2
            assert all("embedding" in chunk for chunk in result["content_chunks"])

            # Check that embedder was called for title, abstract, and chunks
            assert mock_components["embedder"].encode.call_count >= 3

        finally:
            markdown_path.unlink()

    def test_paper_with_missing_fields(self, mock_components):
        """Test processing paper with missing title or abstract"""
        processor = PaperProcessor()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            # Content without clear title or abstract
            f.write("Just some content without proper structure.")
            f.flush()
            markdown_path = Path(f.name)

        try:
            result = processor.process_paper(markdown_path)

            assert result is not None
            # Should handle missing title/abstract gracefully
            assert "paper_id" in result

        finally:
            markdown_path.unlink()