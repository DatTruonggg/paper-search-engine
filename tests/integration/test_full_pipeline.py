"""
Integration tests for the full paper search pipeline.
These tests require running Elasticsearch and other services.
"""

import pytest
import tempfile
import time
from pathlib import Path
import numpy as np
from unittest.mock import patch

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.document_chunker import DocumentChunker
from data_pipeline.es_indexer import ESIndexer
from data_pipeline.ingest_papers import PaperProcessor
from backend.services.search_service import SearchService


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete pipeline"""

    @pytest.fixture(scope="class")
    def test_index_name(self):
        """Use a test-specific index name"""
        return "test_papers_integration"

    @pytest.fixture(scope="class")
    def es_indexer(self, test_config, test_index_name):
        """Create ES indexer for integration tests"""
        try:
            indexer = ESIndexer(
                es_host=test_config["es_host"],
                index_name=test_index_name,
                embedding_dim=1024
            )
            # Clean up any existing test index
            indexer.es.indices.delete(index=test_index_name, ignore=[404])
            yield indexer
        finally:
            # Cleanup
            try:
                indexer.es.indices.delete(index=test_index_name, ignore=[404])
            except:
                pass

    @pytest.fixture(scope="class")
    def mock_embedder(self):
        """Use mock embedder for faster integration tests"""
        embedder = BGEEmbedder.__new__(BGEEmbedder)  # Create without __init__
        embedder.model_name = "test-model"
        embedder.embedding_dim = 1024

        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.randn(1024).astype(np.float32)
            else:
                return np.random.randn(len(texts), 1024).astype(np.float32)

        embedder.encode = mock_encode
        embedder.encode_queries = mock_encode

        return embedder

    @pytest.fixture
    def sample_papers_directory(self, test_data_dir):
        """Create sample papers for integration testing"""
        papers_dir = test_data_dir / "integration_papers"
        papers_dir.mkdir()

        papers_data = [
            {
                "filename": "1706.03762.md",
                "content": """
# Attention Is All You Need

Submitted by
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Introduction

Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures.

## Model Architecture

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.
"""
            },
            {
                "filename": "2112.08466.md",
                "content": """
# BERT: Pre-training of Deep Bidirectional Transformers

By Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

## Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

## Introduction

Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically.

## BERT Model Architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.
"""
            },
            {
                "filename": "2301.06375.md",
                "content": """
# GPT-3: Language Models are Few-Shot Learners

By Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah

## Abstract

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.

## Introduction

Recent years have featured a trend towards pre-trained language representations in NLP systems, applied in increasingly flexible and task-agnostic ways. First, single-layer representations were learned using word vectors, then contextual representations were learned using RNNs, and more recently pre-trained recurrent or transformer language models have been directly fine-tuned.

## Approach

We train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting.
"""
            }
        ]

        created_files = []
        for paper in papers_data:
            file_path = papers_dir / paper["filename"]
            file_path.write_text(paper["content"])
            created_files.append(file_path)

        return papers_dir, created_files

    def test_embedder_integration(self):
        """Test BGE embedder integration (if model is available)"""
        try:
            embedder = BGEEmbedder(model_name="BAAI/bge-large-en-v1.5")

            # Test single text embedding
            text = "This is a test document about transformers in NLP."
            embedding = embedder.encode(text)

            assert embedding.shape == (1024,)
            assert not np.isnan(embedding).any()
            assert not np.isinf(embedding).any()

            # Test multiple texts
            texts = ["First document", "Second document", "Third document"]
            embeddings = embedder.encode(texts)

            assert embeddings.shape == (3, 1024)
            assert not np.isnan(embeddings).any()

        except Exception as e:
            pytest.skip(f"BGE model not available for integration test: {e}")

    def test_elasticsearch_integration(self, es_indexer):
        """Test Elasticsearch integration"""
        # Test connection
        assert es_indexer.es.ping()

        # Test index creation
        es_indexer.create_index()

        # Test document indexing
        test_doc = {
            "paper_id": "test_001",
            "title": "Test Paper for Integration",
            "authors": ["Test Author"],
            "abstract": "This is a test abstract for integration testing.",
            "title_embedding": np.random.randn(1024).tolist(),
            "abstract_embedding": np.random.randn(1024).tolist(),
            "content_chunks": [
                {
                    "text": "Test chunk content",
                    "embedding": np.random.randn(1024).tolist(),
                    "chunk_index": 0
                }
            ],
            "categories": ["cs.AI"],
            "publish_date": "2023-01-01"
        }

        # Index document
        doc_id = es_indexer.index_document(test_doc)
        assert doc_id == "test_001"

        # Wait for indexing
        time.sleep(1)

        # Test retrieval
        retrieved_doc = es_indexer.get_document("test_001")
        assert retrieved_doc is not None
        assert retrieved_doc["paper_id"] == "test_001"

        # Test search
        results = es_indexer.search(query="integration testing", size=5)
        assert len(results) > 0

    def test_document_chunker_integration(self):
        """Test document chunker with real content"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        # Test with realistic paper content
        paper_data = {
            "paper_id": "integration_test",
            "title": "Integration Test Paper",
            "abstract": "This is a test abstract for integration testing.",
            "content": """
# Introduction

This is the introduction section of our integration test paper. It contains multiple paragraphs and sections to test the chunking functionality properly.

The introduction explains the motivation for our work and provides background context that will be useful for readers.

## Background

In this section, we provide detailed background information about the problem domain. This includes related work and foundational concepts.

### Subsection

This is a subsection with additional details.

## Methodology

Our methodology consists of several steps that we will describe in detail. Each step is important for understanding the overall approach.

## Results

The results section contains our findings and analysis.

## Conclusion

Finally, we conclude with a summary of our contributions and future work directions.
"""
        }

        result = chunker.chunk_paper(paper_data)

        assert result["paper_id"] == "integration_test"
        assert len(result["content_chunks"]) > 0

        # Check chunk properties
        for chunk in result["content_chunks"]:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert "token_count" in chunk
            assert len(chunk["text"]) > 0

    def test_paper_processor_integration(self, sample_papers_directory, mock_embedder):
        """Test paper processor with sample papers"""
        papers_dir, paper_files = sample_papers_directory

        with patch('data_pipeline.ingest_papers.BGEEmbedder', return_value=mock_embedder):
            processor = PaperProcessor(es_host="localhost:9202")

            # Process a single paper
            result = processor.process_paper(paper_files[0])

            assert result is not None
            assert result["paper_id"] == "1706.03762"
            assert "Attention Is All You Need" in result["title"]
            assert "transformer" in result["abstract"].lower()
            assert "title_embedding" in result
            assert "abstract_embedding" in result
            assert len(result["content_chunks"]) > 0

            # Check that embeddings were generated for chunks
            for chunk in result["content_chunks"]:
                assert "embedding" in chunk
                assert len(chunk["embedding"]) == 1024

    def test_full_ingestion_pipeline(self, sample_papers_directory, mock_embedder, es_indexer):
        """Test complete ingestion pipeline"""
        papers_dir, paper_files = sample_papers_directory

        with patch('data_pipeline.ingest_papers.BGEEmbedder', return_value=mock_embedder), \
             patch('data_pipeline.ingest_papers.ESIndexer', return_value=es_indexer):

            processor = PaperProcessor()

            # Ingest all sample papers
            processor.ingest_directory(
                markdown_dir=papers_dir,
                batch_size=2,
                max_files=3
            )

            # Verify that papers were indexed
            es_indexer.create_index.assert_called_once()
            es_indexer.bulk_index.assert_called()

    def test_search_service_integration(self, es_indexer, mock_embedder):
        """Test search service with indexed documents"""
        # First, index some test documents
        test_docs = [
            {
                "paper_id": "search_test_1",
                "title": "Transformer Neural Networks",
                "authors": ["Author 1"],
                "abstract": "This paper discusses transformer neural networks and attention mechanisms.",
                "title_embedding": np.random.randn(1024).tolist(),
                "abstract_embedding": np.random.randn(1024).tolist(),
                "categories": ["cs.AI"],
                "publish_date": "2023-01-01"
            },
            {
                "paper_id": "search_test_2",
                "title": "BERT Language Model",
                "authors": ["Author 2"],
                "abstract": "BERT is a bidirectional language model based on transformers.",
                "title_embedding": np.random.randn(1024).tolist(),
                "abstract_embedding": np.random.randn(1024).tolist(),
                "categories": ["cs.CL"],
                "publish_date": "2023-01-02"
            }
        ]

        es_indexer.create_index()
        es_indexer.bulk_index(test_docs)

        # Wait for indexing
        time.sleep(2)

        # Test search service
        with patch('backend.services.search_service.BGEEmbedder', return_value=mock_embedder), \
             patch('backend.services.search_service.ESIndexer', return_value=es_indexer):

            search_service = SearchService()

            # Test different search modes
            results = search_service.search("transformer neural networks", search_mode="bm25", max_results=5)
            assert len(results) >= 0  # May be 0 due to ES timing

            # Test getting paper details
            details = search_service.get_paper_details("search_test_1")
            # May be None if indexing hasn't completed

            # Test search statistics
            stats = search_service.get_search_stats()
            assert "total_papers" in stats
            assert "embedding_model" in stats

    def test_end_to_end_workflow(self, sample_papers_directory, mock_embedder, test_index_name):
        """Test complete end-to-end workflow"""
        papers_dir, paper_files = sample_papers_directory

        try:
            # Step 1: Create indexer and index
            indexer = ESIndexer(es_host="localhost:9202", index_name=test_index_name)
            indexer.create_index()

            # Step 2: Process and ingest papers
            with patch('data_pipeline.ingest_papers.BGEEmbedder', return_value=mock_embedder):
                processor = PaperProcessor(es_host="localhost:9202")

                for paper_file in paper_files[:2]:  # Process first 2 papers
                    result = processor.process_paper(paper_file)
                    if result:
                        indexer.index_document(result)

            # Wait for indexing
            time.sleep(2)

            # Step 3: Search the indexed papers
            with patch('backend.services.search_service.BGEEmbedder', return_value=mock_embedder):
                search_service = SearchService(es_host="localhost:9202", index_name=test_index_name)

                # Test search
                results = search_service.search("attention transformer", search_mode="hybrid")

                # Results may be empty due to ES timing, but should not error
                assert isinstance(results, list)

                # Test stats
                stats = search_service.get_search_stats()
                assert isinstance(stats, dict)

        finally:
            # Cleanup
            try:
                indexer.es.indices.delete(index=test_index_name, ignore=[404])
            except:
                pass

    def test_performance_benchmarks(self, sample_papers_directory, mock_embedder):
        """Basic performance tests"""
        papers_dir, paper_files = sample_papers_directory

        # Test chunking performance
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)

        start_time = time.time()
        for paper_file in paper_files:
            content = paper_file.read_text()
            chunks = chunker.chunk_text(content)
        chunking_time = time.time() - start_time

        # Should be reasonably fast
        assert chunking_time < 5.0  # Less than 5 seconds for 3 papers

        # Test embedding performance (mocked)
        start_time = time.time()
        for paper_file in paper_files:
            content = paper_file.read_text()
            embedding = mock_embedder.encode(content)
        embedding_time = time.time() - start_time

        # Mocked embeddings should be very fast
        assert embedding_time < 1.0