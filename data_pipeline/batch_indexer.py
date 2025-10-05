#!/usr/bin/env python3
"""
Batch indexer for processing and indexing large collections of papers.
Handles XML parsing, chunking, embedding, and Elasticsearch indexing.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from logs import log
from data_pipeline.tei_xml_parser import TEIXMLParser
from data_pipeline.document_chunker import DocumentChunker
from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.es_indexer import ESIndexer


class BatchIndexer:
    """Batch processor for indexing paper collections."""

    def __init__(
        self,
        es_host: str = "localhost:9200",
        index_name: str = "papers",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        batch_size: int = 100,
        checkpoint_interval: int = 1000
    ):
        """
        Initialize batch indexer.

        Args:
            es_host: Elasticsearch host
            index_name: Index name
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for bulk indexing
            checkpoint_interval: Save checkpoint every N papers
        """
        self.parser = TEIXMLParser()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = BGEEmbedder()
        self.indexer = ESIndexer(es_host=es_host, index_name=index_name, embedding_dim=self.embedder.embedding_dim)

        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        log.info(f"Initialized batch indexer: batch_size={batch_size}, checkpoint_interval={checkpoint_interval}")

    def process_paper(self, xml_path: str) -> Optional[Dict]:
        """
        Process a single paper: parse, chunk, and embed.

        Args:
            xml_path: Path to TEI XML file

        Returns:
            Dictionary with processed paper data ready for indexing
        """
        # Parse XML
        paper_data = self.parser.parse_file(xml_path)
        if not paper_data:
            return None

        # Generate embeddings for title and abstract
        title_embedding = self.embedder.encode(paper_data['title']) if paper_data['title'] else None
        abstract_embedding = self.embedder.encode(paper_data['abstract']) if paper_data['abstract'] else None

        # Chunk the content
        content_chunks = self.chunker.chunk_text(
            paper_data['content'],
            respect_sections=True
        ) if paper_data['content'] else []

        # Generate embeddings for chunks
        chunk_embeddings = []
        if content_chunks:
            chunk_texts = [chunk.text for chunk in content_chunks]
            chunk_embeddings = self.embedder.encode(chunk_texts, batch_size=32)

        # Create paper document (doc_type="paper")
        paper_doc = {
            'paper_id': paper_data['paper_id'],
            'title': paper_data['title'],
            'authors': paper_data['authors'],
            'abstract': paper_data['abstract'],
            'categories': paper_data['categories'],
            'doc_type': 'paper',
            'title_embedding': title_embedding,
            'abstract_embedding': abstract_embedding,
            'total_chunks': len(content_chunks),
            'xml_path': paper_data['xml_path']
        }

        # Create chunk documents (doc_type="chunk")
        chunk_docs = []
        for i, (chunk, embedding) in enumerate(zip(content_chunks, chunk_embeddings)):
            chunk_doc = {
                'paper_id': paper_data['paper_id'],
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'abstract': paper_data['abstract'],
                'categories': paper_data['categories'],
                'doc_type': 'chunk',
                'chunk_index': chunk.chunk_index,
                'chunk_text': chunk.text,
                'chunk_start': chunk.start_pos,
                'chunk_end': chunk.end_pos,
                'total_chunks': len(content_chunks),
                'chunk_embedding': embedding,
                'xml_path': paper_data['xml_path']
            }
            chunk_docs.append(chunk_doc)

        return {
            'paper_doc': paper_doc,
            'chunk_docs': chunk_docs
        }

    def index_papers(
        self,
        xml_files: List[str],
        create_index: bool = False,
        force_recreate: bool = False
    ):
        """
        Index a collection of papers from XML files.

        Args:
            xml_files: List of XML file paths
            create_index: Whether to create index if it doesn't exist
            force_recreate: Whether to recreate index from scratch
        """
        # Create index if needed
        if create_index:
            self.indexer.create_index(force=force_recreate)

        total_papers = len(xml_files)
        log.info(f"Starting to process {total_papers} papers")

        # Track statistics
        stats = {
            'processed': 0,
            'failed': 0,
            'paper_docs': 0,
            'chunk_docs': 0,
            'start_time': datetime.now()
        }

        # Batch documents for bulk indexing
        doc_batch = []

        for i, xml_file in enumerate(tqdm(xml_files, desc="Processing papers")):
            try:
                # Process paper
                result = self.process_paper(xml_file)

                if result:
                    # Add to batch
                    doc_batch.append(result['paper_doc'])
                    doc_batch.extend(result['chunk_docs'])

                    stats['processed'] += 1
                    stats['paper_docs'] += 1
                    stats['chunk_docs'] += len(result['chunk_docs'])

                    # Bulk index when batch is full
                    if len(doc_batch) >= self.batch_size:
                        self.indexer.bulk_index(doc_batch, batch_size=self.batch_size)
                        doc_batch = []
                else:
                    stats['failed'] += 1

                # Checkpoint
                if (i + 1) % self.checkpoint_interval == 0:
                    self._log_progress(stats, i + 1, total_papers)

            except Exception as e:
                log.error(f"Error processing {xml_file}: {e}")
                stats['failed'] += 1

        # Index remaining documents
        if doc_batch:
            self.indexer.bulk_index(doc_batch, batch_size=self.batch_size)

        # Final statistics
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

        log.info(f"\n{'='*60}")
        log.info(f"Indexing completed!")
        log.info(f"Total papers: {total_papers}")
        log.info(f"Successfully processed: {stats['processed']}")
        log.info(f"Failed: {stats['failed']}")
        log.info(f"Paper documents: {stats['paper_docs']}")
        log.info(f"Chunk documents: {stats['chunk_docs']}")
        log.info(f"Duration: {stats['duration']:.2f} seconds")
        log.info(f"Average: {stats['duration']/total_papers:.2f} sec/paper")
        log.info(f"{'='*60}\n")

    def _log_progress(self, stats: Dict, current: int, total: int):
        """Log progress checkpoint."""
        elapsed = (datetime.now() - stats['start_time']).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0

        log.info(f"\nCheckpoint at {current}/{total} papers:")
        log.info(f"  Processed: {stats['processed']}, Failed: {stats['failed']}")
        log.info(f"  Rate: {rate:.2f} papers/sec")
        log.info(f"  Paper docs: {stats['paper_docs']}, Chunk docs: {stats['chunk_docs']}")


def main():
    """Run batch indexer on sample files."""
    # Initialize indexer
    indexer = BatchIndexer(
        es_host="localhost:9200",
        index_name="papers",
        chunk_size=512,
        chunk_overlap=100,
        batch_size=100,
        checkpoint_interval=500
    )

    # Get XML files from directory
    xml_dir = Path("/Users/admin/code/cazoodle/paper-search-engine")
    xml_files = list(xml_dir.glob("*.xml"))

    log.info(f"Found {len(xml_files)} XML files")

    if xml_files:
        # Index papers
        indexer.index_papers(
            xml_files,
            create_index=True,
            force_recreate=False
        )
    else:
        log.warning("No XML files found")


if __name__ == "__main__":
    main()
