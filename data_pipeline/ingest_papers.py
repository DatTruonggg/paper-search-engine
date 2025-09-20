#!/usr/bin/env python3
"""
Main ingestion script for processing markdown papers into Elasticsearch.
Combines document chunking, BGE embedding, and ES indexing.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import re
from datetime import datetime
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from logs import log
from data_pipeline.bge_embedder import BGEEmbedder
from data_pipeline.document_chunker import DocumentChunker
from data_pipeline.es_indexer import ESIndexer
from data_pipeline.minio_storage import MinIOStorage


class PaperProcessor:
    def __init__(
        self,
        es_host: str = "localhost:9200",
        bge_model: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        json_metadata_dir: str = "/Users/admin/code/cazoodle/data/pdfs",
        minio_endpoint: str = "localhost:9002",
        enable_minio: bool = True
    ):
        """
        Initialize paper processor with all components.

        Args:
            es_host: Elasticsearch host
            bge_model: BGE model name
            chunk_size: Chunk size for document splitting
            chunk_overlap: Overlap between chunks
            json_metadata_dir: Directory containing JSON metadata files
            minio_endpoint: MinIO server endpoint
            enable_minio: Whether to enable MinIO storage
        """
        log.info("Initializing paper processor...")

        # Store configuration
        self.json_metadata_dir = Path(json_metadata_dir)
        self.enable_minio = enable_minio

        # Initialize components
        self.embedder = BGEEmbedder(model_name=bge_model)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.indexer = ESIndexer(es_host=es_host, embedding_dim=self.embedder.embedding_dim)

        # Initialize MinIO storage if enabled
        self.minio_storage = None
        if enable_minio:
            try:
                self.minio_storage = MinIOStorage(endpoint=minio_endpoint)
                log.info("MinIO storage initialized")
            except Exception as e:
                log.warning(f"Failed to initialize MinIO storage: {e}")
                log.warning("Continuing without MinIO storage")

        log.info("Paper processor initialized successfully")

    def load_json_metadata(self, paper_id: str, json_dir: Path) -> Dict:
        """
        Load metadata from JSON file in the pdfs directory.

        Args:
            paper_id: Paper ID (ArXiv ID)
            json_dir: Directory containing JSON metadata files

        Returns:
            Dictionary with metadata from JSON file
        """
        json_path = json_dir / f"{paper_id}.json"

        if not json_path.exists():
            log.warning(f"JSON metadata not found: {json_path}")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_metadata = json.load(f)

            # Process authors - split if it's a string
            authors = json_metadata.get('authors', [])
            if isinstance(authors, str):
                # Split by common delimiters
                authors = [author.strip() for author in re.split(r',|;|\band\b', authors)]
                authors = [author for author in authors if author and len(author) > 2]

            # Process categories - split if it's a string
            categories = json_metadata.get('categories', [])
            if isinstance(categories, str):
                categories = [cat.strip() for cat in categories.split()]

            # Ensure categories is a list
            if not categories:
                categories = ['cs.AI']  # Default category

            # Extract publish date from ArXiv ID if not in JSON
            publish_date = json_metadata.get('publish_date')
            if not publish_date and re.match(r'\d{4}\.\d{5}', paper_id):
                year = 2000 + int(paper_id[:2]) if int(paper_id[:2]) < 50 else 1900 + int(paper_id[:2])
                month = int(paper_id[2:4])
                publish_date = f"{year}-{month:02d}-01"

            return {
                'paper_id': json_metadata.get('paper_id', paper_id),
                'title': json_metadata.get('title', ''),
                'authors': authors,
                'abstract': json_metadata.get('abstract', ''),
                'categories': categories,
                'publish_date': publish_date,
                'downloaded_at': json_metadata.get('downloaded_at'),
                'pdf_size': json_metadata.get('pdf_size', 0)
            }

        except Exception as e:
            log.error(f"Error reading JSON metadata {json_path}: {e}")
            return {}

    def extract_metadata_from_markdown(self, markdown_path: Path) -> Dict:
        """
        Extract paper metadata from markdown file and enrich with JSON metadata.

        Args:
            markdown_path: Path to markdown file

        Returns:
            Dictionary with extracted metadata
        """
        try:
            content = markdown_path.read_text(encoding='utf-8')
        except Exception as e:
            log.error(f"Error reading {markdown_path}: {e}")
            return {}

        # Extract paper ID from filename (e.g., "2210.14275.md")
        paper_id = markdown_path.stem

        # Try to load rich metadata from JSON file first
        json_metadata = self.load_json_metadata(paper_id, self.json_metadata_dir)

        # Initialize metadata with JSON data if available, otherwise use defaults
        metadata = {
            'paper_id': paper_id,
            'title': json_metadata.get('title', ''),
            'authors': json_metadata.get('authors', []),
            'abstract': json_metadata.get('abstract', ''),
            'content': content,
            'categories': json_metadata.get('categories', []),
            'publish_date': json_metadata.get('publish_date'),
            'markdown_path': str(markdown_path),
            'word_count': len(content.split()),
            'has_images': '<!-- image -->' in content or '![' in content,
            'downloaded_at': json_metadata.get('downloaded_at'),
            'pdf_size': json_metadata.get('pdf_size', 0)
        }

        # If JSON metadata is missing or incomplete, fall back to markdown parsing
        if not metadata['title']:
            # Extract title (first heading or title-like text)
            title_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
            else:
                # Try to find title in first few lines
                lines = content.split('\n')[:10]
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('<!--'):
                        metadata['title'] = line
                        break

        if not metadata['authors']:
            # Extract authors (look for common patterns)
            author_patterns = [
                r'(?:by|author(?:s)?:?)\s*(.+?)(?:\n|$)',
                r'(?:submitted by)\s*(.+?)(?:\n|$)',
                r'^([A-Z][a-z]+ [A-Z][a-z]+(?:,? (?:and )?[A-Z][a-z]+ [A-Z][a-z]+)*)',
            ]

            for pattern in author_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    authors_text = match.group(1).strip()
                    # Split authors
                    authors = [author.strip() for author in re.split(r',|and', authors_text)]
                    metadata['authors'] = [author for author in authors if author and len(author) > 2]
                    break

        if not metadata['abstract']:
            # Extract abstract (look for abstract section)
            abstract_match = re.search(
                r'(?:^|\n)#+\s*abstract\s*\n(.*?)(?=\n#+|\n\n|\Z)',
                content,
                re.IGNORECASE | re.DOTALL
            )
            if abstract_match:
                metadata['abstract'] = abstract_match.group(1).strip()
            else:
                # Use first paragraph after title as abstract
                paragraphs = re.split(r'\n\s*\n', content)
                for para in paragraphs[:5]:
                    para = para.strip()
                    if len(para) > 100 and not para.startswith('#'):
                        metadata['abstract'] = para[:]
                        break

        if not metadata['categories']:
            # Detect categories from content
            categories = []
            if 'nlp' in content.lower() or 'natural language' in content.lower():
                categories.append('cs.CL')
            if 'machine learning' in content.lower() or 'deep learning' in content.lower():
                categories.append('cs.LG')
            if 'computer vision' in content.lower() or 'image' in content.lower():
                categories.append('cs.CV')

            # Default category if none found
            metadata['categories'] = categories if categories else ['cs.AI']

        if not metadata['publish_date']:
            # Extract date from paper ID if it's ArXiv format (YYMM.NNNNN)
            if re.match(r'\d{4}\.\d{5}', paper_id):
                year = 2000 + int(paper_id[:2]) if int(paper_id[:2]) < 50 else 1900 + int(paper_id[:2])
                month = int(paper_id[2:4])
                metadata['publish_date'] = f"{year}-{month:02d}-01"

        return metadata

    def process_paper(self, markdown_path: Path) -> Optional[List[Dict]]:
        """
        Process a single paper: extract metadata, chunk, embed, and prepare documents for indexing.

        Creates two types of documents:
        1. One paper document with full metadata
        2. Multiple chunk documents (one per chunk) with minimal metadata

        Args:
            markdown_path: Path to markdown file

        Returns:
            List of documents ready for indexing (1 paper doc + N chunk docs)
        """
        try:
            # Extract metadata
            paper_data = self.extract_metadata_from_markdown(markdown_path)
            if not paper_data:
                log.error(f"Failed to extract metadata from {markdown_path}")
                return None

            paper_id = paper_data['paper_id']

            # Chunk the document
            chunked_data = self.chunker.chunk_paper(paper_data)

            # Generate embeddings for title and abstract
            title_embedding = None
            abstract_embedding = None

            if paper_data['title']:
                title_embedding = self.embedder.encode(paper_data['title'])

            if paper_data['abstract']:
                abstract_embedding = self.embedder.encode(paper_data['abstract'])

            # Handle PDF/MinIO uploads first
            minio_pdf_url = None
            minio_md_url = None
            pdf_path = Path(str(markdown_path).replace('/processed/markdown/', '/pdfs/').replace('.md', '.pdf'))
            if pdf_path.exists() and self.minio_storage:
                try:
                    minio_pdf_url = self.minio_storage.upload_pdf(paper_id, pdf_path)
                    minio_md_url = self.minio_storage.upload_markdown(paper_id, markdown_path)
                    log.info(f"Uploaded files to MinIO for paper: {paper_id}")
                except Exception as e:
                    log.warning(f"MinIO upload failed for {paper_id}: {e}")

            # Create list to hold all documents
            documents = []

            # 1. Create the main paper document
            paper_doc = {
                'doc_type': 'paper',
                'paper_id': paper_id,
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'abstract': paper_data['abstract'],
                'categories': paper_data['categories'],
                'publish_date': paper_data['publish_date'],
                'word_count': paper_data['word_count'],
                'has_images': paper_data['has_images'],
                'downloaded_at': paper_data['downloaded_at'],
                'pdf_size': paper_data['pdf_size'],
                'markdown_path': paper_data['markdown_path'],
                'total_chunks': len(chunked_data['content_chunks']) if chunked_data['content_chunks'] else 0,

                # Embeddings for semantic search
                'title_embedding': title_embedding,
                'abstract_embedding': abstract_embedding,

                # MinIO URLs if available
                'minio_pdf_url': minio_pdf_url,
                'minio_markdown_url': minio_md_url,

                # Timestamp for tracking
                'indexed_at': datetime.now().isoformat()
            }
            documents.append(paper_doc)

            # 2. Create chunk documents (minimal metadata)
            if chunked_data['content_chunks']:
                chunk_texts = [chunk['text'] for chunk in chunked_data['content_chunks']]
                chunk_embeddings = self.embedder.encode(chunk_texts, show_progress=False)

                for i, chunk in enumerate(chunked_data['content_chunks']):
                    chunk_doc = {
                        'doc_type': 'chunk',
                        'paper_id': paper_id,  # Reference to parent paper

                        # Chunk-specific data
                        'chunk_index': i,
                        'chunk_text': chunk['text'],
                        'chunk_start': chunk['start_pos'],
                        'chunk_end': chunk['end_pos'],
                        'chunk_embedding': chunk_embeddings[i],

                        # Minimal paper metadata for filtering/display
                        'title': paper_data['title'],  # Keep for display purposes
                        'publish_date': paper_data['publish_date'],  # Keep for date filtering
                        'categories': paper_data['categories'],  # Keep for category filtering

                        # Timestamp
                        'indexed_at': datetime.now().isoformat()
                    }
                    documents.append(chunk_doc)

            return documents

        except Exception as e:
            log.error(f"Error processing {markdown_path}: {e}")
            return None

    def ingest_directory(
        self,
        markdown_dir: Path,
        batch_size: int = 10,
        max_files: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """
        Ingest all markdown files from a directory.

        Args:
            markdown_dir: Directory containing markdown files
            batch_size: Batch size for processing
            max_files: Maximum number of files to process
            resume_from: Resume from this paper ID
        """
        # Find all markdown files
        markdown_files = list(markdown_dir.glob("*.md"))
        markdown_files.sort()

        if max_files:
            markdown_files = markdown_files[:max_files]

        log.info(f"Found {len(markdown_files)} markdown files to process")

        # Create index
        self.indexer.create_index(force=False)

        # Resume logic
        start_index = 0
        if resume_from:
            for i, md_file in enumerate(markdown_files):
                if md_file.stem == resume_from:
                    start_index = i
                    break
            log.info(f"Resuming from {resume_from} (index {start_index})")

        # Process files in batches
        processed_count = 0
        batch = []

        for i, md_file in enumerate(tqdm(markdown_files[start_index:], desc="Processing papers")):
            # Process single paper - now returns list of chunk documents
            chunk_documents = self.process_paper(md_file)

            if chunk_documents:
                # Add all chunk documents to batch
                batch.extend(chunk_documents)
                processed_count += 1

                # Index batch when full
                if len(batch) >= batch_size:
                    try:
                        self.indexer.bulk_index(batch)
                        log.info(f"Indexed batch of {len(batch)} chunk documents")
                        batch = []
                    except Exception as e:
                        log.error(f"Error indexing batch: {e}")
                        # Try individual indexing for this batch
                        for doc in batch:
                            try:
                                self.indexer.index_document(doc)
                            except Exception as e2:
                                log.error(f"Error indexing chunk from {doc.get('paper_id', 'unknown')}: {e2}")
                        batch = []
            else:
                log.warning(f"Failed to process {md_file}")

        # Index remaining documents
        if batch:
            try:
                self.indexer.bulk_index(batch)
                log.info(f"Indexed final batch of {len(batch)} documents")
            except Exception as e:
                log.error(f"Error indexing final batch: {e}")

        log.info(f"Ingestion completed. Processed {processed_count} papers")

        # Print index statistics
        stats = self.indexer.get_index_stats()
        log.info(f"Index statistics: {stats}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ingest markdown papers into Elasticsearch")
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default="./data/processed/markdown",
        help="Directory containing markdown files"
    )
    parser.add_argument(
        "--es-host",
        default="localhost:9202",
        help="Elasticsearch host"
    )
    parser.add_argument(
        "--bge-model",
        default="BAAI/bge-large-en-v1.5",
        help="BGE model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--resume-from",
        help="Resume from this paper ID"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in tokens"
    )
    parser.add_argument(
        "--json-metadata-dir",
        type=Path,
        default="/Users/admin/code/cazoodle/data/pdfs",
        help="Directory containing JSON metadata files"
    )
    parser.add_argument(
        "--minio-endpoint",
        default="localhost:9002",
        help="MinIO server endpoint"
    )
    parser.add_argument(
        "--disable-minio",
        action="store_true",
        help="Disable MinIO storage"
    )

    args = parser.parse_args()

    # Check if markdown directory exists
    if not args.markdown_dir.exists():
        log.error(f"Markdown directory does not exist: {args.markdown_dir}")
        sys.exit(1)

    # Initialize processor
    processor = PaperProcessor(
        es_host=args.es_host,
        bge_model=args.bge_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        json_metadata_dir=str(args.json_metadata_dir),
        minio_endpoint=args.minio_endpoint,
        enable_minio=not args.disable_minio
    )

    # Start ingestion
    processor.ingest_directory(
        markdown_dir=args.markdown_dir,
        batch_size=args.batch_size,
        max_files=args.max_files,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()