#!/usr/bin/env python3
"""
Elasticsearch Indexer for paper documents.
Handles index creation, document indexing, and search operations.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import RequestError
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESIndexer:
    def __init__(
        self,
        es_host: str = "localhost:9200",
        index_name: str = "papers",
        embedding_dim: int = 1024
    ):
        """
        Initialize Elasticsearch indexer.

        Args:
            es_host: Elasticsearch host
            index_name: Name of the index
            embedding_dim: Dimension of embeddings (BGE uses 1024)
        """
        self.es_host = es_host
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        # Connect to Elasticsearch
        if not es_host.startswith(('http://', 'https://')):
            es_host = f"http://{es_host}"

        self.es = Elasticsearch(
            [es_host],
            verify_certs=False,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )

        # Check connection
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {es_host}")

        logger.info(f"Connected to Elasticsearch at {es_host}")

    def create_index(self, force: bool = False):
        """
        Create the papers index with appropriate mappings.

        Args:
            force: If True, delete existing index first
        """
        if force and self.es.indices.exists(index=self.index_name):
            logger.warning(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)

        # Index mapping optimized for chunk-based paper search
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "paper_analyzer": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # Paper metadata (repeated in each chunk for efficient filtering)
                    "paper_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "paper_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "authors": {"type": "keyword"},
                    "abstract": {
                        "type": "text",
                        "analyzer": "paper_analyzer"
                    },
                    "categories": {"type": "keyword"},
                    "publish_date": {"type": "date"},

                    # Document type for filtering
                    "doc_type": {"type": "keyword"},

                    # Chunk-specific fields
                    "chunk_index": {"type": "integer"},
                    "chunk_text": {
                        "type": "text",
                        "analyzer": "paper_analyzer"
                    },
                    "chunk_start": {"type": "integer"},
                    "chunk_end": {"type": "integer"},
                    "total_chunks": {"type": "integer"},

                    # Embeddings for semantic search
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "abstract_embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "chunk_embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },

                    # File references
                    "pdf_path": {"type": "keyword"},
                    "markdown_path": {"type": "keyword"},
                    "minio_pdf_url": {"type": "keyword"},
                    "minio_markdown_url": {"type": "keyword"},

                    # Additional metadata from JSON
                    "word_count": {"type": "integer"},
                    "has_images": {"type": "boolean"},
                    "pdf_size": {"type": "integer"},
                    "downloaded_at": {"type": "date"},
                    "indexed_at": {"type": "date"}
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index {self.index_name} already exists")
            else:
                raise

    def index_document(self, document: Dict) -> str:
        """
        Index a single document.

        Args:
            document: Document to index

        Returns:
            Document ID
        """
        # Add timestamp
        document['indexed_at'] = datetime.now().isoformat()

        # Ensure embeddings are lists (not numpy arrays)
        if 'title_embedding' in document and isinstance(document['title_embedding'], np.ndarray):
            document['title_embedding'] = document['title_embedding'].tolist()
        if 'abstract_embedding' in document and isinstance(document['abstract_embedding'], np.ndarray):
            document['abstract_embedding'] = document['abstract_embedding'].tolist()
        if 'chunk_embedding' in document and isinstance(document['chunk_embedding'], np.ndarray):
            document['chunk_embedding'] = document['chunk_embedding'].tolist()

        # Create unique ID for chunk documents
        doc_id = document.get('paper_id')
        if document.get('doc_type') == 'chunk':
            doc_id = f"{doc_id}_chunk_{document.get('chunk_index', 0)}"

        # Index the document
        response = self.es.index(
            index=self.index_name,
            id=doc_id,
            body=document
        )

        return response['_id']

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()

    def bulk_index(self, documents: List[Dict], batch_size: int = 100):
        """
        Bulk index multiple documents.

        Args:
            documents: List of documents to index
            batch_size: Batch size for bulk indexing
        """
        actions = []

        for doc in documents:
            # Add timestamp
            doc['indexed_at'] = datetime.now().isoformat()

            # Ensure embeddings are lists
            if 'title_embedding' in doc and isinstance(doc['title_embedding'], np.ndarray):
                doc['title_embedding'] = doc['title_embedding'].tolist()
            if 'abstract_embedding' in doc and isinstance(doc['abstract_embedding'], np.ndarray):
                doc['abstract_embedding'] = doc['abstract_embedding'].tolist()
            if 'chunk_embedding' in doc and isinstance(doc['chunk_embedding'], np.ndarray):
                doc['chunk_embedding'] = doc['chunk_embedding'].tolist()

            # Create unique ID for chunk documents
            doc_id = doc.get('paper_id')
            if doc.get('doc_type') == 'chunk':
                doc_id = f"{doc_id}_chunk_{doc.get('chunk_index', 0)}"

            action = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": doc
            }
            actions.append(action)

        # Bulk index
        success, failed = helpers.bulk(
            self.es,
            actions,
            chunk_size=batch_size,
            request_timeout=60
        )

        logger.info(f"Indexed {success} documents, {len(failed)} failed")

        if failed:
            logger.error(f"Failed documents: {failed[:5]}")  # Log first 5 failures

    def search(
        self,
        query: str = None,
        query_embedding: np.ndarray = None,
        size: int = 10,
        search_fields: List[str] = None,
        use_semantic: bool = True,
        use_bm25: bool = True
    ) -> List[Dict]:
        """
        Perform optimized hybrid search on chunk documents.
        Aggregates chunks by paper for final results.

        Args:
            query: Text query
            query_embedding: Query embedding for semantic search
            size: Number of papers to return
            search_fields: Fields to search in (default: title, abstract, chunk_text)
            use_semantic: Whether to use semantic search
            use_bm25: Whether to use BM25 text search

        Returns:
            List of search results aggregated by paper
        """
        if search_fields is None:
            search_fields = ["title^3", "abstract^2", "chunk_text"]

        # Build the search query
        should_clauses = []

<<<<<<< Updated upstream
=======
        logger.info(f"Building ES query - BM25: {use_bm25}, Query: '{query}', Fields: {search_fields}")

>>>>>>> Stashed changes
        # BM25 text search on chunks
        if use_bm25 and query:
            should_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": search_fields,
                    "type": "best_fields",
<<<<<<< Updated upstream
                    "boost": 0.4  # 40% weight for BM25
                }
            })
=======
                    "boost": 0.3  # 30% weight for BM25
                }
            })
            logger.debug(f"Added BM25 clause with query: '{query}'")
>>>>>>> Stashed changes

        # Semantic search
        if use_semantic and query_embedding is not None:
            # Convert to list if numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Title embedding search
            should_clauses.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'title_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    },
<<<<<<< Updated upstream
                    "boost": 0.3  # 30% weight for title semantic
=======
                    "boost": 0.4  # 40% weight for title semantic
>>>>>>> Stashed changes
                }
            })

            # Abstract embedding search
            should_clauses.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'abstract_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    },
<<<<<<< Updated upstream
                    "boost": 0.2  # 20% weight for abstract semantic
=======
                    "boost": 0.25  # 25% weight for abstract semantic
>>>>>>> Stashed changes
                }
            })

            # Chunk embedding search
            should_clauses.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'chunk_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    },
<<<<<<< Updated upstream
                    "boost": 0.6  # 60% weight for chunk semantic
=======
                    "boost": 0.35  # 35% weight for chunk semantic
>>>>>>> Stashed changes
                }
            })

        # Build search query with aggregation by paper_id
        search_body = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                    "filter": [
                        {"term": {"doc_type": "chunk"}}
                    ]
                }
            },
            "aggs": {
                "papers": {
                    "terms": {
                        "field": "paper_id",
                        "size": size
                    },
                    "aggs": {
                        "max_score": {"max": {"script": "_score"}},
                        "best_chunk": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"_score": {"order": "desc"}}],
                                "_source": {
                                    "excludes": ["*_embedding"]
                                }
                            }
                        }
                    }
                }
            },
            "size": 0  # Only get aggregations
        }

<<<<<<< Updated upstream
        # Execute search
        response = self.es.search(index=self.index_name, body=search_body)

        # Extract aggregated results
        results = []
        for bucket in response['aggregations']['papers']['buckets']:
            paper_data = bucket['best_chunk']['hits']['hits'][0]['_source']
            paper_data['_score'] = bucket['max_score']['value']
            paper_data['matching_chunks'] = bucket['doc_count']
            results.append(paper_data)

=======
        # Log the full query for debugging
        import json
        logger.debug(f"Full ES search query: {json.dumps(search_body, indent=2)}")

        # Execute search
        response = self.es.search(index=self.index_name, body=search_body)

        logger.info(f"ES response - Total hits: {response.get('hits', {}).get('total', {}).get('value', 0) if isinstance(response.get('hits', {}).get('total', {}), dict) else response.get('hits', {}).get('total', 0)}")
        logger.debug(f"ES response aggregations: {len(response.get('aggregations', {}).get('papers', {}).get('buckets', []))} paper buckets")

        # Extract aggregated results
        results = []
        for i, bucket in enumerate(response['aggregations']['papers']['buckets']):
            if i < 3:
                logger.debug(f"Bucket {i}: paper_id={bucket['key']}, doc_count={bucket['doc_count']}, max_score={bucket['max_score']['value']}")

            # Check if best_chunk has hits
            if 'best_chunk' not in bucket or 'hits' not in bucket['best_chunk'] or 'hits' not in bucket['best_chunk']['hits']:
                logger.warning(f"Bucket for paper {bucket['key']} has no best_chunk hits!")
                continue

            if len(bucket['best_chunk']['hits']['hits']) == 0:
                logger.warning(f"Bucket for paper {bucket['key']} has empty hits array!")
                continue

            paper_data = bucket['best_chunk']['hits']['hits'][0]['_source']
            paper_data['_score'] = bucket['max_score']['value']
            paper_data['matching_chunks'] = bucket['doc_count']

            # Log all fields in the first paper for debugging
            if i == 0:
                logger.info(f"First paper data keys: {list(paper_data.keys())}")
                logger.info(f"First paper - paper_id: {paper_data.get('paper_id')}, title: {paper_data.get('title', 'N/A')[:50]}")
                logger.info(f"First paper - categories: {paper_data.get('categories')}, authors: {paper_data.get('authors')}")

            results.append(paper_data)

            if i < 3:
                logger.debug(f"Extracted paper {i}: id={paper_data.get('paper_id')}, title={paper_data.get('title', 'N/A')[:50]}")

        logger.info(f"Extracted {len(results)} papers from {len(response['aggregations']['papers']['buckets'])} buckets")

>>>>>>> Stashed changes
        # Sort by score descending
        results.sort(key=lambda x: x['_score'], reverse=True)

        return results

    def get_document(self, paper_id: str) -> Optional[Dict]:
        """
        Get a single document by ID.

        Args:
            paper_id: Paper ID

        Returns:
            Document or None if not found
        """
        try:
            response = self.es.get(index=self.index_name, id=paper_id)
            return response['_source']
        except:
            return None

    def delete_document(self, paper_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            paper_id: Paper ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.es.delete(index=self.index_name, id=paper_id)
            return True
        except:
            return False

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index.

        Returns:
            Index statistics
        """
        stats = self.es.indices.stats(index=self.index_name)
        count = self.es.count(index=self.index_name)

        return {
            "document_count": count['count'],
            "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
            "index_size_mb": stats['indices'][self.index_name]['total']['store']['size_in_bytes'] / (1024 * 1024)
        }


def main():
    """Test the ES indexer"""
    # Initialize indexer
    indexer = ESIndexer(es_host="localhost:9200")

    # Create index
    indexer.create_index(force=True)

    # Test document
    test_doc = {
        "paper_id": "test_001",
        "title": "Test Paper on Transformers",
        "authors": ["John Doe", "Jane Smith"],
        "abstract": "This is a test abstract about transformer models in NLP.",
        "content": "Full paper content goes here...",
        "title_embedding": np.random.randn(1024).tolist(),
        "abstract_embedding": np.random.randn(1024).tolist(),
        "categories": ["cs.CL"],
        "publish_date": "2024-01-01"
    }

    # Index document
    doc_id = indexer.index_document(test_doc)
    print(f"Indexed document: {doc_id}")

    # Search
    results = indexer.search(query="transformer models", size=5)
    print(f"Found {len(results)} results")

    # Get stats
    stats = indexer.get_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()