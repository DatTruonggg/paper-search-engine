#!/usr/bin/env python3
"""
BGE Embedder Service
Uses BAAI/bge-large-en-v1.5 model for generating embeddings
"""

import torch
import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from pathlib import Path
from logs import log

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", cache_dir: str = "./models"):
        """
        Initialize BGE embedder with the specified model.

        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Loading BGE model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        log.info(f"Model loaded successfully on {self.device}")

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        log.info(f"Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings for cosine similarity
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        # Add instruction for better performance (BGE recommendation)
        texts = [f"Represent this document for retrieval: {text}" for text in texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Use [CLS] token embedding (first token)
                embeddings = model_output.last_hidden_state[:, 0]

            # Normalize if requested
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            if show_progress and i + batch_size < len(texts):
                log.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        # Return single embedding if input was single text
        if single_text:
            return embeddings[0]

        return embeddings

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode queries for search. Uses different instruction prefix.

        Args:
            queries: Single query or list of queries
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(queries, str):
            queries = [queries]
            single_query = True
        else:
            single_query = False

        # Add query instruction (BGE recommendation for better performance)
        queries = [f"Represent this sentence for searching relevant passages: {q}" for q in queries]

        # Use the base encode method
        embeddings = self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            show_progress=False
        )

        if single_query:
            return embeddings[0]

        return embeddings

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.

        Args:
            embeddings1: First set of embeddings (n1, dim)
            embeddings2: Second set of embeddings (n2, dim)

        Returns:
            Similarity matrix (n1, n2)
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Normalize if not already normalized
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Compute cosine similarity
        similarity = np.matmul(embeddings1, embeddings2.T)

        return similarity


def main():
    """Test the BGE embedder"""
    # Initialize embedder
    embedder = BGEEmbedder()

    # Test documents
    documents = [
        "Transformer models have revolutionized natural language processing.",
        "BERT is a bidirectional transformer pre-trained on large text corpora.",
        "The attention mechanism allows models to focus on relevant parts of the input."
    ]

    # Test query
    query = "What are transformer models in NLP?"

    # Encode documents
    doc_embeddings = embedder.encode(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")

    # Encode query
    query_embedding = embedder.encode_queries(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Compute similarities
    similarities = embedder.compute_similarity(query_embedding, doc_embeddings)
    print(f"\nSimilarities between query and documents:")
    for i, (doc, sim) in enumerate(zip(documents, similarities[0])):
        print(f"{i+1}. (Score: {sim:.4f}) {doc[:50]}...")


if __name__ == "__main__":
    main()