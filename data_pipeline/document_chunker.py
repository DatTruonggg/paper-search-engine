#!/usr/bin/env python3
"""
Document Chunker for optimal search performance.
Splits documents into overlapping chunks for better retrieval.
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import tiktoken
from logs import log

@dataclass
class Chunk:
    """Represents a document chunk"""
    text: str
    start_pos: int
    end_pos: int
    chunk_index: int
    token_count: int
    metadata: Dict = None


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer_model: Tiktoken model for counting tokens
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_model)

        log.info(f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        Handles common abbreviations and edge cases.
        """
        # Improved sentence splitting regex
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)

        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def split_markdown_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split markdown text by sections (headers).
        Returns list of (section_title, section_content) tuples.
        """
        sections = []
        current_section = ""
        current_content = []

        lines = text.split('\n')

        for line in lines:
            # Check if line is a header
            if line.startswith('#'):
                # Save previous section if exists
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                    current_content = []

                current_section = line.strip('#').strip()
            else:
                current_content.append(line)

        # Add last section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        # If no sections found, treat entire text as one section
        if not sections:
            sections = [("", text)]

        return sections

    def chunk_text(
        self,
        text: str,
        respect_sections: bool = True
    ) -> List[Chunk]:
        """
        Chunk text into overlapping segments.

        Args:
            text: Text to chunk
            respect_sections: Whether to respect markdown sections

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_index = 0

        if respect_sections:
            # Split by sections first
            sections = self.split_markdown_sections(text)

            for section_title, section_content in sections:
                # Add section title to each chunk for context
                if section_title:
                    section_prefix = f"## {section_title}\n\n"
                else:
                    section_prefix = ""

                # Chunk the section content
                section_chunks = self._chunk_section(
                    section_prefix + section_content,
                    chunk_index
                )

                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        else:
            # Chunk entire text without respecting sections
            chunks = self._chunk_section(text, 0)

        return chunks

    def _chunk_section(self, text: str, start_index: int) -> List[Chunk]:
        """
        Chunk a section of text using sentence boundaries.
        """
        chunks = []
        sentences = self.split_into_sentences(text)

        if not sentences:
            return chunks

        current_chunk = []
        current_tokens = 0
        chunk_start_pos = 0
        chunk_index = start_index

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=chunk_start_pos,
                        end_pos=chunk_start_pos + len(chunk_text),
                        chunk_index=chunk_index,
                        token_count=current_tokens
                    ))
                    chunk_index += 1

                # Split long sentence
                chunk_text = sentence[:self.chunk_size * 4]  # Approximate char count
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=chunk_start_pos + len(' '.join(current_chunk)),
                    end_pos=chunk_start_pos + len(' '.join(current_chunk)) + len(chunk_text),
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(chunk_text)
                ))
                chunk_index += 1

                # Reset for next chunk
                current_chunk = []
                current_tokens = 0
                chunk_start_pos += len(chunk_text) + 1

            # If adding sentence exceeds chunk size, create new chunk
            elif current_tokens + sentence_tokens > self.chunk_size:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_pos=chunk_start_pos,
                    end_pos=chunk_start_pos + len(chunk_text),
                    chunk_index=chunk_index,
                    token_count=current_tokens
                ))
                chunk_index += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    # Calculate sentences to keep for overlap
                    overlap_tokens = 0
                    overlap_sentences = []

                    for j in range(len(current_chunk) - 1, -1, -1):
                        sent_tokens = self.count_tokens(current_chunk[j])
                        if overlap_tokens + sent_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, current_chunk[j])
                            overlap_tokens += sent_tokens
                        else:
                            break

                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = overlap_tokens + sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens

                chunk_start_pos += len(chunk_text) + 1
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=chunk_start_pos,
                end_pos=chunk_start_pos + len(chunk_text),
                chunk_index=chunk_index,
                token_count=current_tokens
            ))

        return chunks

    def chunk_paper(self, paper_data: Dict) -> Dict:
        """
        Chunk a paper document intelligently.

        Args:
            paper_data: Dictionary with 'title', 'abstract', 'content' fields

        Returns:
            Dictionary with original data plus chunks
        """
        result = paper_data.copy()

        # Don't chunk title and abstract, keep them separate
        # They will have their own embeddings

        # Chunk the main content
        if 'content' in paper_data and paper_data['content']:
            content_chunks = self.chunk_text(
                paper_data['content'],
                respect_sections=True
            )

            # Convert chunks to dictionaries
            result['content_chunks'] = [
                {
                    'text': chunk.text,
                    'chunk_index': chunk.chunk_index,
                    'token_count': chunk.token_count,
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos
                }
                for chunk in content_chunks
            ]

            log.info(f"Created {len(content_chunks)} chunks for paper {paper_data.get('paper_id', 'unknown')}")
        else:
            result['content_chunks'] = []

        return result


def main():
    """Test the document chunker"""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    # Test document
    test_doc = """
# Introduction

Natural language processing has evolved significantly over the past decade.
The introduction of transformer models marked a paradigm shift in how we approach language understanding tasks.

## Background

BERT, or Bidirectional Encoder Representations from Transformers, revolutionized NLP by introducing bidirectional training.
This approach allows the model to understand context from both directions, leading to superior performance on various tasks.

## Methodology

Our approach combines the strengths of BERT with task-specific fine-tuning.
We utilize a multi-layer architecture with attention mechanisms to capture long-range dependencies.
The model is trained on a large corpus of scientific texts.

## Results

Experimental results demonstrate significant improvements over baseline models.
We achieve state-of-the-art performance on multiple benchmarks.
    """

    # Test chunking
    chunks = chunker.chunk_text(test_doc, respect_sections=True)

    print(f"Created {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index} ({chunk.token_count} tokens):")
        print(f"  {chunk.text[:100]}...")
        print()


if __name__ == "__main__":
    main()