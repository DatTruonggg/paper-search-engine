#!/usr/bin/env python3
"""
Generate example output showing parsing and chunking for sample papers.
Creates example.json with detailed output for the 2 given XML files.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from logs import log
from data_pipeline.tei_xml_parser import TEIXMLParser
from data_pipeline.document_chunker import DocumentChunker


def generate_example_output(xml_files: list, output_file: str = "example.json"):
    """
    Generate example output showing parse and chunking results.

    Args:
        xml_files: List of XML files to process
        output_file: Output JSON file path
    """
    parser = TEIXMLParser()
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=100)

    results = []

    for xml_file in xml_files:
        log.info(f"Processing {Path(xml_file).name}")

        # Parse XML
        paper_data = parser.parse_file(xml_file)

        if not paper_data:
            log.warning(f"Failed to parse {xml_file}")
            continue

        # Chunk content
        content_chunks = chunker.chunk_text(
            paper_data['content'],
            respect_sections=True
        ) if paper_data['content'] else []

        # Build example output
        example = {
            'source_file': str(Path(xml_file).name),
            'parsed_metadata': {
                'paper_id': paper_data['paper_id'],
                'title': paper_data['title'],
                'authors': paper_data['authors'],
                'keywords': paper_data['categories'],
                'abstract': paper_data['abstract'],
                'content_length': len(paper_data['content']),
                'reference_count': len(paper_data['references'])
            },
            'chunks': [
                {
                    'chunk_index': chunk.chunk_index,
                    'token_count': chunk.token_count,
                    'text_length': len(chunk.text),
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos,
                    'text_preview': chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                    'full_text': chunk.text
                }
                for chunk in content_chunks
            ],
            'statistics': {
                'total_chunks': len(content_chunks),
                'total_tokens': sum(chunk.token_count for chunk in content_chunks),
                'avg_tokens_per_chunk': sum(chunk.token_count for chunk in content_chunks) / len(content_chunks) if content_chunks else 0,
                'min_tokens': min(chunk.token_count for chunk in content_chunks) if content_chunks else 0,
                'max_tokens': max(chunk.token_count for chunk in content_chunks) if content_chunks else 0
            }
        }

        results.append(example)

        # Log summary
        log.info(f"  Paper: {example['parsed_metadata']['title'][:60]}...")
        log.info(f"  Authors: {len(paper_data['authors'])} | Keywords: {len(paper_data['categories'])}")
        log.info(f"  Abstract: {len(paper_data['abstract'])} chars")
        log.info(f"  Content: {len(paper_data['content'])} chars")
        log.info(f"  Chunks: {len(content_chunks)} (avg {example['statistics']['avg_tokens_per_chunk']:.1f} tokens/chunk)")
        log.info("")

    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"Example output saved to: {output_path}")
    log.info(f"Processed {len(results)} papers")

    return results


def main():
    """Generate example output for the 2 sample XML files."""
    # Sample XML files
    xml_files = [
        "/Users/admin/code/cazoodle/paper-search-engine/0704.2083.grobid.tei.xml",
        "/Users/admin/code/cazoodle/paper-search-engine/0704.3662.grobid.tei.xml"
    ]

    # Check files exist
    for xml_file in xml_files:
        if not Path(xml_file).exists():
            log.error(f"File not found: {xml_file}")
            return

    # Generate example
    generate_example_output(
        xml_files,
        output_file="/Users/admin/code/cazoodle/paper-search-engine/example.json"
    )


if __name__ == "__main__":
    main()
