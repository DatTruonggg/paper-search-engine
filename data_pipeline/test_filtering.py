#!/usr/bin/env python3
"""
Test the NLP filtering functionality with sample data
"""

import pandas as pd
from arxiv_nlp_pipeline import ArxivDataPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keyword_detection():
    """Test keyword detection functionality"""
    pipeline = ArxivDataPipeline()
    
    # Test cases
    test_cases = [
        ("We present a transformer-based model for natural language processing", True),
        ("BERT achieves state-of-the-art results on question answering tasks", True),
        ("This paper introduces GPT-3, a large language model with 175B parameters", True),
        ("Our chatbot uses retrieval augmented generation for better responses", True),
        ("We propose a new machine translation system using attention mechanisms", True),
        ("This work focuses on sentiment analysis of social media text", True),
        ("Quantum computing applications in cryptography", False),
        ("Novel approaches to protein folding prediction", False),
        ("Image classification using convolutional neural networks", False)  # CV, should be False for pure keyword test
    ]
    
    logger.info("Testing keyword detection...")
    correct = 0
    for text, expected in test_cases:
        has_keywords, matched = pipeline.contains_nlp_keywords(text)
        if has_keywords == expected:
            correct += 1
            logger.info(f"✓ '{text[:50]}...' -> {has_keywords} (matched: {matched[:3]})")
        else:
            logger.error(f"✗ '{text[:50]}...' -> {has_keywords}, expected {expected}")
    
    logger.info(f"Keyword detection accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.1f}%")
    return correct == len(test_cases)

def test_filtering_with_sample_data():
    """Test filtering with sample paper data"""
    pipeline = ArxivDataPipeline()
    
    # Create sample papers
    sample_papers = [
        {
            'id': '2301.00001',
            'title': 'BERT-based Sentiment Analysis for Social Media',
            'abstract': 'We propose a BERT-based approach for sentiment analysis of social media posts. Our method achieves state-of-the-art results on Twitter sentiment classification.',
            'categories': 'cs.CL cs.LG'
        },
        {
            'id': '2301.00002', 
            'title': 'Retrieval Augmented Generation for Question Answering',
            'abstract': 'This paper presents RAG, a novel approach that combines retrieval with large language models for improved question answering performance.',
            'categories': 'cs.AI cs.IR'
        },
        {
            'id': '2301.00003',
            'title': 'Convolutional Networks for Image Classification', 
            'abstract': 'We study deep convolutional neural networks for image classification on ImageNet dataset.',
            'categories': 'cs.CV'
        },
        {
            'id': '2301.00004',
            'title': 'Quantum Algorithms for Cryptography',
            'abstract': 'Novel quantum computing approaches for breaking RSA encryption using Shor algorithm.',
            'categories': 'quant-ph cs.CR'
        },
        {
            'id': '2301.00005',
            'title': 'GPT-4 for Conversational AI Systems',
            'abstract': 'We evaluate GPT-4 performance on dialogue generation tasks and propose improvements for chatbot applications.',
            'categories': 'cs.AI cs.HC'
        }
    ]
    
    df = pd.DataFrame(sample_papers)
    logger.info(f"Created sample dataset with {len(df)} papers")
    
    # Test category-only filtering
    logger.info("\n--- Testing Category-only Filtering ---")
    cat_filtered = pipeline.filter_nlp_papers_advanced(df, use_categories=True, use_keywords=False)
    logger.info(f"Category filtering: {len(cat_filtered)}/{len(df)} papers")
    for _, paper in cat_filtered.iterrows():
        logger.info(f"  ✓ {paper['id']}: {paper['title'][:50]}... (cats: {paper['categories']})")
    
    # Test keyword-only filtering  
    logger.info("\n--- Testing Keyword-only Filtering ---")
    kw_filtered = pipeline.filter_nlp_papers_advanced(df, use_categories=False, use_keywords=True)
    logger.info(f"Keyword filtering: {len(kw_filtered)}/{len(df)} papers")
    for _, paper in kw_filtered.iterrows():
        keywords = paper.get('matched_nlp_keywords', [])
        logger.info(f"  ✓ {paper['id']}: {paper['title'][:50]}... (keywords: {keywords[:3]})")
    
    # Test combined filtering
    logger.info("\n--- Testing Combined Filtering ---")
    combined_filtered = pipeline.filter_nlp_papers_advanced(df, use_categories=True, use_keywords=True)
    logger.info(f"Combined filtering: {len(combined_filtered)}/{len(df)} papers")
    for _, paper in combined_filtered.iterrows():
        keywords = paper.get('matched_nlp_keywords', [])
        logger.info(f"  ✓ {paper['id']}: {paper['title'][:50]}... (cats: {paper['categories']}, kw: {keywords[:3]})")
    
    # Expected results
    expected_category = 4  # All except quantum (has cs.AI, cs.CL, cs.CV, cs.HC)
    expected_keyword = 3   # BERT sentiment, RAG QA, GPT-4 chat (not pure CV or quantum)
    expected_combined = 3  # Papers that pass both filters
    
    success = (len(cat_filtered) == expected_category and 
              len(kw_filtered) == expected_keyword and 
              len(combined_filtered) == expected_combined)
    
    if success:
        logger.info("✓ All filtering tests passed!")
    else:
        logger.error(f"✗ Filtering test failed: cat={len(cat_filtered)} (exp {expected_category}), "
                    f"kw={len(kw_filtered)} (exp {expected_keyword}), "
                    f"combined={len(combined_filtered)} (exp {expected_combined})")
    
    return success

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("TESTING NLP FILTERING FUNCTIONALITY")
    logger.info("="*60)
    
    # Test 1: Keyword detection
    kw_success = test_keyword_detection()
    
    logger.info("\n" + "="*60)
    
    # Test 2: Full filtering pipeline
    filter_success = test_filtering_with_sample_data()
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    if kw_success and filter_success:
        logger.info("✓ All tests passed! Filtering is working correctly.")
        return True
    else:
        logger.error("✗ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)